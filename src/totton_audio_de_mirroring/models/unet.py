"""U-Net architecture for STFT magnitude masking."""

from __future__ import annotations

from typing import Literal, cast

import torch
from torch import nn

ActivationName = Literal["relu", "leaky_relu"]


class ConvBlock(nn.Module):
    """Two-layer convolutional block used in U-Net.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        activation: Activation function name.
        use_batch_norm: Whether to use batch normalization.

    Physical Basis:
        Local 2D convolutions across (frequency, time) capture structured
        mirror artifacts in the STFT magnitude while preserving locality.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: ActivationName = "leaky_relu",
        use_batch_norm: bool = True,
    ) -> None:
        super().__init__()
        self._validate_positive(in_channels, "in_channels")
        self._validate_positive(out_channels, "out_channels")

        act_layer = self._build_activation(activation)
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        ]
        if use_batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(act_layer)
        layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
        if use_batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(act_layer)

        self.net = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Apply the convolutional block.

        Args:
            features: Input feature map (batch, channels, freq, time).

        Returns:
            Output feature map with preserved spatial resolution.

        Physical Basis:
            Convolutions preserve time-frequency alignment while allowing
            the network to learn mask-relevant patterns.
        """
        return cast(torch.Tensor, self.net(features))

    @staticmethod
    def _validate_positive(value: int, name: str) -> None:
        """Validate a positive integer parameter.

        Args:
            value: Value to validate.
            name: Parameter name for error messages.

        Physical Basis:
            Valid dimensions ensure stable convolutional processing.
        """
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}.")

    @staticmethod
    def _build_activation(name: ActivationName) -> nn.Module:
        """Create activation module from name.

        Args:
            name: Activation name.

        Returns:
            Activation module.

        Physical Basis:
            Nonlinearities enable suppression masks to model complex
            mirror artifact patterns across time-frequency bins.
        """
        if name == "relu":
            return nn.ReLU(inplace=False)
        if name == "leaky_relu":
            return nn.LeakyReLU(negative_slope=0.1, inplace=False)
        raise ValueError(f"Unsupported activation: {name}.")


class DownBlock(nn.Module):
    """Downsampling block with stride-2 convolution.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels after downsampling.
        activation: Activation function name.
        use_batch_norm: Whether to use batch normalization.

    Physical Basis:
        Stride-2 downsampling increases receptive field to capture
        long-range mirror patterns across the STFT.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: ActivationName = "leaky_relu",
        use_batch_norm: bool = True,
    ) -> None:
        super().__init__()
        self.conv = ConvBlock(
            in_channels,
            in_channels,
            activation=activation,
            use_batch_norm=use_batch_norm,
        )
        self.downsample = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=4,
            stride=2,
            padding=1,
        )

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply convolution and downsampling.

        Args:
            features: Input feature map.

        Returns:
            Tuple of (skip_features, downsampled_features).

        Physical Basis:
            Skip features preserve fine structure while downsampling
            aggregates larger-scale mirror artifacts.
        """
        skip = self.conv(features)
        down = self.downsample(skip)
        return skip, down


class UpBlock(nn.Module):
    """Upsampling block with transpose convolution and skip connection.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels after upsampling.
        activation: Activation function name.
        use_batch_norm: Whether to use batch normalization.

    Physical Basis:
        Upsampling reconstructs time-frequency resolution while skip
        connections preserve low-level spectral details.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: ActivationName = "leaky_relu",
        use_batch_norm: bool = True,
    ) -> None:
        super().__init__()
        self.upsample = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self.conv = ConvBlock(
            out_channels * 2,
            out_channels,
            activation=activation,
            use_batch_norm=use_batch_norm,
        )

    def forward(self, features: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """Apply upsampling with skip connection fusion.

        Args:
            features: Input feature map.
            skip: Skip connection feature map.

        Returns:
            Output feature map at higher resolution.

        Physical Basis:
            Skip fusion ensures that suppression decisions respect
            original local spectral patterns.
        """
        up = self.upsample(features)
        if up.shape[-2:] != skip.shape[-2:]:
            raise ValueError(
                "Upsampled feature map does not match skip connection size."
            )
        merged = torch.cat([up, skip], dim=1)
        return cast(torch.Tensor, self.conv(merged))


class UNet2D(nn.Module):
    """2D U-Net for STFT magnitude masking.

    Args:
        in_channels: Number of input channels (default 1).
        out_channels: Number of output channels (default 1).
        base_channels: Base number of channels for the first level.
        num_downsamples: Number of stride-2 downsampling steps.
        channel_multiplier: Multiplier for channels per depth.
        activation: Activation function name.
        use_batch_norm: Whether to use batch normalization.
        output_activation: Output activation ("sigmoid" or None).

    Physical Basis:
        U-Net combines multi-scale context with local detail preservation,
        matching the structured nature of mirror artifacts in STFT space.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 32,
        num_downsamples: int = 4,
        channel_multiplier: int = 2,
        activation: ActivationName = "leaky_relu",
        use_batch_norm: bool = True,
        output_activation: Literal["sigmoid", "none"] = "sigmoid",
    ) -> None:
        super().__init__()
        self._validate_positive(in_channels, "in_channels")
        self._validate_positive(out_channels, "out_channels")
        self._validate_positive(base_channels, "base_channels")
        self._validate_positive(num_downsamples, "num_downsamples")
        self._validate_positive(channel_multiplier, "channel_multiplier")

        channels = [
            base_channels * (channel_multiplier**i) for i in range(num_downsamples + 1)
        ]
        self.input_conv = ConvBlock(
            in_channels,
            base_channels,
            activation=activation,
            use_batch_norm=use_batch_norm,
        )
        self.down_blocks = nn.ModuleList(
            [
                DownBlock(
                    channels[i],
                    channels[i + 1],
                    activation=activation,
                    use_batch_norm=use_batch_norm,
                )
                for i in range(num_downsamples)
            ]
        )
        self.bottleneck = ConvBlock(
            channels[-1],
            channels[-1],
            activation=activation,
            use_batch_norm=use_batch_norm,
        )
        self.up_blocks = nn.ModuleList(
            [
                UpBlock(
                    channels[i + 1],
                    channels[i],
                    activation=activation,
                    use_batch_norm=use_batch_norm,
                )
                for i in reversed(range(num_downsamples))
            ]
        )
        self.output_conv = nn.Conv2d(channels[0], out_channels, kernel_size=1)
        self.output_activation = output_activation

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Run forward pass of the U-Net.

        Args:
            features: Input tensor (batch, channels, freq, time).

        Returns:
            Output tensor with the same spatial shape as input.

        Physical Basis:
            Multi-resolution fusion enables robust suppression masks
            without altering time response in the audible band.
        """
        if features.ndim != 4:
            raise ValueError(
                f"features must be 4D (batch, channels, freq, time), got {features.ndim}D."
            )

        skips: list[torch.Tensor] = []
        x = self.input_conv(features)
        for down in self.down_blocks:
            skip, x = down(x)
            skips.append(skip)
        x = self.bottleneck(x)
        for up, skip in zip(self.up_blocks, reversed(skips), strict=True):
            x = up(x, skip)
        x = cast(torch.Tensor, self.output_conv(x))
        if self.output_activation == "sigmoid":
            return cast(torch.Tensor, torch.sigmoid(x))
        if self.output_activation == "none":
            return cast(torch.Tensor, x)
        raise ValueError(f"Unsupported output activation: {self.output_activation}.")

    @staticmethod
    def _validate_positive(value: int, name: str) -> None:
        """Validate a positive integer parameter.

        Args:
            value: Value to validate.
            name: Parameter name for error messages.

        Physical Basis:
            Valid dimensions ensure U-Net topology is well-defined.
        """
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}.")
