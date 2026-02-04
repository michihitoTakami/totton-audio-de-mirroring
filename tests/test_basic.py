"""Basic test to verify pytest setup."""


def test_basic_addition() -> None:
    """Test that basic arithmetic works."""
    assert 1 + 1 == 2


def test_import() -> None:
    """Test that package can be imported."""
    import totton_audio_de_mirroring

    assert totton_audio_de_mirroring.__version__ == "0.1.0"
