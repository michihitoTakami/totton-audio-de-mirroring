---
name: abx-protocol
description: "Use to run or document CAPB ABX listening tests with frozen reference/CAPB pairs. Trigger: ABX, listening test, subjective evaluation, 試聴テスト, 主観評価."
---

# CAPB ABX Protocol

Follow `docs/abx_listening_protocol.md`. Freeze input, reference, CAPB render, config, checkpoint hash, and commit hash before the session. Loudness-match reference/CAPB within ±0.1 dB, randomize A/B/X, and store the trial log and summary under `reports/abx/<session_id>/`.

ABX supplements but never replaces canonical and held-out probe gates.
