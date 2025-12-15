---
trigger: always_on
---

## Hardware Constraints CPU HOST: Intel J3455 + 8gb RAM
Critically important rules for maintenance on this specific low-end hardware:
1.  **NO AVX**: The CPU does NOT support AVX instructions. **Do not upgrade** dependencies (especially `onnxruntime` or `audio-separator`) to versions that drop non-AVX support without verifying.
2.  **Monkey Patching is Required**: `librosa.load` MUST be patched to `sr=None` in `audio_extractor.py`, otherwise audio loading hangs for 40s.
3.  **Model Constraints**: `UVR_MDXNET_KARA_2.onnx` is the primary model. Heavier models cause timeouts.
4.  **GPU Acceleration**: OpenVINO is forced via `ONNXRUNTIME_EXECUTION_PROVIDERS`. QSV is used for video via `jellyfin-ffmpeg`. Do not revert to standard `ffmpeg`.