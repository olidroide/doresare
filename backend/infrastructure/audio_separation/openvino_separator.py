from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict

import numpy as np

logger = logging.getLogger(__name__)


class AudioSeparator(ABC):
    @abstractmethod
    def separate(self, audio_path: str, chunk_duration: int = 30) -> Dict[str, str]:
        """Separates an audio file into stems and returns paths to generated files."""


class OpenVINOAudioSeparator(AudioSeparator):
    def __init__(
        self,
        model_path: str,
        device: str = "CPU",
        precision: str = "FP16",
        num_streams: str | int = "AUTO",
    ) -> None:
        try:
            from openvino.runtime import Core
        except Exception as exc:  # pragma: no cover - import path depends on environment
            raise ImportError("OpenVINO runtime is required for OpenVINOAudioSeparator") from exc

        self.core = Core()
        self.model_path = model_path
        self.device = device
        self.precision = precision

        config = {
            "DEVICE_TYPE": device,
            "PRECISION": precision,
            "NUM_STREAMS": str(num_streams),
        }

        self.model = self.core.read_model(model=model_path)
        self.compiled = self.core.compile_model(self.model, device, config)
        self.infer_request = self.compiled.create_infer_request()
        logger.info("OpenVINOAudioSeparator initialized: %s %s", device, precision)

    def separate(self, audio_path: str, chunk_duration: int = 30) -> Dict[str, str]:
        import librosa
        import soundfile as sf

        y, sr = librosa.load(audio_path, sr=44100, mono=True)

        vocals = np.zeros_like(y)
        instrumental = np.zeros_like(y)

        chunk_samples = int(chunk_duration * sr)

        for i in range(0, len(y), chunk_samples):
            chunk = y[i : i + chunk_samples]
            if chunk.size == 0:
                continue

            # Normalize chunk
            maxv = float(np.abs(chunk).max())
            if maxv > 0:
                norm_chunk = chunk / maxv
            else:
                norm_chunk = chunk

            # Prepare input tensor shape expected by the model.
            # Many ONNX U-Net like models expect (1, 1, N)
            input_tensor = norm_chunk[np.newaxis, np.newaxis, :].astype(np.float32)

            # Run inference. The compiled infer_request may return a dict of outputs.
            outputs = self.infer_request.infer({0: input_tensor})

            # Try to extract vocals/instrumental from outputs
            if isinstance(outputs, dict):
                # Heuristic: look for keys containing 'voc' and 'inst' or 'other'
                voc = None
                inst = None
                for k, v in outputs.items():
                    kn = k.lower()
                    if 'voc' in kn:
                        voc = np.asarray(v).ravel()
                    elif 'inst' in kn or 'other' in kn:
                        inst = np.asarray(v).ravel()

                # Fallback: if only two outputs, take first as vocals, second as instrumental
                if voc is None and inst is None and len(outputs) >= 2:
                    vals = list(outputs.values())
                    voc = np.asarray(vals[0]).ravel()
                    inst = np.asarray(vals[1]).ravel()

                if voc is not None:
                    vocals[i : i + len(voc)] = voc[: len(chunk)]
                if inst is not None:
                    instrumental[i : i + len(inst)] = inst[: len(chunk)]
            else:
                # If outputs is array-like, assume it's instrumental and place into instrumental
                arr = np.asarray(outputs).ravel()
                instrumental[i : i + len(arr)] = arr[: len(chunk)]

        base = Path(audio_path).with_suffix("")
        vocal_path = str(base) + "_vocals.wav"
        inst_path = str(base) + "_instrumental.wav"

        sf.write(vocal_path, vocals, sr)
        sf.write(inst_path, instrumental, sr)

        return {"vocals": vocal_path, "instrumental": inst_path}
