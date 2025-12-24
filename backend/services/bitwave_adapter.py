import logging
import os
import ssl
from pathlib import Path
from typing import Dict

import demucs.separate
import librosa
import numpy as np
import torch

logger = logging.getLogger(__name__)


class BitwaveAdapter:
    def __init__(self):
        """
        Initialize BitwaveAdapter using Demucs directly.
        """
        self.model_dir = os.getenv("AUDIO_SEPARATOR_MODEL_DIR", "models")
        # Ensure model dir exists
        os.makedirs(self.model_dir, exist_ok=True)

        # Bypass SSL for model downloads
        ssl._create_default_https_context = ssl._create_unverified_context

        # Set TORCH_HOME to our models directory so demucs downloads there
        os.environ["TORCH_HOME"] = os.path.abspath(self.model_dir)

        # Detect device
        self.device = "cpu"
        if torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        logger.info(f"BitwaveAdapter using device: {self.device}")

    async def separate_audio(
        self, audio_path: str, model_name: str = "htdemucs_ft"
    ) -> tuple[Dict[str, np.ndarray], float]:
        """
        Separates the audio into stems using Demucs.
        Optimized to only load required stems (bass/other).
        """
        logger.info(
            f"Separating audio using Demucs {model_name} on {self.device}: {audio_path}"
        )

        stems = {}
        sr = 22050.0
        work_dir = os.path.dirname(audio_path)

        output_dir = os.path.join(work_dir, "demucs_out")
        os.makedirs(output_dir, exist_ok=True)

        try:
            # Run demucs separation with hardware acceleration
            # --device: use mps/cuda if available
            # --jobs: use multiple CPU cores for pre/post processing
            cmd = [
                "-n",
                model_name,
                "-o",
                output_dir,
                "--device",
                self.device,
                audio_path,
            ]
            demucs.separate.main(cmd)

            filename_no_ext = Path(audio_path).stem
            model_out_dir = os.path.join(output_dir, model_name, filename_no_ext)

            if not os.path.exists(model_out_dir):
                base_out = os.path.join(output_dir, model_name)
                if os.path.exists(base_out):
                    dirs = [
                        d
                        for d in os.listdir(base_out)
                        if os.path.isdir(os.path.join(base_out, d))
                    ]
                    if dirs:
                        model_out_dir = os.path.join(base_out, dirs[0])

            if not os.path.exists(model_out_dir):
                raise Exception(f"Demucs output directory not found: {model_out_dir}")

            # Define which stems we actually need to load to save time/RAM
            needed_stems = ["bass", "other", "guitar", "piano"]

            for f in os.listdir(model_out_dir):
                full_path = os.path.join(model_out_dir, f)
                f_lower = f.lower()

                # Identify stem type
                stem_type = None
                if "bass" in f_lower:
                    stem_type = "bass"
                elif "other" in f_lower:
                    stem_type = "other"
                elif "guitar" in f_lower:
                    stem_type = "guitar"
                elif "piano" in f_lower:
                    stem_type = "piano"

                # ONLY load if it's a stem we need
                if stem_type and stem_type in needed_stems:
                    y, native_sr = librosa.load(full_path, sr=22050)
                    stems[stem_type] = y
                    sr = float(native_sr)

            # Post-processing for 6-stem model (combine guitar/piano into 'other')
            if "bass" in stems and "other" not in stems:
                if "guitar" in stems and "piano" in stems:
                    min_len = min(len(stems["guitar"]), len(stems["piano"]))
                    stems["other"] = (
                        stems["guitar"][:min_len] + stems["piano"][:min_len]
                    )
                elif "guitar" in stems:
                    stems["other"] = stems["guitar"]
                elif "piano" in stems:
                    stems["other"] = stems["piano"]

            if "bass" not in stems or "other" not in stems:
                raise Exception(
                    f"Failed to extract required stems (bass/other) using {model_name}"
                )

            # Cleanup demucs output directory
            import shutil

            try:
                shutil.rmtree(output_dir)
            except Exception:
                pass

        except Exception as e:
            logger.error(f"Error during {model_name} separation: {e}")
            raise

        return stems, sr
