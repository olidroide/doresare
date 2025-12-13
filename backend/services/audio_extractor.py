import logging
import os
import re
import sys
import tempfile
import time
from typing import Optional

import librosa
import numpy as np

# --- MONKEY PATCH INIT ---
try:
    import onnxruntime as ort
    _original_inference_session = ort.InferenceSession

    class PatchedInferenceSession(_original_inference_session):
        def __init__(self, path_or_bytes, **kwargs):
            # Auto-detect OpenVINO
            available = ort.get_available_providers()
            if 'OpenVINOExecutionProvider' in available:
                # Force OpenVINO if the caller tries to default to CPU or provides nothing
                current_providers = kwargs.get('providers', [])
                if not current_providers or current_providers == ['CPUExecutionProvider']:
                    if os.path.exists('/dev/dri'):
                        print(f"🐵 MONKEY PATCH: Forcing OpenVINOExecutionProvider for model!")
                        kwargs['providers'] = ['OpenVINOExecutionProvider', 'CPUExecutionProvider']
                    else:
                        print("🐵 MONKEY PATCH: OpenVINO detected but /dev/dri missing. Falling back to CPU.")
            
            super().__init__(path_or_bytes, **kwargs)

    # Apply patch
    ort.InferenceSession = PatchedInferenceSession
    print("🐵 ONNX Runtime InferenceSession has been patched to force OpenVINO.")
except ImportError:
    pass
# -------------------------

# Global variables for singleton model
_global_separator = None
_global_separator_output_dir = None

def check_resources():
    """
    Checks if necessary resources (AI models) are available.
    """
    print("🧠 Checking AudioExtractor resources...")
    try:
        from audio_separator.separator import Separator
        print("✅ audio-separator library is available.")
        # We assume model download is handled by download_models.py at build time
    except ImportError:
        print("❌ audio-separator library NOT found. Please install dependencies.")
        raise ImportError("audio-separator not found")


class TqdmProgressCapturer:
    def __init__(self, callback):
        self.callback = callback
        self.original_stderr = sys.stderr
        self.buffer = ""
        self._pattern = re.compile(r'(\d+)%')
        self.last_progress = 0  # Track last reported progress
        self.last_logged_pct = -10  # Track last logged percentage (for printing)

    def write(self, data):
        self.original_stderr.write(data)
        self.original_stderr.flush()
        
        # Accumulate data to handle partial writes
        self.buffer += data
        
        # Try to find progress percentage
        # tqdm usually writes lines like: " 50%|██████████████              | 1/2 [00:01<00:01,  1.63s/it]"
        # We look for the last occurrence of "N%"
        matches = self._pattern.findall(self.buffer)
        if matches:
            try:
                # Take the last match as current progress
                pct = int(matches[-1])
                normalized_pct = pct / 100.0
                
                # IMPORTANT: Only report if progress increased
                # This prevents jumps when audio-separator processes multiple stems
                # (vocals at 96%, then instrumental starting at 0%)
                if normalized_pct > self.last_progress:
                    self.last_progress = normalized_pct
                    
                    # Print to stdout for Docker logs visibility (every 10%)
                    if pct >= self.last_logged_pct + 10:
                        print(f"🎵 Audio separation progress: {pct}%", flush=True)
                        self.last_logged_pct = pct
                    
                    # Call the callback
                    if self.callback:
                        self.callback(normalized_pct)
            except ValueError:
                pass
        
        # Keep buffer size manageable, just keep last 100 chars which is enough for tqdm line
        if len(self.buffer) > 100:
            self.buffer = self.buffer[-100:]

    def flush(self):
        self.original_stderr.flush()

    def __enter__(self):
        sys.stderr = self
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        sys.stderr = self.original_stderr

def extract_audio_from_video(video_file: str, output_dir: Optional[str] = None, output_path: Optional[str] = None) -> Optional[str]:
    """Extracts audio from a video file and saves it to a temporary file or output_dir/output_path."""
    try:
        from moviepy import VideoFileClip
        video = VideoFileClip(video_file)
        if video.audio is None:
            print("❌ Video has no audio.")
            return None
        
        if output_path:
            path = output_path
        elif output_dir:
            filename = f"extracted_{int(time.time())}.wav"
            path = os.path.join(output_dir, filename)
        else:
            fd, path = tempfile.mkstemp(suffix='.wav')
            os.close(fd)
            
        video.audio.write_audiofile(path, logger=None)
        video.close()
        return path
    except Exception as e:
        print(f"❌ Error extracting audio: {e}")
        return None

def load_global_model():
    """
    Initializes the global Separator instance and loads the model.
    This should be called at application startup.
    """
    global _global_separator, _global_separator_output_dir
    
    try:
        print("🧠 Loading global AI separation model...")
        from audio_separator.separator import Separator
        
        # Create a fixed temporary directory for the global separator
        # We need a persistent temp dir because the separator instance will keep writing there
        _global_separator_output_dir = os.path.join(tempfile.gettempdir(), "doresare_global_separator")
        os.makedirs(_global_separator_output_dir, exist_ok=True)
        
        # Detect available providers automatically
        detected_providers = []
        try:
            import onnxruntime as ort
            available = ort.get_available_providers()
            print(f"🔎 Available ONNX Runtime providers: {available}")
            
            # Prioritize OpenVINO if available
            if 'OpenVINOExecutionProvider' in available:
                detected_providers.append('OpenVINOExecutionProvider')
                # Check for GPU device access
                if not os.path.exists('/dev/dri'):
                    print("⚠️ OpenVINO is available but /dev/dri is missing. GPU acceleration will NOT work. Expect slow CPU performance on J3455.")
                else:
                    print("✅ /dev/dri found. OpenVINO should be able to use GPU.")
            
            # Add others if provided in env
            env_providers = os.getenv('ONNXRUNTIME_EXECUTION_PROVIDERS')
            if env_providers:
                reqs = [p.strip() for p in env_providers.split(',') if p.strip()]
                for r in reqs:
                    if r in available and r not in detected_providers:
                        detected_providers.append(r)
        except Exception as e:
            print(f"⚠️ Could not detect ONNX providers: {e}")

        # Determine log level
        log_level_str = os.getenv('AUDIO_SEPARATOR_LOG_LEVEL', 'INFO').upper()
        log_level = getattr(logging, log_level_str, logging.INFO)
        
        # Initialize separator with fixed output dir
        model_dir = os.getenv('AUDIO_SEPARATOR_MODEL_DIR', '/home/user/models')
        print(f"📂 Checking model directory: {model_dir}")
        if os.path.exists(model_dir):
            print(f"📄 Files in model dir: {os.listdir(model_dir)}")
        else:
            print(f"⚠️ Model directory does not exist: {model_dir}")

        sep_kwargs = {"output_dir": _global_separator_output_dir, "log_level": log_level, "model_file_dir": model_dir}
        
        # Use detected providers if any
        if detected_providers:
            print(f"🚀 Configuring Separator with providers: {detected_providers}")
            os.environ['ONNXRUNTIME_EXECUTION_PROVIDERS'] = ','.join(detected_providers)
            # audio-separator >= 0.17 supports 'providers' arg (not sure of exact version, but try/except handles it)
            sep_kwargs["providers"] = detected_providers

        try:
            _global_separator = Separator(**sep_kwargs)
        except TypeError:
            # Fallback if the Separator constructor signature doesn't accept 'providers'
            # We already set the env var, so just init without kwarg
            print("⚠️ Separator does not accept 'providers' kwarg. Relying on ONNXRUNTIME_EXECUTION_PROVIDERS env var.")
            # Remove providers from kwargs and retry
            if "providers" in sep_kwargs:
                del sep_kwargs["providers"]
            _global_separator = Separator(**sep_kwargs)
        
        # Validate model file existence and integrity
        # Using UVR-MDX-NET-Inst_Main.onnx (lighter)
        model_name = os.getenv('AUDIO_SEPARATOR_MODEL', 'UVR-MDX-NET-Inst_Main.onnx')
        model_path = os.path.join(model_dir, model_name)
        if os.path.exists(model_path):
            file_size = os.path.getsize(model_path)
            print(f"📊 Model file exists. Size: {file_size / 1024 / 1024:.2f} MB")
            # Calculate partial MD5 for debugging
            try:
                import hashlib
                with open(model_path, 'rb') as f:
                    # Read first 1MB for quick check or full? Let's do partial to be fast
                    file_hash = hashlib.md5(f.read(1024*1024)).hexdigest()
                    print(f"🔑 Model file partial MD5 (first 1MB): {file_hash}")
            except Exception as e:
                print(f"⚠️ Could not hash model file: {e}")
        else:
            print(f"❌ Model file MISSING at: {model_path}")

        # FORCE DEBUG LOGGING to find out why it redownloads
        print(f"🧠 Loading global AI model: {model_name} (forcing DEBUG log level)...")
        _global_separator = Separator(output_dir=_global_separator_output_dir, log_level=logging.DEBUG, model_file_dir=model_dir)
        _global_separator.load_model(model_filename=model_name)
        print("✅ Global AI separation model loaded successfully.")
        
    except Exception as e:
        print(f"❌ Failed to load global AI model: {e}")
        _global_separator = None
        # We don't raise here, so the app can still start (will fall back to per-request loading or fail later)

def separate_audio_ai(
    input_file: str, 
    output_dir: Optional[str] = None, 
    progress_callback=None,
    timeout_seconds: int = 300
) -> Optional[str]:
    """
    Separates audio using audio-separator with 2 stems model.
    Returns only the 'Instrumental' (or 'other') stem for clean chord detection.
    If output_dir is provided, saves files there.
    
    Args:
        input_file: Path to input audio file
        output_dir: Directory to save separated files
        progress_callback: Optional callback for progress updates
        timeout_seconds: Maximum time to wait for separation (default: 300s = 5 minutes)
        
    Returns:
        Path to instrumental stem, or None if separation fails/times out
    """
    import shutil
    import threading
    
    result = {'output_files': None, 'error': None, 'timed_out': False}
    separation_done = threading.Event()
    last_progress = [0.0]  # Use list to share state between threads
    
    def separation_worker():
        """Worker function that performs the actual separation"""
        try:
            print(f"🧠 Starting AI separation (2 stems: vocals/instrumental)...")
            print(f"⏱️  Timeout set to {timeout_seconds} seconds")
            
            from audio_separator.separator import Separator
            
            # Use global separator if available
            global _global_separator, _global_separator_output_dir
            
            if _global_separator:
                print("🚀 Using pre-loaded global AI model")
                separator = _global_separator
                using_global = True
            else:
                print("⚠️ Global model not loaded, creating new instance (slower)")
                # Configure output dir
                # Configure output dir
                log_level_str = os.getenv('AUDIO_SEPARATOR_LOG_LEVEL', 'INFO').upper()
                log_level = getattr(logging, log_level_str, logging.INFO)
                
                # Auto-detect providers again for per-request instance
                detected_providers = []
                try:
                    import onnxruntime as ort
                    available = ort.get_available_providers()
                    if 'OpenVINOExecutionProvider' in available:
                        detected_providers.append('OpenVINOExecutionProvider')
                    
                    env_providers = os.getenv('ONNXRUNTIME_EXECUTION_PROVIDERS')
                    if env_providers:
                        for p in env_providers.split(','):
                             p = p.strip()
                             if p and p in available and p not in detected_providers:
                                 detected_providers.append(p)
                except:
                    pass
                
                if detected_providers:
                     print(f"🚀 Configuring ONNX environment with providers: {detected_providers}")
                     os.environ['ONNXRUNTIME_EXECUTION_PROVIDERS'] = ','.join(detected_providers)

                model_dir = os.getenv('AUDIO_SEPARATOR_MODEL_DIR', '/home/user/models')
                
                if output_dir:
                    os.makedirs(output_dir, exist_ok=True)
                    sep_kwargs = {"output_dir": output_dir, "log_level": log_level, "model_file_dir": model_dir}
                else:
                    sep_kwargs = {"log_level": log_level, "model_file_dir": model_dir}

                separator = Separator(**sep_kwargs)
                
                # Load model (heavy op)
                model_name = os.getenv('AUDIO_SEPARATOR_MODEL', 'UVR-MDX-NET-Inst_Main.onnx')
                print(f"🧠 Loading specific model: {model_name}")
                separator.load_model(model_filename=model_name)
                using_global = False
            
            # Wrapper to update shared progress state with logging
            last_logged_callback_pct = [-10]  # Track last logged pct in callback
            
            def internal_progress_callback(pct):
                last_progress[0] = pct
                
                # Log every 10% for Docker visibility
                pct_int = int(pct * 100)
                if pct_int >= last_logged_callback_pct[0] + 10:
                    print(f"🎵 Audio separation progress: {pct_int}% (via callback)", flush=True)
                    last_logged_callback_pct[0] = pct_int
                
                if progress_callback:
                    progress_callback(pct)
            
            # Separate capturing stderr for progress
            print(f"⤵️ Calling separator.separate({input_file})...", flush=True)
            if progress_callback:
                with TqdmProgressCapturer(internal_progress_callback):
                    output_files = separator.separate(input_file)
            else:
                output_files = separator.separate(input_file)
            print(f"✅ separator.separate() returned: {output_files}", flush=True)
            
            # If using global separator, we need to move files to the requested output_dir
            if using_global and output_dir and output_files:
                moved_files = []
                os.makedirs(output_dir, exist_ok=True)
                
                for filename in output_files:
                    source_path = os.path.join(_global_separator_output_dir, filename)
                    dest_path = os.path.join(output_dir, filename)
                    
                    # Move file
                    if os.path.exists(source_path):
                        shutil.move(source_path, dest_path)
                        moved_files.append(filename) # Keep just filename as that's what separate returns
                    else:
                        print(f"⚠️ Could not find expected output file: {source_path}")
                
                result['output_files'] = moved_files
            else:
                result['output_files'] = output_files
                
        except Exception as e:
            result['error'] = e
        finally:
            separation_done.set()
    
    # Start separation in a separate thread
    worker_thread = threading.Thread(target=separation_worker, daemon=True)
    worker_thread.start()
    
    # Heartbeat thread to report progress every 1 seconds
    # This ensures the UI doesn't appear stuck
    def heartbeat_worker():
        """Reports progress periodically to keep the UI updating"""
        elapsed = 0
        heartbeat_interval = 1  # seconds
        
        while not separation_done.is_set() and elapsed < timeout_seconds:
            time.sleep(heartbeat_interval)
            elapsed += heartbeat_interval
            
            # If no progress from separator, simulate gradual progress
            # This prevents the UI from appearing completely stuck
            current_progress = last_progress[0]
            if progress_callback and current_progress < 1.0:
                # Gradual incremental progress to show activity
                # Max out at 95% to avoid showing 100% before completion
                estimated_progress = min(0.95, current_progress + 0.01)
                if estimated_progress > current_progress:
                    progress_callback(estimated_progress)
    
    heartbeat_thread = threading.Thread(target=heartbeat_worker, daemon=True)
    heartbeat_thread.start()
    
    # Wait for completion or timeout
    worker_thread.join(timeout=timeout_seconds)
    
    # Signal heartbeat to stop
    separation_done.set()
    
    # Check if thread is still alive (timed out)
    if worker_thread.is_alive():
        result['timed_out'] = True
        print(f"⏱️ Audio separation timed out after {timeout_seconds} seconds")
        print("⚠️ Falling back to original audio...")
        return None
    
    # Check for errors
    if result['error']:
        print(f"❌ Error in AI separation: {result['error']}")
        import traceback
        traceback.print_exc()
        print("⚠️ Falling back to original audio...")
        return None
    
    # Process output files
    output_files = result['output_files']
    if not output_files:
        print("⚠️ No separated files generated")
        return None
        
    print(f"📂 Separated files generated: {output_files}")
    
    # Find instrumental stem
    target_stem = None
    
    for file in output_files:
        # If output_dir was set, audio-separator returns only filenames, not full paths (sometimes)
        # Build full path
        full_path = os.path.join(output_dir, file) if output_dir else file
            
        if "Instrumental" in file or "other" in file:
            target_stem = full_path
            break
    
    if target_stem:
        print(f"✅ Instrumental stem found: {target_stem}")
        return target_stem
        
    # If explicit not found, return second (assuming first is vocals)
    if len(output_files) > 1:
        # Assume the one that is NOT vocals is the good one
        for file in output_files:
            if "Vocals" not in file:
                if output_dir:
                    return os.path.join(output_dir, file)
                return file
                
    # Fallback
    if output_files:
        first = output_files[0]
        if output_dir:
            return os.path.join(output_dir, first)
        return first
    
    print("⚠️ No separated files generated")
    return None


def clean_audio_for_chords(y: np.ndarray, sr: int) -> np.ndarray:
    """
    Cleans audio to highlight harmonic instruments (strings) and remove percussion.
    Uses harmonic-percussive separation with adjusted margins and high-pass filter.
    """
    # Harmonic/percussive separation
    # margin > 1 forces stricter separation.
    # We want harmonics (strings), so we use a high margin for percussion so it doesn't leak.
    y_harm, y_perc = librosa.effects.hpss(y, margin=(1.0, 5.0))
    
    # High-pass filter to remove very low frequency noise (hum, low bumps)
    # Ukulele is usually above 200Hz (G4 is 392Hz, C4 is 261Hz, low G is 196Hz)
    # Guitar goes down to E2 (82Hz). Bass E1 (41Hz).
    # If we want "uke, guitar and bass", we need from 40Hz.
    # Filter below 40Hz to clean useless sub-bass.
    
    from scipy.signal import butter, filtfilt
    
    # Design high-pass Butterworth filter
    # Cutoff at 40Hz, order 5 (steeper rolloff)
    nyquist = sr / 2
    cutoff = 40.0
    order = 5
    
    # Normalize cutoff frequency
    normal_cutoff = cutoff / nyquist
    
    # Get filter coefficients
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    
    # Apply filter (filtfilt applies filter forward and backward to avoid phase distortion)
    y_filtered = filtfilt(b, a, y_harm)
    
    return y_filtered


def separate_with_openvino_wrapper(input_file: str, output_dir: Optional[str] = None, model_path: Optional[str] = None, chunk_duration: int = 30) -> Optional[str]:
    """Use the OpenVINOAudioSeparator wrapper to separate audio and return instrumental path.

    This helper will copy the input into `output_dir` (if provided) and run the wrapper.
    Returns path to instrumental stem or None on failure.
    """
    try:
        import shutil
        from infrastructure.audio_separation.openvino_separator import OpenVINOAudioSeparator
        
        # Prepare working input path inside output_dir so outputs are created nearby
        # Prepare working input path inside output_dir so outputs are created nearby
        work_input = input_file
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            candidate_work_input = os.path.join(output_dir, os.path.basename(input_file))
            if os.path.abspath(input_file) != os.path.abspath(candidate_work_input):
                 shutil.copy2(input_file, candidate_work_input)
                 work_input = candidate_work_input
            else:
                 work_input = input_file

        # Respect the single-use env var `USE_OPENVINO` which enables OpenVINO behavior
        model = model_path or os.getenv('OPENVINO_MODEL_PATH') or os.getenv('OPENVINO_MODEL', None)
        if not model:
            # Default to converted model location inside image
            model = os.path.join('/app/models_openvino', os.getenv('AUDIO_SEPARATOR_MODEL', 'UVR-MDX-NET-Inst_Main.onnx').rsplit('.', 1)[0] + '.xml')
        device = os.getenv('OPENVINO_DEVICE', 'CPU')
        precision = os.getenv('OPENVINO_PRECISION', 'FP16')

        sep = OpenVINOAudioSeparator(model_path=model, device=device, precision=precision)
        outputs = sep.separate(work_input, chunk_duration=chunk_duration)

        inst = outputs.get('instrumental') or outputs.get('inst') or outputs.get('other')
        if not inst:
            # If keys are different, pick the one that contains 'inst' or not 'voc'
            for v in outputs.values():
                pn = str(v)
                if 'voc' not in pn.lower():
                    inst = pn
                    break

        if inst and output_dir:
            # Ensure path is absolute in output_dir
            inst_path = os.path.join(output_dir, os.path.basename(inst))
            if os.path.exists(inst) and os.path.abspath(os.path.dirname(inst)) != os.path.abspath(output_dir):
                shutil.move(inst, inst_path)
            else:
                inst_path = inst
            return inst_path

        return inst
    except Exception as e:
        print(f"⚠️ OpenVINO wrapper separation failed: {e}")
        return None
