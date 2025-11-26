import os
import tempfile
import time
import logging
import sys
import re
from typing import Optional
import numpy as np
import librosa

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
                if normalized_pct > self.last_progress and self.callback:
                    self.last_progress = normalized_pct
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
    import threading
    
    result = {'output_files': None, 'error': None, 'timed_out': False}
    
    def separation_worker():
        """Worker function that performs the actual separation"""
        try:
            print(f"🧠 Starting AI separation (2 stems: vocals/instrumental)...")
            print(f"⏱️  Timeout set to {timeout_seconds} seconds")
            
            from audio_separator.separator import Separator
            
            # Configure output dir
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                separator = Separator(output_dir=output_dir, log_level=logging.WARNING)
            else:
                separator = Separator(log_level=logging.WARNING)

            # Use MDX-Net Inst model (2 stems: vocals/instrumental)
            separator.load_model(model_filename='UVR-MDX-NET-Inst_HQ_3.onnx')
            
            # Separate capturing stderr for progress
            if progress_callback:
                with TqdmProgressCapturer(progress_callback):
                    result['output_files'] = separator.separate(input_file)
            else:
                result['output_files'] = separator.separate(input_file)
                
        except Exception as e:
            result['error'] = e
    
    # Start separation in a separate thread
    worker_thread = threading.Thread(target=separation_worker, daemon=True)
    worker_thread.start()
    
    # Wait for completion or timeout
    worker_thread.join(timeout=timeout_seconds)
    
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
