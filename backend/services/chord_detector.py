import math
import logging
import os
import tempfile
from typing import Dict, List, Any, Tuple, Optional
from collections import Counter
import numpy as np
import librosa
import scipy.ndimage

from domain.models import DetectedChord
from domain.music_theory import NOTES

def detect_chords_chroma_improved(audio_file: str, sr: int = 22050) -> List[DetectedChord]:
    """
    Improved chord detection using Chroma CQT and temporal smoothing.
    Includes prior audio cleaning (Demucs or HPSS).
    """
    try:
        # 1. Use provided audio directly
        # Assumes caller (generate_video.py) already handled separation if needed.
        audio_to_process = audio_file
        
        # Determine if HPSS cleaning is needed
        # If file has "Instrumental" or "other" in name, assume it's a clean stem
        if "Instrumental" in audio_file or "other" in audio_file:
            print("✨ Using pre-separated audio (stem).")
            use_hpss = False
        else:
            print("⚠️ Using standard audio (applying HPSS cleaning).")
            use_hpss = True
            
        y, sr_used = librosa.load(audio_to_process, sr=sr)
        
        # 2. Additional cleaning if not using Demucs
        if use_hpss:
            from services.audio_extractor import clean_audio_for_chords
            y_clean = clean_audio_for_chords(y, sr_used)
        else:
            y_clean = y # Already 'other', should not have percussion
        
        # 3. Calculate CQT Chroma (better resolution for music)
        # Larger hop length for smoothing (approx 0.1s)
        hop_length = 2048 
        chroma = librosa.feature.chroma_cqt(y=y_clean, sr=sr_used, hop_length=hop_length, n_chroma=12)
        
        # 4. Temporal smoothing (median filter)
        # Smooth over ~0.5 seconds (5 frames approx if hop=2048 @ 22050 -> ~0.1s/frame)
        chroma_smooth = scipy.ndimage.median_filter(chroma, size=(1, 9))
        
        # 4. Matching with templates
        # Simple Major and Minor templates
        templates = {}
        for i, note in enumerate(NOTES):
            # Major: 0, 4, 7
            vec = np.zeros(12)
            vec[i] = 1.0
            vec[(i+4)%12] = 0.8
            vec[(i+7)%12] = 0.8
            templates[note] = vec / np.linalg.norm(vec)
            
            # Minor: 0, 3, 7
            vec = np.zeros(12)
            vec[i] = 1.0
            vec[(i+3)%12] = 0.8
            vec[(i+7)%12] = 0.8
            templates[note+'m'] = vec / np.linalg.norm(vec)
            
        # Calculate similarity
        best_score = -1
        best_chords = []
        
        chord_names = list(templates.keys())
        template_mat = np.array([templates[c] for c in chord_names]) # (24, 12)
        
        # Normalize chroma
        chroma_norm = chroma_smooth / (np.linalg.norm(chroma_smooth, axis=0) + 1e-9)
        
        # Dot product (Cosine Similarity)
        similarity = np.dot(template_mat, chroma_norm) # (24, T)
        
        # Get best chord per frame
        max_idx = np.argmax(similarity, axis=0)
        max_val = np.max(similarity, axis=0)
        
        # 5. Convert to temporal events
        times = librosa.frames_to_time(np.arange(similarity.shape[1]), sr=sr_used, hop_length=hop_length)
        
        detected_chords = []
        current_chord = None
        start_time = 0.0
        
        # Confidence threshold
        MIN_CONFIDENCE = 0.6
        MIN_DURATION = 0.5 # seconds
        
        for t, idx, val in zip(times, max_idx, max_val):
            chord = chord_names[idx]
            
            if val < MIN_CONFIDENCE:
                chord = None # Silence or uncertainty
            
            if chord != current_chord:
                # Close previous
                if current_chord is not None:
                    duration = t - start_time
                    if duration >= MIN_DURATION:
                        detected_chords.append(DetectedChord(
                            symbol=current_chord,
                            start=start_time,
                            end=t,
                            notes=[], # We don't calculate individual notes here
                            percentage=float(val)*100
                        ))
                
                current_chord = chord
                start_time = t
                
        # Close last
        if current_chord is not None:
            detected_chords.append(DetectedChord(
                symbol=current_chord,
                start=start_time,
                end=times[-1],
                notes=[],
                percentage=0.0
            ))
            
        return detected_chords
        
    except Exception as e:
        print(f"❌ Error in improved detection: {e}")
        return []
