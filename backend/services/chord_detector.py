from typing import List

import librosa

from domain.models import DetectedChord
from services.chord_analyzer import ChordAnalyzer


def detect_chords_chroma_improved(
    audio_file: str, sr: int = 22050
) -> List[DetectedChord]:
    """
    Improved chord detection using Chroma CQT and temporal smoothing.
    Includes prior audio cleaning (Demucs or HPSS).
    """
    try:
        # Keep this function as a thin compatibility wrapper.
        # The canonical implementation lives in services/chord_analyzer.py.
        y, sr_used = librosa.load(audio_file, sr=sr)
        analyzer = ChordAnalyzer(sr=int(sr_used))
        return analyzer.detect_chords_from_audio(y, sr=float(sr_used))
    except Exception as e:
        print(f"❌ Error in improved detection: {e}")
        return []
