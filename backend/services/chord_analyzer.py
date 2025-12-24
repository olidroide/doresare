import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import librosa
import numpy as np
import scipy.ndimage

from domain.models import DetectedChord
from domain.music_theory import NOTES

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ChordDetectionConfig:
    hop_length: int = 2048
    chroma_median_filter: int = 7
    bass_root_median_filter: int = 15
    min_confidence: float = 0.6
    min_duration_seconds: float = 0.2
    root_lock_threshold: float = 0.55
    rms_threshold: float = 0.01
    time_shift_seconds: float = -0.3


class ChordAnalyzer:
    def __init__(self, sr: int = 22050, config: Optional[ChordDetectionConfig] = None):
        self.sr = sr
        self.config = config or ChordDetectionConfig()
        self._chord_names, self._template_mat = self._build_templates()

    def detect_chords_from_stems(
        self, stems: Dict[str, np.ndarray], sr: Optional[float] = None
    ) -> List[DetectedChord]:
        """Detect chords from Demucs stems.

        We only need `bass` and `other` for chord detection.
        Implementation is based on template matching (major/minor triads)
        with confidence and minimum duration filtering.

        The bass stem is used to stabilize the chord root (optional root locking).
        """
        analysis_sr = float(sr) if sr is not None else float(self.sr)

        bass_audio = stems.get("bass")
        harmony_audio = stems.get("other")
        if bass_audio is None or harmony_audio is None:
            logger.error("Missing required stems (bass/other)")
            return []

        # Compute RMS to detect silence
        # We combine bass and harmony to get a better sense of "musical" energy
        combined_audio = bass_audio + harmony_audio
        rms = librosa.feature.rms(
            y=combined_audio, hop_length=self.config.hop_length
        )[0]
        rms = scipy.ndimage.median_filter(rms, size=self.config.chroma_median_filter)

        # Compute chroma for harmony (templates)
        harmony_chroma = self._compute_chroma(harmony_audio, sr=analysis_sr)

        # Compute bass chroma (for root locking)
        bass_chroma = librosa.feature.chroma_cqt(
            y=bass_audio, sr=analysis_sr, hop_length=self.config.hop_length, n_chroma=12
        )
        bass_chroma = scipy.ndimage.median_filter(
            bass_chroma, size=(1, self.config.bass_root_median_filter)
        )

        root_notes, root_strength = self._estimate_roots(bass_chroma)

        # Frame-wise best chord selection
        similarity = self._template_mat @ harmony_chroma  # (24, T)
        best_idx, best_val = self._pick_best_chords(
            similarity, root_notes, root_strength
        )

        # Apply RMS threshold: if energy is too low, it's not a chord
        # We set confidence to 0 for these frames
        for t in range(len(best_val)):
            if rms[t] < self.config.rms_threshold:
                best_val[t] = 0.0

        # Convert to temporal events
        times = librosa.frames_to_time(
            np.arange(similarity.shape[1]),
            sr=analysis_sr,
            hop_length=self.config.hop_length,
        )
        return self._events_from_frames(times, best_idx, best_val)

    def detect_chords_from_audio(
        self, audio: np.ndarray, sr: float
    ) -> List[DetectedChord]:
        """Detect chords from a single audio track (no stems)."""
        analysis_sr = float(sr)

        # Compute RMS for silence detection
        rms = librosa.feature.rms(y=audio, hop_length=self.config.hop_length)[0]
        rms = scipy.ndimage.median_filter(rms, size=self.config.chroma_median_filter)

        chroma = self._compute_chroma(audio, sr=analysis_sr)
        similarity = self._template_mat @ chroma
        best_idx = np.argmax(similarity, axis=0)
        best_val = np.max(similarity, axis=0)

        # Apply RMS threshold
        for t in range(len(best_val)):
            if rms[t] < self.config.rms_threshold:
                best_val[t] = 0.0

        times = librosa.frames_to_time(
            np.arange(similarity.shape[1]),
            sr=analysis_sr,
            hop_length=self.config.hop_length,
        )
        return self._events_from_frames(times, best_idx, best_val)

    def _compute_chroma(self, audio: np.ndarray, sr: float) -> np.ndarray:
        chroma = librosa.feature.chroma_cqt(
            y=audio, sr=sr, hop_length=self.config.hop_length, n_chroma=12
        )
        chroma = scipy.ndimage.median_filter(
            chroma, size=(1, self.config.chroma_median_filter)
        )
        return chroma / (np.linalg.norm(chroma, axis=0) + 1e-9)

    def _build_templates(self) -> Tuple[List[str], np.ndarray]:
        templates: Dict[str, np.ndarray] = {}
        for i, note in enumerate(NOTES):
            major = np.zeros(12, dtype=np.float32)
            major[i] = 1.0
            major[(i + 4) % 12] = 0.8
            major[(i + 7) % 12] = 0.8
            templates[note] = major / (np.linalg.norm(major) + 1e-9)

            minor = np.zeros(12, dtype=np.float32)
            minor[i] = 1.0
            minor[(i + 3) % 12] = 0.8
            minor[(i + 7) % 12] = 0.8
            templates[f"{note}m"] = minor / (np.linalg.norm(minor) + 1e-9)

        chord_names = list(templates.keys())
        template_mat = np.array([templates[c] for c in chord_names], dtype=np.float32)
        return chord_names, template_mat

    def _estimate_roots(self, bass_chroma: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Root is the argmax; strength is peak-to-total ratio.
        root_notes = np.argmax(bass_chroma, axis=0)
        peak = np.max(bass_chroma, axis=0)
        total = np.sum(bass_chroma, axis=0) + 1e-9
        strength = peak / total
        return root_notes.astype(int), strength.astype(np.float32)

    def _pick_best_chords(
        self, similarity: np.ndarray, root_notes: np.ndarray, root_strength: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        # If bass root is strong enough, only consider major/minor for that root.
        best_idx = np.empty(similarity.shape[1], dtype=int)
        best_val = np.empty(similarity.shape[1], dtype=np.float32)

        for t in range(similarity.shape[1]):
            root = int(root_notes[t])
            if float(root_strength[t]) >= self.config.root_lock_threshold:
                major_idx = root * 2
                minor_idx = root * 2 + 1
                if similarity[major_idx, t] >= similarity[minor_idx, t]:
                    best_idx[t] = major_idx
                    best_val[t] = similarity[major_idx, t]
                else:
                    best_idx[t] = minor_idx
                    best_val[t] = similarity[minor_idx, t]
            else:
                best_idx[t] = int(np.argmax(similarity[:, t]))
                best_val[t] = float(np.max(similarity[:, t]))

        return best_idx, best_val

    def _events_from_frames(
        self, times: np.ndarray, best_idx: np.ndarray, best_val: np.ndarray
    ) -> List[DetectedChord]:
        detected: List[DetectedChord] = []
        if len(times) == 0:
            return detected

        current_symbol: Optional[str] = None
        start_time = float(times[0])
        segment_vals: List[float] = []

        def close_segment(end_time: float):
            nonlocal current_symbol, start_time, segment_vals
            if current_symbol is None:
                segment_vals = []
                start_time = end_time
                return

            duration = end_time - start_time
            if duration < self.config.min_duration_seconds:
                segment_vals = []
                start_time = end_time
                return

            confidence_pct = (
                float(np.median(segment_vals) * 100.0) if segment_vals else 0.0
            )
            
            # Apply time shift (negative value moves chords earlier)
            final_start = max(0.0, start_time + self.config.time_shift_seconds)
            final_end = max(0.0, end_time + self.config.time_shift_seconds)
            
            detected.append(
                DetectedChord(
                    symbol=current_symbol,
                    start=final_start,
                    end=final_end,
                    notes=[],
                    percentage=confidence_pct,
                )
            )
            segment_vals = []
            start_time = end_time

        for t, idx, val in zip(times, best_idx, best_val):
            confidence = float(val)
            symbol = (
                self._chord_names[int(idx)]
                if confidence >= self.config.min_confidence
                else None
            )

            if symbol != current_symbol:
                close_segment(float(t))
                current_symbol = symbol

            if current_symbol is not None:
                segment_vals.append(confidence)

        # Close the last segment
        if len(times) >= 2:
            last_end = float(times[-1] + (times[-1] - times[-2]))
        else:
            last_end = float(times[-1] + (self.config.hop_length / float(self.sr)))
        close_segment(last_end)

        # Post-process: Merge consecutive identical chords
        if not detected:
            return []

        merged: List[DetectedChord] = []
        for chord in detected:
            if not merged:
                merged.append(chord)
                continue

            if merged[-1].symbol == chord.symbol:
                prev = merged[-1]
                # Merge by extending end time and averaging confidence
                merged[-1] = DetectedChord(
                    symbol=prev.symbol,
                    start=prev.start,
                    end=chord.end,
                    notes=prev.notes,
                    percentage=(prev.percentage + chord.percentage) / 2.0,
                )
            else:
                merged.append(chord)

        return merged
