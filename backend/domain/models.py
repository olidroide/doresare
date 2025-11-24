from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum
import os

# Import music theory constants (avoid circular imports if possible, or import inside methods)
# We'll import inside methods to be safe or assume they are injected/available.
# For now, we'll import at top but be careful.
from domain import music_theory

@dataclass
class DetectedChord:
    symbol: str
    start: float
    end: float
    notes: List[str]
    percentage: float

    def get_definition(self) -> List[int]:
        """Returns the chord definition (frets G C E A)."""
        return music_theory.get_chord_definition(self.symbol)

    def get_diagram_data(self) -> List[str]:
        """Returns the ASCII diagram data if available."""
        return music_theory.DIAGRAMS.get(self.symbol, [])

class AnalysisState(Enum):
    CREATED = "created"
    AUDIO_EXTRACTED = "audio_extracted"
    CHORDS_DETECTED = "chords_detected"
    VIDEO_GENERATED = "video_generated"
    FAILED = "failed"

@dataclass
class VideoAnalysis:
    """
    Aggregate Root for a video analysis session.
    Encapsulates the state of the input, output, and intermediate processing results.
    """
    input_path: str
    output_path: Optional[str] = None
    state: AnalysisState = AnalysisState.CREATED
    audio_path: Optional[str] = None
    chords: List[DetectedChord] = field(default_factory=list)
    error: Optional[str] = None

    def set_audio(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Audio file not found: {path}")
        self.audio_path = path
        self.state = AnalysisState.AUDIO_EXTRACTED

    def set_chords(self, chords: List[DetectedChord]):
        if not chords:
            raise ValueError("No chords provided")
        self.chords = chords
        self.state = AnalysisState.CHORDS_DETECTED

    def complete(self):
        if not os.path.exists(self.output_path):
            raise FileNotFoundError(f"Output video not found: {self.output_path}")
        self.state = AnalysisState.VIDEO_GENERATED

    def fail(self, error_message: str):
        self.error = error_message
        self.state = AnalysisState.FAILED
