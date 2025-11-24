import pytest
import os
import sys
import shutil
import tempfile

# Add project root to path (parent of backend)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.file_manager import FileManager
from domain.models import VideoAnalysis, DetectedChord

@pytest.fixture
def temp_dir():
    """Creates a temporary directory for tests and cleans it up afterwards."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)

@pytest.fixture
def test_file_manager(temp_dir):
    """Returns a FileManager instance using the temporary directory."""
    return FileManager(base_dir=temp_dir)

@pytest.fixture
def sample_chords():
    """Returns a list of sample DetectedChord objects."""
    return [
        DetectedChord(symbol="C", start=0.0, end=1.0, notes=["C", "E", "G"], percentage=95.0),
        DetectedChord(symbol="G", start=1.0, end=2.0, notes=["G", "B", "D"], percentage=90.0)
    ]

@pytest.fixture
def sample_video_analysis(temp_dir, sample_chords):
    """Returns a populated VideoAnalysis object."""
    input_path = os.path.join(temp_dir, "input.mp4")
    # Create dummy input file
    with open(input_path, "w") as f:
        f.write("dummy video content")
        
    analysis = VideoAnalysis(input_path=input_path)
    analysis.output_path = os.path.join(temp_dir, "output.mp4")
    analysis.set_chords(sample_chords)
    return analysis
