import pytest
import os
from unittest.mock import MagicMock, patch
from services.pipeline import generate_video
from domain.models import DetectedChord

@patch("services.pipeline.audio_extractor")
@patch("services.pipeline.chord_detector")
@patch("services.pipeline.video_renderer")
def test_generate_video_success(mock_renderer, mock_detector, mock_extractor, test_file_manager, temp_dir):
    """Test successful video generation flow."""
    
    # Create dummy audio file
    audio_file = os.path.join(temp_dir, "audio.wav")
    with open(audio_file, "w") as f:
        f.write("dummy audio")
    
    # Setup Mocks
    mock_extractor.extract_audio_from_video.return_value = audio_file
    mock_extractor.separate_audio_ai.return_value = None # No separation
    
    mock_detector.detect_chords_chroma_improved.return_value = [
        DetectedChord("C", 0, 1, [], 100)
    ]
    
    # Create dummy input file
    input_file = os.path.join(temp_dir, "input.mp4")
    with open(input_file, "w") as f:
        f.write("dummy")
        
    # We need to ensure the output file is "created" by the renderer mock
    def side_effect_render(analysis, **kwargs):
        # Simulate file creation
        with open(analysis.output_path, "w") as f:
            f.write("video content")
            
    mock_renderer.render_video_with_overlays.side_effect = side_effect_render

    # Create mock FontManager
    mock_font_manager = MagicMock()

    # Run Pipeline with cleanup disabled for testing (uses essentia by default)
    output_path = generate_video(input_file, file_manager=test_file_manager, cleanup=False, font_manager=mock_font_manager)
    
    # Assertions
    # Since path contains timestamp, we check if it exists and ends with .mp4
    assert os.path.exists(output_path)
    assert output_path.endswith(".mp4")
    assert "process_" in output_path
    
    # Verify calls
    mock_extractor.extract_audio_from_video.assert_called_once()
    mock_detector.detect_chords_chroma_improved.assert_called_once()
    mock_renderer.render_video_with_overlays.assert_called_once()

def test_generate_video_missing_input(test_file_manager):
    """Test failure when input file is missing."""
    mock_font_manager = MagicMock()
    with pytest.raises(FileNotFoundError):
        generate_video("non_existent.mp4", file_manager=test_file_manager, cleanup=False, font_manager=mock_font_manager)

def test_generate_video_missing_manager():
    """Test failure when file_manager is missing."""
    with pytest.raises(ValueError):
        generate_video("dummy.mp4", file_manager=None, cleanup=False)
