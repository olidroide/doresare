import pytest
import os
from unittest.mock import MagicMock, patch
from services.pipeline import generate_video
from domain.models import DetectedChord


def test_generate_video_missing_input(test_file_manager):
    """Test failure when input file is missing."""
    mock_font_manager = MagicMock()
    with pytest.raises(FileNotFoundError):
        generate_video("non_existent.mp4", file_manager=test_file_manager, cleanup=False, font_manager=mock_font_manager)

def test_generate_video_missing_manager():
    """Test failure when file_manager is missing."""
    with pytest.raises(ValueError):
        generate_video("dummy.mp4", file_manager=None, cleanup=False)
