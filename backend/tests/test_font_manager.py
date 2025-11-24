import pytest
from unittest.mock import MagicMock, patch
import pathlib
from services.font_manager import FontManager

@pytest.fixture
def font_manager(tmp_path):
    # Mock assets directory to use a temporary path
    with patch.object(FontManager, '_get_assets_dir', return_value=tmp_path / "assets"):
        fm = FontManager()
        yield fm

def test_check_resources_success(font_manager):
    # Create dummy font file
    font_manager.fonts_dir.mkdir(parents=True, exist_ok=True)
    with open(font_manager.font_path, "wb") as f:
        f.write(b"existing_font")
        
    # Mock PIL validation
    with patch('PIL.ImageFont.truetype'):
        # Should not raise exception
        font_manager.check_resources()
        assert font_manager.font_path.exists()

def test_check_resources_failure_missing(font_manager):
    # Ensure file missing
    if font_manager.font_path.exists():
        font_manager.font_path.unlink()
        
    # Should raise FileNotFoundError
    with pytest.raises(FileNotFoundError):
        font_manager.check_resources()

def test_get_font_path_success(font_manager):
    # Create dummy font file
    font_manager.fonts_dir.mkdir(parents=True, exist_ok=True)
    with open(font_manager.font_path, "wb") as f:
        f.write(b"existing_font")
        
    path = font_manager.get_font_path()
    assert str(font_manager.font_path.resolve()) == path

def test_get_font_path_failure(font_manager):
    # Ensure file missing
    if font_manager.font_path.exists():
        font_manager.font_path.unlink()
        
    with pytest.raises(FileNotFoundError):
        font_manager.get_font_path()

def test_get_pillow_font_success(font_manager):
    # Mock ImageFont.truetype to avoid actual font parsing of dummy file
    with patch('PIL.ImageFont.truetype') as mock_truetype:
        font_manager.fonts_dir.mkdir(parents=True, exist_ok=True)
        with open(font_manager.font_path, "wb") as f:
            f.write(b"fake_font")
            
        font_manager.get_pillow_font(30)
        mock_truetype.assert_called_with(str(font_manager.font_path), 30)

def test_get_pillow_font_failure(font_manager):
    # Ensure file missing
    if font_manager.font_path.exists():
        font_manager.font_path.unlink()
        
    with pytest.raises(FileNotFoundError):
        font_manager.get_pillow_font(30)
