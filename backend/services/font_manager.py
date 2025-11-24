import os
import pathlib
import pathlib
from PIL import ImageFont

class FontManager:
    def __init__(self):
        # User requested strict usage of provided Inter font
        self.font_name = "Inter-VariableFont_opsz,wght.ttf"
        self.assets_dir = self._get_assets_dir()
        self.fonts_dir = self.assets_dir / "fonts"
        
        # Strict local path only
        self.font_candidates = [
            self.fonts_dir / self.font_name
        ]
        self.font_path = self._find_font()

    def _find_font(self) -> pathlib.Path:
        for path in self.font_candidates:
            if path.exists():
                return path
        
        # If missing, we return the expected path so check_resources can warn about it
        return self.fonts_dir / self.font_name
        
    def _get_assets_dir(self):
        # backend/services/font_manager.py -> backend/assets
        current_dir = pathlib.Path(__file__).parent.resolve()
        return current_dir.parent / "assets"

    def check_resources(self):
        """Checks if font exists. Raises error if missing."""
        print("🎨 Checking FontManager resources...")
        self.fonts_dir.mkdir(parents=True, exist_ok=True)
        
        if not self.font_path.exists():
            msg = f"❌ CRITICAL: Font not found at {self.font_path}. Please ensure 'assets/fonts/{self.font_name}' exists."
            print(msg)
            raise FileNotFoundError(msg)
        else:
            # Verify existing file
            try:
                ImageFont.truetype(str(self.font_path), 10)
                print(f"✅ Font found and verified: {self.font_path}")
            except Exception as e:
                msg = f"❌ CRITICAL: Existing font is corrupted: {e}"
                print(msg)
                raise ValueError(msg)

    def get_font_path(self) -> str:
        """Returns the absolute path to the font file for MoviePy."""
        if self.font_path.exists():
            return str(self.font_path.resolve())
        raise FileNotFoundError(f"Font not found: {self.font_path}")

    def get_pillow_font(self, size: int) -> ImageFont.FreeTypeFont:
        """Returns a PIL ImageFont object."""
        if self.font_path.exists():
            return ImageFont.truetype(str(self.font_path), size)
        raise FileNotFoundError(f"Font not found: {self.font_path}")

# Default instance for backward compatibility if needed, 
# but we should use dependency injection.
default_font_manager = FontManager()
