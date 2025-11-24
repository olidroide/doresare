import os
import time
import shutil
import tempfile
import logging
from typing import List, Optional

class FileManager:
    def __init__(self, base_dir: str = "video_processing"):
        self.base_dir = os.path.abspath(base_dir)
        self.ensure_directory(self.base_dir)
        self.logger = logging.getLogger(__name__)

    def ensure_directory(self, path: str):
        """Ensures that a directory exists."""
        os.makedirs(path, exist_ok=True)

    def get_output_path(self, filename: str, directory: Optional[str] = None) -> str:
        """Generates a full path for an output file."""
        target_dir = directory if directory else self.base_dir
        self.ensure_directory(target_dir)
        return os.path.join(target_dir, filename)

    def create_unique_filename(self, prefix: str = "video", extension: str = "mp4") -> str:
        """Generates a unique filename based on timestamp."""
        return f"{prefix}_{int(time.time())}.{extension}"

    def create_temp_file(self, suffix: str = ".tmp", directory: Optional[str] = None) -> str:
        """Creates a temporary file and returns its path."""
        target_dir = directory if directory else self.base_dir
        self.ensure_directory(target_dir)
        fd, path = tempfile.mkstemp(suffix=suffix, dir=target_dir)
        os.close(fd)
        return path

    def cleanup_old_files(self, max_age_seconds: int = 3600, directory: Optional[str] = None):
        """Removes files older than max_age_seconds in the specified directory."""
        target_dir = directory if directory else self.base_dir
        if not os.path.exists(target_dir):
            return

        current_time = time.time()
        count = 0
        try:
            for f in os.listdir(target_dir):
                f_path = os.path.join(target_dir, f)
                if os.path.isfile(f_path):
                    if current_time - os.path.getmtime(f_path) > max_age_seconds:
                        os.remove(f_path)
                        count += 1
            if count > 0:
                print(f"🧹 FileManager: Removed {count} old files from {target_dir}")
        except Exception as e:
            print(f"⚠️ FileManager: Cleanup error: {e}")

    def cleanup_files(self, file_paths: List[str]):
        """Removes a list of specific files."""
        print("🧹 FileManager: Cleaning up temporary files...")
        for f in file_paths:
            if f and os.path.exists(f):
                try:
                    os.remove(f)
                    print(f"   - Removed: {f}")
                except Exception as e:
                    print(f"   ⚠️ Could not remove {f}: {e}")

    def cleanup_directory(self, directory: Optional[str] = None, pattern: str = "*") -> int:
        """
        Removes all files matching pattern from the specified directory.
        Returns count of removed files.
        
        Args:
            directory: Target directory to clean. Defaults to base_dir.
            pattern: File pattern to match (currently unused, cleans all files).
        
        Returns:
            Number of files removed.
        """
        target_dir = directory if directory else self.base_dir
        if not os.path.exists(target_dir):
            return 0
        
        count = 0
        try:
            for f in os.listdir(target_dir):
                f_path = os.path.join(target_dir, f)
                if os.path.isfile(f_path):
                    os.remove(f_path)
                    count += 1
            if count > 0:
                print(f"🧹 FileManager: Removed {count} files from {target_dir}")
        except Exception as e:
            print(f"⚠️ FileManager: Cleanup error: {e}")
        
        return count

# Global instance for easy access if needed, though dependency injection is preferred
default_file_manager = FileManager()
