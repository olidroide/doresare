import os
import time
import pytest

def test_ensure_directory(test_file_manager):
    """Test directory creation."""
    new_dir = os.path.join(test_file_manager.base_dir, "subdir")
    test_file_manager.ensure_directory(new_dir)
    assert os.path.exists(new_dir)
    assert os.path.isdir(new_dir)

def test_create_unique_filename(test_file_manager):
    """Test unique filename generation."""
    name1 = test_file_manager.create_unique_filename("test", "txt")
    time.sleep(1.01)  # Need at least 1 second since filename uses int(time.time())
    name2 = test_file_manager.create_unique_filename("test", "txt")
    
    assert name1.startswith("test_")
    assert name1.endswith(".txt")
    assert name1 != name2

def test_get_output_path(test_file_manager):
    """Test output path resolution."""
    filename = "test.txt"
    path = test_file_manager.get_output_path(filename)
    expected = os.path.join(test_file_manager.base_dir, filename)
    assert path == expected

def test_cleanup_old_files(test_file_manager):
    """Test cleanup of old files."""
    # Create an old file
    old_file = os.path.join(test_file_manager.base_dir, "old.txt")
    with open(old_file, "w") as f:
        f.write("content")
    
    # Manually set mtime to 2 hours ago
    two_hours_ago = time.time() - 7200
    os.utime(old_file, (two_hours_ago, two_hours_ago))
    
    # Create a new file
    new_file = os.path.join(test_file_manager.base_dir, "new.txt")
    with open(new_file, "w") as f:
        f.write("content")
        
    # Run cleanup (max age 1 hour)
    test_file_manager.cleanup_old_files(max_age_seconds=3600)
    
    assert not os.path.exists(old_file)
    assert os.path.exists(new_file)

def test_cleanup_directory(test_file_manager):
    """Test cleanup of all files in a directory."""
    # Create multiple files
    file1 = os.path.join(test_file_manager.base_dir, "file1.txt")
    file2 = os.path.join(test_file_manager.base_dir, "file2.mp4")
    file3 = os.path.join(test_file_manager.base_dir, "file3.wav")
    
    for f in [file1, file2, file3]:
        with open(f, "w") as fp:
            fp.write("content")
    
    # Verify files exist
    assert os.path.exists(file1)
    assert os.path.exists(file2)
    assert os.path.exists(file3)
    
    # Cleanup directory
    count = test_file_manager.cleanup_directory()
    
    # Verify all files removed
    assert count == 3
    assert not os.path.exists(file1)
    assert not os.path.exists(file2)
    assert not os.path.exists(file3)

