---
trigger: always_on
---

# Project Rules and Instructions

## Project Overview
For detailed architecture, component specifications, and setup instructions, please refer to the [README.md](../../README.md).

**Summary**:
This project is a Ukulele Chord Video Generator. It takes a video file as input or video from youtube url, extracts audio, detects chords, and overlays the chords on the video.
- **Backend**: Python (FastAPI, Gradio, MoviePy, Librosa). Deployed on Hugging Face Spaces.
- **Frontend**: Python (FastAPI, Jinja2), HTML/CSS (Tailwind), HTMX, SSE.

## Technology Stack
- **Package Manager**: `uv` (Universal Python Package Installer).
- **Backend Framework**: FastAPI (with Gradio for internal tools/demos).
- **Frontend Framework**: FastAPI serving HTML templates.
- **Styling**: Tailwind CSS 4.
- **Interactivity**: HTMX (v2.0.8), Server-Sent Events (SSE) (htmx-ext-sse v2.2.4).
- **Deployment**: Hugging Face Spaces (Docker) or Local using uv or Home Server Intel J3455.

## Error Handling
- **Exceptions**: Use exceptions for error handling rather than returning error codes or None (unless 'None' is a valid result).
- **Fail Fast**: Validate inputs early and raise exceptions if invalid.
- **Catch Specific**: Catch specific exceptions rather than broad `Exception` where possible.
- **Error Handling**: comprehensive try/except blocks, especially in video processing and file operations.

## Testing
- **Framework**: Use `pytest` for all backend testing.
- **Fixtures**: Use `conftest.py` and fixtures for setup/teardown to avoid code duplication.
- **Mocking**: Use `unittest.mock` or `pytest-mock` to isolate units (especially for heavy services like audio extraction or video rendering).
- **Coverage**: Aim for high coverage in `domain` and `services` logic.

## Deployment
- The backend is designed to run in a Docker container on Hugging Face Spaces.
- Port 7860 is standard for HF Spaces.

## File Structure
- `backend/`: Core logic for video processing, chord detection, and API.
- `frontend/`: User-facing web application.
- `video_processing/`: Temporary directory for processing artifacts (cleaned up automatically).
