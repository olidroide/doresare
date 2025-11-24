# Musical Note and Chord Extractor

This project is a web application to detect chords in music videos and generate a new video with the chords overlaid.

## Architecture

The project is divided into two main components:

1.  **Backend (Python/Gradio)**:
    *   Handles heavy processing: audio extraction, source separation (AI), chord detection, and video rendering.
    *   Exposes an API via Gradio Client.
    *   Designed to be deployed on Hugging Face Spaces (Docker).

2.  **Frontend (FastAPI/Jinja2/HTMX)**:
    *   Modern and reactive web user interface.
    *   Communicates with the backend to submit jobs and receive real-time status updates via SSE (Server-Sent Events).
    *   Can run locally or on any cloud server.

### Architecture Diagram

```mermaid
graph TD
    User[User] -->|Uploads Video| Frontend[Frontend (FastAPI + HTMX)]
    Frontend -->|Sends Job (Async)| Backend[Backend (Gradio + Docker)]
    Backend -->|Processes Video| Processing[Processing (Librosa/MoviePy)]
    Processing -->|Generates Video| Backend
    Backend -->|Status/Result| Frontend
    Frontend -->|SSE Updates| User
    Frontend -->|Final Video| User
```

## Project Structure

*   `backend/`: Processing server code (Gradio).
*   `frontend/`: Web interface code (FastAPI).

## Local Execution

### Prerequisites

*   Python 3.10+
*   `uv` (Package Manager)
*   `ffmpeg` installed on the system.

### Backend

```bash
cd backend
uv sync
uv run app.py
```
The backend will run at `http://localhost:7860`.

### Frontend

```bash
cd frontend
uv sync
uv run uvicorn main:app --reload
```
The frontend will run at `http://localhost:8000`.

## Deployment

*   **Backend**: Build the Docker image in `backend/` and deploy to Hugging Face Spaces.
*   **Frontend**: Configure `HF_SPACE` and `HF_TOKEN` environment variables to point to the deployed backend.
