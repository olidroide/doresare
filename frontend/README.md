# Note Extractor - Frontend

Web user interface for the note extractor.

## Technologies

*   **FastAPI**: Web server.
*   **Jinja2**: Template engine.
*   **HTMX**: Interactivity and real-time updates (SSE).
*   **TailwindCSS**: Styling.

## Configuration

Environment variables:

*   `ENV`: Environment mode. Options: `LOCAL`, `DEV`, `PROD`. Default: `LOCAL`.
*   `HF_SPACE`: Hugging Face Space name (e.g., `user/space`) if `ENV` is `PROD`.
*   `HF_TOKEN`: Hugging Face Token (optional, if the space is private).

## Execution

1.  Copy `.env.example` to `.env`:
    ```bash
    cp .env.example .env
    ```
2.  Run the server:
    ```bash
    uv sync
    uv run uvicorn main:app --reload
    ```
Access at `http://localhost:8000`.
