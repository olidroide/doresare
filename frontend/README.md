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

## Frontend YouTube Download Option

The frontend can optionally download YouTube (and other) videos locally before uploading them to the backend. Behavior:

- The download prefers single-file `mp4` formats up to `720p` to avoid needing `ffmpeg`.
- If the selected video requires stream merging (ffmpeg), the frontend will surface an error and advise rebuilding the Docker image with ffmpeg support.

To enable ffmpeg at build time (only if strictly necessary), build the frontend image with the build-arg `INSTALL_JELLYFIN_FFMPEG=true`:

```bash
docker build --build-arg INSTALL_JELLYFIN_FFMPEG=true -t doresare-frontend .
```

By default the image does not include ffmpeg to keep the runtime lightweight for home servers.

## Troubleshooting: SSL / yt-dlp errors

Sometimes `yt-dlp` fails on some systems (notably macOS) with SSL certificate verification errors like:

```bash
[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: unable to get local issuer certificate
```

Recommended fixes (choose one):

* Install and use `certifi` in the Python environment and/or run the macOS "Install Certificates.command" that comes with some Python installers.
* Ensure your system root certificates are up-to-date.
* Update `yt-dlp` to the latest version: `yt-dlp -U`.

The frontend includes a best-effort fallback: if `yt-dlp` fails due to SSL verification, it will retry the download with certificate checking disabled. If a requested format is not available it will also retry with relaxed format filters and finally with the unconstrained `best` format. When these fallbacks run, the server will log a sample of available formats to help debugging.

If you prefer not to use the fallback behavior, update `use_cases/download_video.py` to remove the fallback logic and surface the original `yt-dlp` error.
