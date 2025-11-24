---
title: Ukulele Chord Video Generator
emoji: 🎸
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# 🎸 Ukulele Chord Video Generator

Automatically detect and overlay chords on your music videos using AI-powered audio analysis.

## Features

- 🎵 **Audio Extraction**: Extract audio from video files
- 🧠 **AI Vocal Separation**: Separate vocals from instruments for better chord detection
- 🎼 **Chord Detection**: Advanced audio analysis using librosa and chroma features
- 🎥 **Video Rendering**: Beautiful chord diagrams and timeline overlays
- 🎨 **Real-time Timeline**: Scrolling chord timeline for easy following
- 🐳 **Dockerized**: Fully containerized for easy deployment

## Architecture & Technologies

- **Backend**: Python (FastAPI/Gradio)
- **Package Manager**: `uv` (Universal Python Package Installer)
- **Audio Processing**: `librosa`, `audio-separator` (UVR MDX-Net)
- **Video Processing**: `MoviePy`
- **Deployment**: Docker (optimized for Hugging Face Spaces Free Tier)

## Setup & Installation

### Prerequisites
- Docker (Desktop or Colima)
- `uv` (optional, for local dev without Docker)

### Local Development (Docker - Recommended)

1. **Clone the repository**:
   ```bash
   git clone <your-repo>
   cd backend
   ```

2. **Run with Docker Compose**:
   ```bash
   docker-compose up --build
   ```
   
   > ⚠️ **Note for Colima Users**: If you encounter `Exit Code 137` (OOM), increase memory:
   > `colima start --cpu 4 --memory 8`

3. **Access the App**:
   Open [http://localhost:7860](http://localhost:7860)

### Local Development (Manual)

1. **Install dependencies**:
   ```bash
   uv sync
   ```

2. **Run the app**:
   ```bash
   uv run app.py
   ```

## Deployment to Hugging Face Spaces

This project is configured for **Hugging Face Spaces** using the **Docker SDK**.

1. Create a new Space (Select **Docker** as SDK).
2. Upload the contents of the `backend` directory.
3. The `Dockerfile` handles everything:
   - Installs system dependencies (`ffmpeg`, `libsndfile1`, `build-essential`)
   - Sets up non-root user
   - Installs Python dependencies via `uv`
   - Pre-downloads AI models and fonts

## Resource Checks

The application performs automatic startup checks:
- **Fonts**: Checks for `Roboto-Regular.ttf` in `assets/fonts`. Downloads automatically from Google Fonts if missing.
- **AI Models**: Verifies `audio-separator` availability. Models are pre-cached during Docker build.

## License

MIT License
