# Interface Layer

This directory is intended for the **Interface Layer** of the Clean Architecture.

## Purpose
It should contain the code responsible for:
- **API Definitions**: FastAPI routers, Gradio interfaces, CLI commands.
- **Adapters**: Code that adapts external input (HTTP requests, user input) to the internal Use Cases (Services).
- **Presenters**: Code that formats the output from Use Cases for the user (JSON responses, UI updates).
