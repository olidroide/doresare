import asyncio
import json
import os
import re
import shutil
import time
import uuid
from typing import Dict, Optional

from config import Environment, settings
from fastapi import (
    BackgroundTasks,
    Cookie,
    FastAPI,
    File,
    Form,
    HTTPException,
    Request,
    Response,
    UploadFile,
)
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from gradio_client import Client, handle_file

app = FastAPI()

# Get absolute path to templates directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# Mount directory for static files (generated videos)
STATIC_DIR = os.path.join(BASE_DIR, "static")
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# --- In-memory Store ---
# jobs: job_id -> { "job": JobObject, "filename": str, "start_time": float, "temp_file": str, "status": "running"|"done"|"error", "output": str|None, "user_id": str }
jobs: Dict[str, dict] = {}

# user_active_job: user_id -> job_id (Only one active job per user)
user_active_job: Dict[str, str] = {}






# --- Dependencies ---

async def get_user_id(request: Request, response: Response, user_session: Optional[str] = Cookie(None)):
    """
    Dependency to get the user ID from cookie.
    If not present, generates a new one and sets the cookie.
    NOTE: When using HTMX, cookies are handled automatically by browser.
    """
    if not user_session:
        user_session = str(uuid.uuid4())
        # Set cookie on response. 
        # Note: In FastAPI, if you inject Response, you can set headers/cookies on it.
        # But for the *first* request, we need to return it.
        # We'll set it in the endpoint if missing.
        # However, purely relying on it here might be tricky if the endpoint doesn't return `response` object directly but a TemplateResponse.
        # Simpler approach: Check in endpoint, set in response.
        pass
    return user_session


# --- Helpers ---

def get_client():
    if settings.BACKEND_URL:
        # print(f"🔌 Connecting to backend at {settings.BACKEND_URL}")
        return Client(settings.BACKEND_URL)
    elif settings.ENV == Environment.LOCAL:
        return Client("http://127.0.0.1:7860")
    else:
        return Client(settings.HF_SPACE, hf_token=settings.HF_TOKEN)

# --- Endpoints ---

@app.middleware("http")
async def session_middleware(request: Request, call_next):
    """
    Global Middleware to ensure every user has a session ID.
    This simplifies tracking across all endpoints.
    """
    user_id = request.cookies.get("user_session")
    if not user_id:
        user_id = str(uuid.uuid4())
        request.state.user_id = user_id
    else:
        request.state.user_id = user_id
        
    response = await call_next(request)
    
    # Always set/refresh cookie to ensure it sticks
    if not request.cookies.get("user_session"):
         response.set_cookie(key="user_session", value=user_id, max_age=3600*24*30) # 30 days
         
    return response


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    user_id = request.state.user_id
    
    # Check if user has active job
    active_job_id = user_active_job.get(user_id)
    
    # If active job exists, we verify it's valid
    if active_job_id:
        if active_job_id not in jobs:
            # Stale reference, clean up
            del user_active_job[user_id]
            active_job_id = None
            
    return templates.TemplateResponse("index.html", {
        "request": request, 
        "active_job_id": active_job_id,
        "filename": jobs[active_job_id]["filename"] if active_job_id else None
    })

@app.post("/process")
async def process(request: Request, background_tasks: BackgroundTasks, video_url: str = Form(None), file: UploadFile = File(None)):
    """
    Submit a video for processing.
    User is limited to 1 active job.
    """
    user_id = request.state.user_id
    
    # 1. Enforce Single Active Job
    if user_id in user_active_job:
        existing_job_id = user_active_job[user_id]
        if existing_job_id in jobs and jobs[existing_job_id]["status"] in ["queued", "running"]:
             # Return error partial
             return templates.TemplateResponse("partials/error.html", {
                "request": request,
                "error_message": "You already have a job in progress. Please wait for it to finish."
            })
        else:
            # Stale, remove
            del user_active_job[user_id]

    temp_path = None
    try:
        input_val = None
        filename = "video"
        
        # Handle Input
        if file:
            import tempfile
            filename = file.filename
            with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.filename}") as temp_file:
                shutil.copyfileobj(file.file, temp_file)
                temp_path = temp_file.name
            input_val = handle_file(temp_path)
            print(f"🚀 Saved temp file to {temp_path}")
        elif video_url:
            # URL Processing Flow: submit URL directly to backend (backend handles download)
            client = get_client()
            print(f"🚀 Submitting URL job for {video_url}...")

            # Submit URL directly; backend pipeline expects two inputs: (file_input, url_input)
            # Provide None for the file_input and the URL as the second positional argument.
            job = client.submit(None, video_url, fn_index=0)

            job_id = str(uuid.uuid4())
            jobs[job_id] = {
                "job": job,
                "filename": video_url,
                "start_time": time.time(),
                "temp_file": None,
                "status": "queued",
                "user_id": user_id
            }
            user_active_job[user_id] = job_id

            # Return Partial
            template_name = "process.html"
            if request.headers.get("HX-Request"):
                template_name = "partials/process_content.html"

            return templates.TemplateResponse(template_name, {
                "request": request,
                "job_id": job_id,
                "filename": video_url
            })

        if not input_val:
             raise HTTPException(status_code=400, detail="No file uploaded or URL provided")

        # Connect and Submit
        client = get_client()
        print(f"🚀 Submitting job for {filename}...")
        
        # Submit (file upload). The backend expects two inputs: (file_input, url_input)
        # Provide the file value first and an empty string for the URL input.
        job = client.submit(input_val, "", fn_index=0)
        
        job_id = str(uuid.uuid4())
        jobs[job_id] = {
            "job": job,
            "filename": filename,
            "start_time": time.time(),
            "temp_file": temp_path,
            "status": "queued",
            "user_id": user_id
        }
        
        # Track active job
        user_active_job[user_id] = job_id
        
        # Return HTMX Partial
        template_name = "process.html"
        if request.headers.get("HX-Request"):
            template_name = "partials/process_content.html"

        return templates.TemplateResponse(template_name, {
            "request": request,
            "job_id": job_id,
            "filename": filename
        })

    except Exception as e:
        print(f"❌ Error: {e}")
        if temp_path and os.path.exists(temp_path):
             os.remove(temp_path)
        return templates.TemplateResponse("partials/error.html", {
            "request": request,
            "error_message": str(e)
        })

# --- Unified SSE Endpoint ---

@app.get("/events")
async def events(request: Request):
    """
    Unified SSE stream.
    Broadcasts:
    1. 'queue_status': Global queue stats (every 2s)
    2. 'job_progress': Specific job progress if user has active job (every 1s)
    """
    user_id = request.state.user_id
    print(f"🔌 SSE Connected for User: {user_id}")

    async def event_generator():
        try:
            while True:
                # 1. Global Queue Stats
                active_count = len(jobs)
                queue_data = {
                    "active_jobs": active_count,
                    "status": "busy" if active_count > 0 else "available"
                }
                yield f"event: queue_status\ndata: {json.dumps(queue_data)}\n\n"
                
                # 2. Check for Active Job for this User
                active_job_id = user_active_job.get(user_id)
                
                if active_job_id and active_job_id in jobs:
                    # Fetch Job Info
                    job_data = jobs[active_job_id]
                    job = job_data["job"]
                    
                    if job and job.done():
                        # Handle completion logic here (once)
                        if job_data["status"] != "done":
                            try:
                                result = job.result()

                                # Determine a safe output filename. If the original filename
                                # looks like a URL, generate a UUID-based filename and preserve
                                # the extension from the result file. Otherwise sanitize the
                                # original filename.
                                ext = os.path.splitext(result)[1] if result else ".mp4"
                                original = job_data.get("filename", "video")
                                if original.startswith("http"):
                                    output_filename = f"processed_{uuid.uuid4().hex}{ext}"
                                else:
                                    safe_base = re.sub(r'[^A-Za-z0-9_.-]', '_', os.path.basename(original))
                                    # Ensure extension exists
                                    if not os.path.splitext(safe_base)[1]:
                                        safe_base = safe_base + ext
                                    output_filename = f"processed_{safe_base}"

                                final_path = os.path.join(STATIC_DIR, output_filename)

                                if os.path.exists(result):
                                    shutil.move(result, final_path)
                                    # Cleanup input
                                    temp_file = job_data.get("temp_file")
                                    if temp_file and os.path.exists(temp_file):
                                        os.remove(temp_file)

                                    job_data["status"] = "done"
                                    job_data["output"] = output_filename

                                    # Send success event
                                    data = {
                                        "type": "complete",
                                        "progress": 100,
                                        "status": "Completed",
                                        "video_url": f"/static/{output_filename}"
                                    }
                                    yield f"event: job_progress\ndata: {json.dumps(data)}\n\n"

                            except Exception as e:
                                print(f"❌ Job Error: {e}")
                                data = {"type": "error", "error_message": str(e)}
                                yield f"event: job_progress\ndata: {json.dumps(data)}\n\n"

                            # Ensure we clear the active job entry for this user exactly once
                            try:
                                if user_active_job.get(user_id) == active_job_id:
                                    del user_active_job[user_id]
                            except Exception:
                                # Best-effort cleanup; avoid crashing the SSE stream
                                pass
                    
                    elif job_data["status"] == "error":
                         # Error happened during download or other sync steps
                         data = {"type": "error", "error_message": job_data.get("error_message", "Unknown error")}
                         yield f"event: job_progress\ndata: {json.dumps(data)}\n\n"
                         del user_active_job[user_id]

                    elif job: # Regular Gradio Job Running
                        # Job Running
                        status = job.status()
                        
                        # Extract info
                        progress = 0
                        desc = "Processing..."
                        detail = ""
                        
                        if status.progress_data:
                            latest = status.progress_data[-1]
                            if latest.progress is not None:
                                progress = int(latest.progress * 100)
                            if latest.desc:
                                desc = latest.desc
                            
                            # Attempt to extract granular details if available
                            # Gradio Client ProgressUnit has: index, length, unit, desc, progress
                            idx = getattr(latest, "index", None)
                            length = getattr(latest, "length", None)
                            unit = getattr(latest, "unit", "steps")
                            
                            if idx is not None and length is not None:
                                detail = f"{idx} / {length} {unit}"
                            
                            # Clean up desc (remove generic 0% - 100%)
                            if desc:
                                # Remove " 1%" or " 100%" or " (Frame ...)" if it exists
                                # If desc is "Rendering video: 1% (Frame 60/4583)"
                                # We want status="Rendering video" and detail="Frame 60/4583" (if detail is empty)

                                # 1. Extract potential detail from desc if detail is empty
                                if not detail:
                                    frame_match = re.search(r'\((Frame.*?)\)', desc)
                                    if frame_match:
                                        detail = frame_match.group(1)

                                # 2. Remove percentage and parens from status
                                desc = re.sub(r'\s*\d+%', '', desc) # Remove 1%
                                desc = re.sub(r'\s*\(Frame.*?\)', '', desc) # Remove (Frame...) if we extracted it or not
                                desc = desc.strip().rstrip(':')

                        # Calculate queue pos
                        # ... (Queue logic simplified for brevity, same as before)
                        queue_pos = "-"
                        if "QUEUE" in str(status.code) and status.rank is not None:
                             queue_pos = str(status.rank + 1)

                        data = {
                            "type": "progress",
                            "progress": progress,
                            "status": desc,
                            "detail": detail,
                            "position": queue_pos,
                            "queue_size": str(len(jobs))
                        }
                        yield f"event: job_progress\ndata: {json.dumps(data)}\n\n"
                
                # Sleep
                await asyncio.sleep(1)

        except asyncio.CancelledError:
            print(f"🔌 SSE Disconnected for User: {user_id}")

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/process/{job_id}", response_class=HTMLResponse)
async def view_process(request: Request, job_id: str):
    # This endpoint is kept for manual URL access, 
    # but the logic is now mostly handled by checking session in /
    if job_id not in jobs:
         return templates.TemplateResponse("partials/error.html", {
            "request": request,
            "error_message": "Job not found or expired."
        })
        
    template_name = "process.html"
    if request.headers.get("HX-Request"):
        template_name = "partials/process_content.html"

    return templates.TemplateResponse(template_name, {
        "request": request,
        "job_id": job_id,
        "filename": jobs[job_id]["filename"]
    })
