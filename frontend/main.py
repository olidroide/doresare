import asyncio
import json
import os
import re
import shutil
import time
import uuid
from typing import Dict, Optional

from config import Environment, settings
from use_cases.download_video import (
    DownloadError,
    RequiresFFmpegError,
    download_video_stream,
)
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

# --- Background Tasks ---

async def background_process_video(job_id: str, user_id: str, video_url: Optional[str] = None, file_path: Optional[str] = None):
    """
    Handles the heavy lifting of downloading and/or uploading to the backend.
    Updates the global `jobs` dict to reflect progress.
    """
    temp_path = file_path
    filename = "video"
    if file_path:
        filename = os.path.basename(file_path)
    elif video_url:
        filename = video_url

    try:
        input_val = None
        
        # 1. Handle YouTube/URL Download if needed
        if video_url:
            print(f"📥 [Background] Starting download for {video_url}...")
            try:
                # Stream download updates
                async for status in download_video_stream(video_url, max_height=720):
                    if isinstance(status, (int, float)):
                        jobs[job_id]["status"] = "downloading"
                        jobs[job_id]["progress"] = int(status)
                    elif isinstance(status, dict):
                        # Rich progress update
                        jobs[job_id]["status"] = "downloading"
                        jobs[job_id]["progress"] = int(status["pct"])
                        
                        # Format detail
                        total_mb = status["total"] / (1024 * 1024) if status["total"] else 0
                        downloaded_mb = status["downloaded"] / (1024 * 1024) if status["downloaded"] else 0
                        speed_mb = (status["speed"] or 0) / (1024 * 1024)
                        
                        jobs[job_id]["detail"] = f"{downloaded_mb:.1f} / {total_mb:.1f} MB ({speed_mb:.1f} MB/s)"
                    elif hasattr(status, "exists"):  # It's a Path object
                        temp_path = str(status)
                        filename = os.path.basename(temp_path)
                        jobs[job_id]["temp_file"] = temp_path
                        jobs[job_id]["filename"] = filename

                if not temp_path:
                    raise DownloadError("Download failed to produce a file.")

            except Exception as e:
                print(f"❌ [Background] Download failed: {e}")
                jobs[job_id]["status"] = "error"
                jobs[job_id]["error_message"] = f"Download failed: {str(e)}"
                return

        # 2. Upload to Backend
        if not temp_path or not os.path.exists(temp_path):
             jobs[job_id]["status"] = "error"
             jobs[job_id]["error_message"] = "Local file missing for processing."
             return

        print(f"🚀 [Background] Submitting {filename} to backend...")
        jobs[job_id]["status"] = "queued"
        jobs[job_id]["progress"] = 0
        jobs[job_id]["detail"] = "Uploading to analytical engine..."

        try:
             # Prepare file for upload to backend
             input_val = handle_file(temp_path)
             client = get_client()
             
             # Submit (file upload). The backend expects two inputs: (file_input, url_input)
             job = client.submit(input_val, "", fn_index=0)
             
             # Update job entry with the live Gradio job
             jobs[job_id].update({
                 "job": job,
                 "status": "queued",
                 "detail": "Connected to backend"
             })
             
             print(f"✅ [Background] Job {job_id} successfully submitted to backend.")

        except Exception as e:
             print(f"❌ [Background] Client submit failed: {e}")
             jobs[job_id]["status"] = "error"
             jobs[job_id]["error_message"] = f"Failed to connect to backend: {str(e)}"
             return

    except Exception as e:
        print(f"❌ [Background] Critical Error: {e}")
        import traceback
        traceback.print_exc()
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error_message"] = str(e)

    finally:
        # We don't delete temp_path here yet because the Gradio client might still be using it 
        # (if it handles the upload asynchronously/later, though submit() usually does it).
        # Actually, handle_file returns a path that Gradio knows how to upload.
        # The best place to cleanup is after the backend job is done (already handled in SSE stream).
        pass


@app.post("/process")
async def process(request: Request, background_tasks: BackgroundTasks, video_url: str = Form(None), file: UploadFile = File(None)):
    """
    Submit a video for processing.
    User is limited to 1 active job.
    Returns the UI partial IMMEDIATELY while processing continues in background.
    """
    user_id = request.state.user_id
    
    # 1. Enforce Single Active Job
    if user_id in user_active_job:
        existing_job_id = user_active_job[user_id]
        if existing_job_id in jobs and jobs[existing_job_id]["status"] in ["downloading", "queued", "running"]:
             return templates.TemplateResponse("partials/error.html", {
                "request": request,
                "error_message": "You already have a job in progress. Please wait for it to finish."
            })
        else:
            # Stale/Done, remove from active track
            if user_id in user_active_job:
                del user_active_job[user_id]

    job_id = str(uuid.uuid4())
    temp_path = None
    filename = "video"

    try:
        # 2. Setup Job Entry
        jobs[job_id] = {
            "job": None,
            "filename": "",
            "start_time": time.time(),
            "temp_file": None,
            "status": "downloading" if video_url else "queued",
            "progress": 0,
            "detail": "Initializing..." if video_url else "Saving upload...",
            "user_id": user_id,
        }
        user_active_job[user_id] = job_id

        # 3. Handle File Upload (Sync Part)
        # We save the file immediately so the UploadFile object isn't closed when the request returns
        if file and file.filename:
            filename = file.filename
            jobs[job_id]["filename"] = filename
            
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.filename}") as temp_file:
                shutil.copyfileobj(file.file, temp_file)
                temp_path = temp_file.name
            
            jobs[job_id]["temp_file"] = temp_path
            print(f"🚀 [Sync] Saved temp upload to {temp_path}")
            
            # Start background upload logic
            background_tasks.add_task(background_process_video, job_id, user_id, file_path=temp_path)
            
        elif video_url:
            filename = video_url
            jobs[job_id]["filename"] = "YouTube Video"
            # Start background download/upload logic
            background_tasks.add_task(background_process_video, job_id, user_id, video_url=video_url)
        
        else:
             raise HTTPException(status_code=400, detail="No file uploaded or URL provided")

        # 4. Return UI Partial IMMEDIATELY
        template_name = "process.html"
        if request.headers.get("HX-Request"):
            template_name = "partials/process_content.html"

        return templates.TemplateResponse(template_name, {
            "request": request,
            "job_id": job_id,
            "filename": filename
        })

    except Exception as e:
        print(f"❌ Error in process endpoint: {e}")
        # Cleanup
        if job_id in jobs: del jobs[job_id]
        if user_id in user_active_job: del user_active_job[user_id]
        
        return templates.TemplateResponse("partials/error.html", {
            "request": request,
            "error_message": f"Failed to start processing: {str(e)}"
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
                active_count = sum(1 for j in jobs.values() if j.get("status") in ["queued", "running", "downloading"])
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

                    # If this is a frontend download-in-progress, expose its progress
                    if job_data.get("status") == "downloading":
                        data = {
                            "type": "progress",
                            "progress": int(job_data.get("progress", 0)),
                            "status": "Downloading",
                            "detail": job_data.get("detail", ""),
                        }
                        yield f"event: job_progress\ndata: {json.dumps(data)}\n\n"
                        await asyncio.sleep(0.2) # Fast updates for downloading
                        continue

                    job = job_data.get("job")
                    
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
                                    # Build a safe filename: keep extension if valid, but disallow dots
                                    # inside the base name to avoid traversal or trick filenames.
                                    base = os.path.basename(original)
                                    name, orig_ext = os.path.splitext(base)
                                    # Normalize and validate extension
                                    orig_ext = (orig_ext or ext).lower()
                                    if not re.match(r'^\.[a-z0-9]{1,6}$', orig_ext):
                                        orig_ext = ext

                                    # Allow only alphanumerics, dash and underscore in the base name
                                    safe_name = re.sub(r'[^A-Za-z0-9_-]', '_', name)

                                    # If name is empty or contains suspicious sequences, fall back to UUID
                                    if (not safe_name) or ('..' in safe_name) or safe_name.startswith('..') or safe_name.endswith('..'):
                                        safe_name = uuid.uuid4().hex

                                    output_filename = f"processed_{safe_name}{orig_ext}"

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
                            
                            # Clean up desc and extract detail
                            if desc:
                                # 1. Extract detail if desc contains "|" or "-"
                                if not detail:
                                    if " | " in desc:
                                        desc, detail = desc.split(" | ", 1)
                                    elif " - " in desc:
                                        desc, detail = desc.split(" - ", 1)

                                # 2. Extract potential detail from desc if detail is still empty (Frame info)
                                if not detail:
                                    frame_match = re.search(r'\((Frame.*?)\)', desc)
                                    if frame_match:
                                        detail = frame_match.group(1)

                                # 3. Clean status text: remove percentage, empty parens, and trailing colons
                                desc = re.sub(r'\s*\d+%', '', desc) # Remove " 82%"
                                desc = re.sub(r'\s*\(Frame.*?\)', '', desc) # Remove "(Frame...)"
                                desc = re.sub(r'\s*\(\)', '', desc) # Remove empty "()"
                                desc = desc.strip().rstrip(':').strip()

                        # Calculate queue pos and ETA
                        queue_pos = "-"
                        eta_formatted = ""
                        
                        # rank_eta is an estimate in seconds (safely accessed)
                        rank_eta = getattr(status, "rank_eta", None)
                        if rank_eta is not None:
                            eta_seconds = int(rank_eta)
                            if eta_seconds > 0:
                                if eta_seconds >= 60:
                                    eta_formatted = f"{eta_seconds // 60}m {eta_seconds % 60}s"
                                else:
                                    eta_formatted = f"{eta_seconds}s"
                        
                        if "QUEUE" in str(status.code) and status.rank is not None:
                             queue_pos = str(status.rank + 1)
                        
                        # Heartbeat Logic: Find if ANY job is currently 'Processing'
                        # This allows queued users to see that the queue is actually moving.
                        global_progress = None
                        active_job_status = "Waiting..."
                        
                        for other_job_id, other_data in list(jobs.items()):
                            if other_data.get("status") == "running" and other_data.get("job"):
                                # This is the active job at the front of the queue
                                other_status = other_data["job"].status()
                                if other_status.progress_data:
                                    p_unit = other_status.progress_data[-1]
                                    if p_unit.progress is not None:
                                        global_progress = int(p_unit.progress * 100)
                                        active_job_status = p_unit.desc or "Processing..."
                                        # Clean active_job_status
                                        active_job_status = re.sub(r'\s*\d+%', '', active_job_status).strip().rstrip(':').strip()
                                break

                        data = {
                            "type": "progress",
                            "progress": progress,
                            "status": desc,
                            "detail": detail,
                            "position": queue_pos,
                            "queue_size": str(len(jobs)),
                            "eta": eta_formatted,
                            "global_progress": global_progress,
                            "active_job_status": active_job_status
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
