from fastapi import FastAPI, UploadFile, File, Request, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from gradio_client import Client, handle_file
import shutil
import os
import time
import uuid
import json
import asyncio
from config import settings, Environment

app = FastAPI()

# Get absolute path to templates directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# Mount directory for static files (generated videos)
STATIC_DIR = os.path.join(BASE_DIR, "static")
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# In-memory Job Store
# Structure: job_id -> { "job": JobObject, "filename": str, "start_time": float, "temp_file": str, "status": "running"|"done"|"error", "output": str|None }
jobs = {}

def get_client():
    if settings.BACKEND_URL:
        print(f"🔌 Connecting to backend at {settings.BACKEND_URL}")
        return Client(settings.BACKEND_URL)
    elif settings.ENV == Environment.LOCAL:
        return Client("http://127.0.0.1:7860")
    else:
        return Client(settings.HF_SPACE, hf_token=settings.HF_TOKEN)

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/queue-status")
async def queue_status(request: Request):
    """Get the current queue status"""
    active_count = len(jobs)
    return templates.TemplateResponse(
        "partials/queue_status.html",
        {
            "request": request,
            "active_jobs": active_count,
            "status": "busy" if active_count > 0 else "available"
        }
    )

@app.get("/process/{job_id}", response_class=HTMLResponse)
async def view_process(request: Request, job_id: str):
    """
    View a specific job's progress. Useful for reconnection.
    """
    if job_id not in jobs:
        return templates.TemplateResponse("partials/error.html", {
            "request": request,
            "error_message": "Job not found or expired."
        })
    
    # Check if HTMX request (unlikely for direct GET, but supportive)
    template_name = "process.html"
    if request.headers.get("HX-Request"):
        template_name = "partials/process_content.html"

    return templates.TemplateResponse(template_name, {
        "request": request,
        "job_id": job_id,
        "filename": jobs[job_id]["filename"]
    })

@app.post("/process")
async def process(request: Request, video_url: str = None, file: UploadFile = File(None)):
    """
    Submit a video for processing. Supports both URL and File upload.
    Returns HTMX partial with SSE connection.
    """
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
            pass

        if not input_val:
             raise HTTPException(status_code=400, detail="No file uploaded")

        # Connect and Submit
        client = get_client()
        print(f"🚀 Submitting job for {filename}...")
        
        job = client.submit(
            input_video=input_val,
            api_name="/predict"
        )
        
        job_id = str(uuid.uuid4())
        jobs[job_id] = {
            "job": job,
            "filename": filename,
            "start_time": time.time(),
            "temp_file": temp_path,
            "status": "queued"
        }
        
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

@app.get("/sse/{job_id}")
async def sse_progress(job_id: str):
    if job_id not in jobs:
        # Return a special event to tell client to stop or show error
        async def error_stream():
            data = {"type": "error", "error_message": "Job not found"}
            yield f"data: {json.dumps(data)}\n\n"
        return StreamingResponse(error_stream(), media_type="text/event-stream")

    async def event_generator():
        job_data = jobs[job_id]
        job = job_data["job"]
        
        print(f"🔌 SSE Connected for job {job_id}")
        
        while not job.done():
            status = job.status()
            
            # Extract status info
            progress = 0
            desc = "Processing..."
            if status.progress_data:
                latest = status.progress_data[-1]
                if latest and latest.progress is not None:
                    progress = int(latest.progress * 100)
                if latest and latest.desc:
                    desc = latest.desc
            
            # Map Gradio status to UI status
            status_code = str(status.code)
            
            # Send Progress Event
            data = {
                "type": "progress",
                "progress": progress,
                "status": desc,
                "raw_status": status_code
            }
            yield f"data: {json.dumps(data)}\n\n"
            
            await asyncio.sleep(1) # Poll interval
            
        # Job Finished
        print(f"✅ Job {job_id} finished")
        try:
            result = job.result()
            
            # Handle Result (Move file, etc.)
            output_filename = f"processed_{job_data['filename']}"
            final_path = os.path.join(STATIC_DIR, output_filename)
            
            # The result from Gradio Client is a path to the file
            if os.path.exists(result):
                shutil.move(result, final_path)
                print(f"Moved result to {final_path}")
                
                # Cleanup Input
                if job_data["temp_file"] and os.path.exists(job_data["temp_file"]):
                    os.remove(job_data["temp_file"])
                
                # Update Job Store (optional, keeps it available for a bit?)
                # We can remove it or mark it done. User might reload page.
                jobs[job_id]["status"] = "done"
                jobs[job_id]["output"] = output_filename
                
                data = {
                    "type": "complete",
                    "progress": 100,
                    "status": "Completed",
                    "video_url": f"/static/{output_filename}"
                }
                yield f"data: {json.dumps(data)}\n\n"
                
            else:
                raise Exception("Result file missing")
                
        except Exception as e:
            print(f"❌ Job Error: {e}")
            data = {"type": "error", "error_message": str(e)}
            yield f"data: {json.dumps(data)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
