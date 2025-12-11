from fastapi import FastAPI, UploadFile, File, Request
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
from fastapi.responses import StreamingResponse
import asyncio
import json

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

import tempfile

@app.post("/upload")
async def process(request: Request, file: UploadFile):
    try:
        # 1. Create a named temporary file
        # delete=False because we need to close it before passing path to gradio_client,
        # but we will manually delete it immediately after submission.
        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.filename}") as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_path = temp_file.name
            
        print(f"🚀 Saved temp file to {temp_path}")
            
        # 2. Connect and Submit (Async)
        client = get_client()
        print(f"🚀 Sending {temp_path} to backend (Async)...")
        
        # Use submit() to not block and get a Job
        # The client uploads the file during submit(), so we can delete it right after.
        job = client.submit(
            input_video=handle_file(temp_path),
            api_name="/predict"
        )
        
        # 3. Immediate Cleanup
        # We don't need the file locally anymore as it's been uploaded to the backend (or HF Space)
        if os.path.exists(temp_path):
            os.remove(temp_path)
            print(f"🧹 Deleted temp file {temp_path}")
        
        # Save job ID
        job_id = str(uuid.uuid4())
        jobs[job_id] = {
            "job": job,
            "filename": file.filename,
            "start_time": time.time()
        }
        
        # Return initial HTML of progress bar which will start polling
        return templates.TemplateResponse("partials/queued.html", {
            "request": request,
            "job_id": job_id
        })

    except Exception as e:
        print(f"❌ Error: {e}")
        # Ensure cleanup on error
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.remove(temp_path)
            
        return templates.TemplateResponse("partials/error.html", {
            "request": request,
            "error_message": str(e)
        })



@app.get("/status/{job_id}")
async def stream_status(request: Request, job_id: str):
    if job_id not in jobs:
        return HTMLResponse("<div>Error: Job not found</div>")

    async def event_generator():
        job_data = jobs[job_id]
        job = job_data["job"]
        
        print(f"🔍 DEBUG: Starting SSE stream for job {job_id}")
        
        while not job.done():
            status = job.status()
            print(f"🔍 DEBUG: Job Status Code: {status.code}, rank={status.rank}, queue_size={status.queue_size}")
            
            # Calculate progress
            progress = 0
            progress_desc = "Processing..."
            if status.progress_data:
                print(f"🔍 DEBUG: Progress Data: {status.progress_data}")
                latest = status.progress_data[-1]
                if latest:
                    try:
                        # Gradio client progress is 0.0-1.0, we want 0-100
                        progress = int(latest.progress * 100) if latest.progress is not None else 0
                        if latest.desc: progress_desc = latest.desc
                    except Exception as e: 
                        print(f"⚠️ Error parsing progress: {e}")
            
            # Determine queue position
            # Manually count active jobs to have real total
            total_active_jobs = len(jobs)
            
            # Check if job is queued or processing
            # Gradio status.code is a string: "QUEUED", "PROCESSING", "STARTING", "SUCCESS", "FAILED"
            status_code = str(status.code).upper()
            
            if "QUEUE" in status_code or status_code == "PENDING":
                # Job is definitely waiting in queue
                if status.rank is not None:
                    # Use Gradio's rank (0-indexed position in queue)
                    position = str(status.rank + 1)  # Make it 1-indexed for display
                else:
                    # Gradio didn't provide rank, estimate based on total jobs
                    position = "Queued"
                queue_size = str(total_active_jobs) if total_active_jobs > 1 else "-"
            elif "PROCESS" in status_code or "START" in status_code:
                # Job is actively being processed or starting
                position = "Processing"
                queue_size = "-"
            else:
                # Fallback for other states
                if status.rank is not None and status.rank > 0:
                    position = str(status.rank + 1)
                    queue_size = str(total_active_jobs)
                else:
                    position = "Processing"
                    queue_size = "-"
            
            elapsed = int(time.time() - job_data['start_time'])
            
            # Send JSON instead of HTML for fluid updates
            data = {
                "type": "progress",
                "progress": progress,
                "progress_desc": progress_desc,
                "position": position,
                "queue_size": queue_size,
                "elapsed_time": elapsed
            }
            print(f"📤 Sending to client: position={position}, queue_size={queue_size}, progress={progress}%")
            yield f"data: {json.dumps(data)}\n\n"
            await asyncio.sleep(1)
            
        # When finished
        print(f"✅ DEBUG: Job {job_id} finished!")
        try:
            result_path = job.result()
            print(f"📂 DEBUG: Job Result Path: {result_path}")
            
            output_filename = f"processed_{job_data['filename']}"
            final_path = os.path.join(STATIC_DIR, output_filename)
            
            # Check if result_path exists
            if os.path.exists(result_path):
                print(f"🚚 Moving {result_path} to {final_path}")
                shutil.move(result_path, final_path)
            else:
                print(f"❌ Error: Result file not found at {result_path}")
                raise Exception(f"Result file not found at {result_path}")
            
            # Cleanup
            # if os.path.exists(job_data["temp_input"]):
            #     os.remove(job_data["temp_input"])
            del jobs[job_id]
            
            # Send result as JSON
            data = {
                "type": "success",
                "output_filename": output_filename
            }
            yield f"data: {json.dumps(data)}\n\n"
            yield "event: close\ndata: \n\n"
            
        except Exception as e:
            print(f"❌ Error in SSE completion: {e}")
            import traceback
            traceback.print_exc()
            
            # Send error as JSON
            data = {
                "type": "error",
                "error_message": str(e)
            }
            yield f"data: {json.dumps(data)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
