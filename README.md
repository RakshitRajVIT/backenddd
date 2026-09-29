# Civic Mirror — Backend

AI-powered traffic-violation detection backend. Runs three independently
trained YOLOv8 models (helmet, triple-riding, red-light) over an uploaded
video, produces a single annotated output video, and a JSON violation
report — via FastAPI, with background job processing.

## 1. Requirements

- Python 3.10–3.12 (Windows, macOS or Linux)
- Optional: NVIDIA GPU + CUDA-enabled PyTorch for acceleration (CPU works too)
- Optional: `ffmpeg` on PATH, for re-encoding output to browser-playable H.264
  (falls back to a plain OpenCV MP4 automatically if not present)

## 2. Windows installation

Open **PowerShell** in the `backend/` folder:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

GPU users: install a CUDA build of PyTorch that matches your driver **before**
`pip install -r requirements.txt` skips torch, e.g.:

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

Copy the environment template and edit it:

```powershell
copy .env.example .env
notepad .env
```

## 3. Add your model weights

Place your three trained weight files here (or point `*_MODEL_PATH` in `.env`
at their actual location):

```
backend/models/helmet.pt
backend/models/tripling.pt
backend/models/red_light.pt
```

The backend starts even if a model is missing — `/api/health` reports which
models loaded and why any failed. Processing works with whatever subset of
models loaded, unless `REQUIRE_ALL_MODELS=true`.

## 4. Configure class names (required for confirmed violations)

**Class names are never assumed.** On first run, check `GET /api/health` (or
the console log at startup) — it lists each model's actual class dictionary
(`{0: "..."}` etc). Then set the matching env vars in `.env`:

| Variable | Purpose |
|---|---|
| `HELMET_VIOLATION_CLASSES` | class name(s)/id(s) meaning "rider without helmet" |
| `TRIPLING_VIOLATION_CLASSES` | class meaning "triple riding" directly (if your model has one) |
| `TRIPLING_PERSON_CLASSES` / `TRIPLING_MOTORCYCLE_CLASSES` | alternative: person + motorcycle classes, used for spatial association (≥3 people on one motorcycle) |
| `RED_LIGHT_RED_SIGNAL_CLASSES` | class meaning "signal is red" |
| `RED_LIGHT_VEHICLE_CLASSES` | vehicle classes to track |
| `RED_LIGHT_STOP_LINE` | `x1,y1,x2,y2` normalized (0–1) stop-line coordinates |
| `RED_LIGHT_VIOLATION_CLASSES` | optional direct "violation" class — shown as an **unverified candidate only**, never auto-confirmed |

Until these are set for a category, that category still runs and draws raw
detections, but reports **zero confirmed violations** — by design, detection
alone is never presented as a confirmed violation (see `## Violation logic`).

## 5. Run the server

```powershell
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Or with auto-reload during development:

```powershell
uvicorn app.main:app --reload --port 8000
```

Interactive API docs: `http://localhost:8000/docs`

## 6. macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## API endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | Backend + per-model status, loaded classes, config warnings |
| POST | `/api/videos/upload` | Upload a video (`multipart/form-data`, field `file`) |
| POST | `/api/videos/{job_id}/process` | Start background processing |
| GET | `/api/videos/{job_id}/status` | Progress %, frames processed, current state |
| GET | `/api/videos/{job_id}/results` | Full JSON report (once completed) |
| GET | `/api/videos/{job_id}/download` | Download annotated MP4 |
| GET | `/api/videos/{job_id}/report` | Download the JSON report file |
| DELETE | `/api/videos/{job_id}` | Delete uploaded/processed files + job record |

All responses use the envelope `{"success": bool, "message": str, "data": ...}`
on success, or `{"success": false, "error": {"code", "message"}}` on failure,
with meaningful HTTP status codes (400/404/409/413/415/500).

## Violation logic — detection vs. confirmation

Detection and verification are deliberately separate:

- **Helmet / Tripling**: a candidate must be tracked (IoU tracker) and
  observed for `*_MIN_HITS` consecutive frames before being reported as a
  confirmed violation (filters one-off false positives). Tripling additionally
  supports person→motorcycle spatial association as an alternative to a
  dedicated model class.
- **Red-light**: confirmation requires *all* of — the signal classified as
  RED over a sliding window, a tracked vehicle crossing the configured stop
  line, in the configured direction if any. A model's own "violation" class
  (if configured) is shown only as an **unverified candidate**, never
  auto-confirmed, since detection alone can't prove signal state + crossing.

Every category still reports its raw/candidate detections and clearly labels
them `candidate` / `unverified` vs `confirmed` in the JSON report — nothing is
silently fabricated as a violation.

## Notes

- Models load once at startup and are reused across all jobs (thread-safe,
  serialized per-model inference).
- `DEVICE=auto` picks CUDA if available, else CPU automatically.
- `FRAME_STRIDE>1` runs inference every Nth frame and reuses the last known
  boxes on skipped frames, to speed up long videos.
- Only one processing job per `job_id` can run at a time; calling `/process`
  again while a job is queued/processing/completed returns HTTP 409.
- Deleting a job mid-processing signals cancellation and removes files once
  released (handles Windows file-lock timing with a short retry).
"# backenddd" 
