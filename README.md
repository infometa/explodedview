# Exploded View OCR Platform

This project provides a web interface and FastAPI backend for running PaddleOCR on product exploded-view diagrams. Users can upload multiple images, review auto-detected component codes, correct bounding boxes directly in the browser, upscale blurry images, and export datasets in PaddleOCR training format.

## Features

- Multi-image upload with async background OCR processing.
- Image quality analysis (blur / contrast) with optional Real-ESRGAN upscaling.
- PaddleOCR recognition constrained to digits, letters A–E, and `-`.
- Interactive Bootstrap UI with Konva.js viewport for reviewing and editing bounding boxes.
- Manual bbox creation, editing, and deletion with live updates via WebSocket.
- SQLite storage of files and detections; PaddleOCR-compatible export (`label.txt` + images zip).

## Project Structure

```text
app/
  main.py             # FastAPI app factory and router setup
  models.py           # SQLAlchemy models for files and detections
  schemas.py          # Pydantic schemas for API contracts
  database.py         # SQLite engine/session helpers
  services/           # OCR, upscaling, and processing utilities
  routers/            # REST endpoints (files, detections, export)
  websocket_manager.py
  utils.py
assets/
  ocr_char_dict.txt   # Restricted character set for PaddleOCR
static/
  index.html          # Bootstrap + Konva.js frontend
images/               # Sample exploded-view inputs (for reference)
```

## Prerequisites

- Python 3.9+
- Recommended: create a virtual environment.
- Install dependencies: `pip install -r requirements.txt`
- Real-ESRGAN weights (optional) should be placed under `weights/` (e.g., `weights/realesr-general-x4v3.pth`). Without weights, the backend falls back to OpenCV interpolation.

## Running Locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 7860
```

Open http://127.0.0.1:7860/static/index.html to access the UI.

## Notes

- Uploads and exports are stored under `uploads/` and `exports/` (created automatically on startup).
- PaddleOCR model directories can be customized in `app/services/ocr_service.py` if needed.
- The export endpoint supports exporting all processed images or only manual-confirmed detections.
