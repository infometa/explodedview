import mimetypes
from pathlib import Path
from typing import List, Optional
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from ..config import UPLOAD_DIR
from ..database import get_db
from ..models import File
from ..schemas import FileDetail, FileRead
from ..services.processing import process_file

router = APIRouter(prefix="/api/files", tags=["files"])


@router.get("/", response_model=List[FileRead])
def list_files(db: Session = Depends(get_db)) -> List[FileRead]:
    files = db.query(File).order_by(File.created_at.desc()).all()
    return files


@router.get("/{file_id}", response_model=FileDetail)
def get_file(file_id: int, db: Session = Depends(get_db)) -> FileDetail:
    file_obj: Optional[File] = db.query(File).filter(File.id == file_id).first()
    if not file_obj:
        raise HTTPException(status_code=404, detail="File not found")
    file_obj.detections.sort(key=lambda det: det.id)
    return file_obj


@router.post("/upload", response_model=List[FileRead])
async def upload_files(
    request: Request,
    files: List[UploadFile],
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
) -> List[FileRead]:
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    stored_records: List[File] = []

    for file in files:
        if not file.filename:
            raise HTTPException(status_code=400, detail="File name missing")
        content_type = file.content_type or mimetypes.guess_type(file.filename)[0]
        if not content_type or not content_type.startswith("image"):
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {file.filename}")

        storage_key = uuid4().hex
        target_dir = UPLOAD_DIR / storage_key
        target_dir.mkdir(parents=True, exist_ok=True)

        suffix = Path(file.filename).suffix or ".png"
        original_path = target_dir / f"original{suffix}"

        contents = await file.read()
        original_path.write_bytes(contents)

        record = File(
            original_filename=file.filename,
            storage_dir=storage_key,
            original_path=str(original_path),
            status="pending",
        )
        db.add(record)
        db.flush()
        stored_records.append(record)

        background_tasks.add_task(process_file, record.id, request.app.state)

    db.commit()
    return stored_records


@router.get("/{file_id}/image")
def download_image(
    file_id: int,
    processed: bool = False,
    db: Session = Depends(get_db),
) -> FileResponse:
    file_obj: Optional[File] = db.query(File).filter(File.id == file_id).first()
    if not file_obj:
        raise HTTPException(status_code=404, detail="File not found")

    target_path = Path(file_obj.processed_path if processed and file_obj.processed_path else file_obj.original_path)
    if not target_path.exists():
        raise HTTPException(status_code=404, detail="Image file not available")

    return FileResponse(target_path)
