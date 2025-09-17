import json
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import List, Optional
from zipfile import ZipFile, ZIP_DEFLATED

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from ..config import EXPORT_DIR
from ..database import get_db
from ..models import Detection, File
from ..schemas import ExportRequest

router = APIRouter(prefix="/api/export", tags=["export"])


@router.post("/paddleocr")
def export_paddleocr(payload: ExportRequest, db: Session = Depends(get_db)) -> FileResponse:
    files_query = db.query(File)
    if payload.file_ids:
        files_query = files_query.filter(File.id.in_(payload.file_ids))

    files: List[File] = files_query.order_by(File.created_at.asc()).all()
    if not files:
        raise HTTPException(status_code=404, detail="No files available for export")

    export_root = Path(tempfile.mkdtemp(prefix="paddleocr_export_"))
    images_dir = export_root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    label_path = export_root / "label.txt"

    total_detections = 0
    with open(label_path, "w", encoding="utf-8") as label_file:
        for file_obj in files:
            image_source = Path(file_obj.processed_path or file_obj.original_path)
            if not image_source.exists():
                continue

            detections = _filter_detections(file_obj.detections, payload.include_manual_only)
            if not detections:
                continue

            target_name = f"{file_obj.id}_{image_source.name}"
            target_path = images_dir / target_name
            shutil.copy(image_source, target_path)

            detection_payload = []
            for detection in detections:
                detection_payload.append(
                    {
                        "transcription": detection.text,
                        "points": [
                            [detection.bbox_x1, detection.bbox_y1],
                            [detection.bbox_x2, detection.bbox_y1],
                            [detection.bbox_x2, detection.bbox_y2],
                            [detection.bbox_x1, detection.bbox_y2],
                        ],
                        "confidence": detection.confidence,
                        "source": detection.source,
                    }
                )
            label_line = f"images/{target_name}\t{json.dumps(detection_payload, ensure_ascii=False)}\n"
            label_file.write(label_line)
            total_detections += len(detections)

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    archive_name = f"paddleocr_export_{timestamp}.zip"
    archive_path = EXPORT_DIR / archive_name

    with ZipFile(archive_path, "w", compression=ZIP_DEFLATED) as archive:
        archive.write(label_path, arcname="label.txt")
        for image_file in images_dir.iterdir():
            archive.write(image_file, arcname=f"images/{image_file.name}")

    shutil.rmtree(export_root, ignore_errors=True)

    if total_detections == 0:
        archive_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="No detections available to export")

    return FileResponse(archive_path, filename=archive_name, media_type="application/zip")


def _filter_detections(detections: List[Detection], manual_only: bool) -> List[Detection]:
    if manual_only:
        return [d for d in detections if d.source != "auto"]
    return list(detections)
