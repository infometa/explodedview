from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session

from ..database import get_db
from ..models import Detection, File
from ..schemas import DetectionCreate, DetectionRead, DetectionUpdate
from ..utils import schedule_async

router = APIRouter(prefix="/api/detections", tags=["detections"])


def _normalize_bbox(x1: float, y1: float, x2: float, y2: float) -> tuple[float, float, float, float]:
    left = min(x1, x2)
    right = max(x1, x2)
    top = min(y1, y2)
    bottom = max(y1, y2)
    return left, top, right, bottom


@router.post("/", response_model=DetectionRead)
def create_detection(
    payload: DetectionCreate,
    request: Request,
    db: Session = Depends(get_db),
) -> DetectionRead:
    file_obj: Optional[File] = db.query(File).filter(File.id == payload.file_id).first()
    if not file_obj:
        raise HTTPException(status_code=404, detail="File not found")

    x1, y1, x2, y2 = _normalize_bbox(payload.bbox_x1, payload.bbox_y1, payload.bbox_x2, payload.bbox_y2)

    detection = Detection(
        file_id=file_obj.id,
        bbox_x1=x1,
        bbox_y1=y1,
        bbox_x2=x2,
        bbox_y2=y2,
        text=payload.text,
        confidence=payload.confidence if payload.confidence is not None else 1.0,
        source=payload.source or "manual",
        version=1,
    )
    db.add(detection)
    file_obj.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(detection)

    schedule_async(
        request.app.state.websocket_manager.broadcast(
            {
                "event": "detection_created",
                "file_id": file_obj.id,
                "detection_id": detection.id,
            }
        )
    )

    return detection


@router.patch("/{detection_id}", response_model=DetectionRead)
def update_detection(
    detection_id: int,
    payload: DetectionUpdate,
    request: Request,
    db: Session = Depends(get_db),
) -> DetectionRead:
    detection: Optional[Detection] = db.query(Detection).filter(Detection.id == detection_id).first()
    if not detection:
        raise HTTPException(status_code=404, detail="Detection not found")

    modified = False

    if payload.text is not None:
        detection.text = payload.text
        modified = True

    if payload.confidence is not None:
        detection.confidence = payload.confidence
        modified = True

    bbox_values = [payload.bbox_x1, payload.bbox_y1, payload.bbox_x2, payload.bbox_y2]
    if any(value is not None for value in bbox_values):
        x1 = payload.bbox_x1 if payload.bbox_x1 is not None else detection.bbox_x1
        y1 = payload.bbox_y1 if payload.bbox_y1 is not None else detection.bbox_y1
        x2 = payload.bbox_x2 if payload.bbox_x2 is not None else detection.bbox_x2
        y2 = payload.bbox_y2 if payload.bbox_y2 is not None else detection.bbox_y2
        x1, y1, x2, y2 = _normalize_bbox(x1, y1, x2, y2)
        detection.bbox_x1 = x1
        detection.bbox_y1 = y1
        detection.bbox_x2 = x2
        detection.bbox_y2 = y2
        modified = True

    if payload.source:
        detection.source = payload.source
    elif detection.source == "auto":
        detection.source = "manual"

    if modified:
        detection.version += 1

    detection.file.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(detection)

    schedule_async(
        request.app.state.websocket_manager.broadcast(
            {
                "event": "detection_updated",
                "file_id": detection.file_id,
                "detection_id": detection.id,
            }
        )
    )

    return detection


@router.delete("/{detection_id}")
def delete_detection(
    detection_id: int,
    request: Request,
    db: Session = Depends(get_db),
) -> dict:
    detection: Optional[Detection] = db.query(Detection).filter(Detection.id == detection_id).first()
    if not detection:
        raise HTTPException(status_code=404, detail="Detection not found")

    file_id = detection.file_id
    parent_file = detection.file
    db.delete(detection)
    if parent_file:
        parent_file.updated_at = datetime.utcnow()
    db.commit()

    schedule_async(
        request.app.state.websocket_manager.broadcast(
            {
                "event": "detection_deleted",
                "file_id": file_id,
                "detection_id": detection_id,
            }
        )
    )

    return {"ok": True}
