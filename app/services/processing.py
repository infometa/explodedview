import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from ..database import SessionLocal
from ..models import Detection, File
from ..utils import schedule_async
from .image_quality import ImageQualityAnalyzer
from .ocr_service import OCRService
from .upscale import UpscaleEngine

logger = logging.getLogger(__name__)


def process_file(file_id: int, app_state: Any) -> None:
    db: Session = SessionLocal()
    try:
        file_obj = db.query(File).filter(File.id == file_id).first()
        if not file_obj:
            logger.warning("File with id %s not found", file_id)
            return

        file_obj.status = "processing"
        file_obj.updated_at = datetime.utcnow()
        db.commit()

        analyzer: ImageQualityAnalyzer = app_state.image_quality
        upscale_engine: UpscaleEngine = app_state.upscale_engine
        ocr_service: OCRService = app_state.ocr_service

        report = analyzer.analyze(file_obj.original_path)
        file_obj.blur_score = report.blur_score
        file_obj.contrast_score = report.contrast_score
        file_obj.width = report.width
        file_obj.height = report.height
        file_obj.need_upscale = report.needs_upscale
        file_obj.notes = report.notes

        processed_path = Path(file_obj.original_path)
        used_upscale = False

        if report.needs_upscale:
            processed_dir = processed_path.parent
            upscale_target = processed_dir / f"processed_x{report.recommended_scale}.png"
            result = upscale_engine.upscale(
                file_obj.original_path,
                str(upscale_target),
                report.recommended_scale,
            )
            processed_path = Path(result.output_path)
            used_upscale = result.scale > 1
            file_obj.processed_path = str(processed_path)
        else:
            file_obj.processed_path = None

        file_obj.used_upscale = used_upscale
        db.commit()

        ocr_boxes = ocr_service.run(str(processed_path))

        # Clear previous detections
        for detection in list(file_obj.detections):
            db.delete(detection)
        db.flush()

        for box in ocr_boxes:
            detection = Detection(
                file_id=file_obj.id,
                bbox_x1=box.bbox[0],
                bbox_y1=box.bbox[1],
                bbox_x2=box.bbox[2],
                bbox_y2=box.bbox[3],
                text=box.text,
                confidence=box.confidence,
                source="auto",
                version=1,
            )
            db.add(detection)

        file_obj.status = "completed"
        file_obj.updated_at = datetime.utcnow()
        db.commit()

        payload = {
            "event": "file_processed",
            "file_id": file_obj.id,
            "status": file_obj.status,
        }
        schedule_async(app_state.websocket_manager.broadcast(payload))

    except Exception as exc:  # pylint: disable=broad-except
        logger.exception("Failed to process file %s", file_id)
        db.rollback()
        file_obj = db.query(File).filter(File.id == file_id).first()
        if file_obj:
            file_obj.status = "failed"
            file_obj.notes = f"Processing error: {exc}"
            file_obj.updated_at = datetime.utcnow()
            db.commit()
        payload = {
            "event": "file_failed",
            "file_id": file_id,
            "status": "failed",
            "message": str(exc),
        }
        if hasattr(app_state, "websocket_manager"):
            schedule_async(app_state.websocket_manager.broadcast(payload))
    finally:
        db.close()
