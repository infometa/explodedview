import inspect
import logging
import re
from dataclasses import dataclass
from typing import List, Optional

from paddleocr import PaddleOCR

from ..config import OCR_CHAR_DICT

try:
    import paddle  # type: ignore
except ImportError:  # pragma: no cover - paddle is an optional runtime dep
    paddle = None

logger = logging.getLogger(__name__)

ALLOWED_CHAR_PATTERN = re.compile(r"^[0-9A-E-]+$")


@dataclass
class OCRBox:
    text: str
    confidence: float
    bbox: List[float]


class OCRService:
    def __init__(
        self,
        lang: str = "en",
        det_model_dir: Optional[str] = None,
        rec_model_dir: Optional[str] = None,
        use_gpu: Optional[bool] = None,
    ) -> None:
        custom_dict = str(OCR_CHAR_DICT) if OCR_CHAR_DICT.exists() else None
        init_kwargs = {
            "lang": lang,
            "use_angle_cls": True,
        }
        if det_model_dir:
            init_kwargs["det_model_dir"] = det_model_dir
        if rec_model_dir:
            init_kwargs["rec_model_dir"] = rec_model_dir
        if custom_dict:
            init_kwargs["rec_char_dict_path"] = custom_dict

        gpu_enabled = self._resolve_gpu_flag(use_gpu)
        if gpu_enabled:
            self._configure_gpu_device()

        def _create_ocr(**extra_kwargs):
            kwargs = {**init_kwargs, **extra_kwargs}
            while True:
                try:
                    return PaddleOCR(**kwargs)
                except (TypeError, ValueError) as exc:
                    message = str(exc)
                    if "Unknown argument" in message:
                        unknown = message.split(":", 1)[-1].strip().strip(" '\"")
                        if unknown in kwargs:
                            kwargs.pop(unknown)
                            continue
                    raise

        extra = {}
        if gpu_enabled:
            extra["use_gpu"] = True

        try:
            self.ocr = _create_ocr(rec_algorithm="SVTR_LCNet", **extra)
        except (TypeError, ValueError):
            self.ocr = _create_ocr(**extra)

    def _resolve_gpu_flag(self, explicit_flag: Optional[bool]) -> bool:
        if explicit_flag is False:
            return False
        available = self._gpu_available()
        if explicit_flag is True and not available:
            logger.warning("GPU requested but not available; falling back to CPU mode")
            return False
        if explicit_flag is None:
            return available
        return explicit_flag and available

    def _gpu_available(self) -> bool:
        if paddle is None:
            return False
        try:
            compiled = paddle.device.is_compiled_with_cuda()
        except AttributeError:  # pragma: no cover
            compiled = getattr(paddle, "is_compiled_with_cuda", lambda: False)()
        if not compiled:
            return False
        try:
            cuda_ns = getattr(paddle.device, "cuda", None)
            if cuda_ns and hasattr(cuda_ns, "device_count"):
                return cuda_ns.device_count() > 0
        except Exception:  # pragma: no cover - defensive
            pass
        return True

    def _configure_gpu_device(self) -> None:
        if paddle is None:
            logger.warning("Paddle is not installed; cannot enable GPU mode")
            return
        try:
            if hasattr(paddle.device, "set_device"):
                paddle.device.set_device("gpu")
            else:  # pragma: no cover - legacy fallback
                paddle.set_device("gpu")
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Failed to switch Paddle to GPU: %s", exc)

    def run(self, image_path: str) -> List[OCRBox]:
        ocr_callable = getattr(self.ocr, "ocr")
        signature_params = set(inspect.signature(ocr_callable).parameters.keys())
        extra_kwargs = {}
        if "cls" in signature_params:
            extra_kwargs["cls"] = True

        result = ocr_callable(image_path, **extra_kwargs)
        return self._parse_result(result)

    def _parse_result(self, result: object) -> List[OCRBox]:
        boxes: List[OCRBox] = []

        if not result:
            return boxes

        if isinstance(result, tuple) and len(result) == 2:
            dt_boxes, rec_res = result
            if isinstance(dt_boxes, (list, tuple)) and isinstance(rec_res, (list, tuple)):
                for bbox, rec in zip(dt_boxes, rec_res):
                    text, score = rec if isinstance(rec, (list, tuple)) and len(rec) >= 2 else (rec, 1.0)
                    self._append_box(boxes, bbox, text, score)
                return boxes

        if isinstance(result, list) and result and isinstance(result[0], dict):
            for item in result:
                if not isinstance(item, dict):
                    continue
                bbox = item.get("bbox") or item.get("box") or item.get("points") or item.get("dt_box")
                text = item.get("text") or item.get("transcription") or item.get("rec_text")
                score = item.get("confidence") or item.get("score") or item.get("rec_score")
                if bbox is None or text is None:
                    continue
                self._append_box(boxes, bbox, text, score if score is not None else 1.0)
            return boxes

        if isinstance(result, list):
            for line in result:
                self._parse_line_entry(line, boxes)
            return boxes

        return boxes

    def _parse_line_entry(self, entry: object, boxes: List[OCRBox]) -> None:
        if isinstance(entry, dict):
            bbox = entry.get("bbox") or entry.get("box") or entry.get("points") or entry.get("dt_box")
            text = entry.get("text") or entry.get("transcription") or entry.get("rec_text")
            score = entry.get("confidence") or entry.get("score") or entry.get("rec_score")
            if bbox is None or text is None:
                return
            self._append_box(boxes, bbox, text, score if score is not None else 1.0)
            return

        if isinstance(entry, (list, tuple)):
            if len(entry) >= 2 and self._looks_like_bbox(entry[0]):
                bbox_candidate = entry[0]
                rec = entry[1]
                if isinstance(rec, (list, tuple)) and len(rec) >= 2:
                    text, score = rec[0], rec[1]
                else:
                    text, score = rec, 1.0
                self._append_box(boxes, bbox_candidate, text, score)
                return

            if all(isinstance(item, (list, tuple, dict)) for item in entry):
                for item in entry:
                    self._parse_line_entry(item, boxes)
                return

        # For any other type we ignore silently.

    def _looks_like_bbox(self, value: object) -> bool:
        if value is None:
            return False
        if hasattr(value, "tolist"):
            value = value.tolist()
        if isinstance(value, dict):
            return any(key in value for key in ("points", "bbox", "box", "dt_box"))
        if isinstance(value, (list, tuple)):
            if len(value) == 4 and all(isinstance(v, (int, float)) for v in value):
                return True
            if value and all(isinstance(p, (list, tuple)) and len(p) >= 2 for p in value):
                return True
        return False

    def _append_box(self, boxes: List[OCRBox], bbox: object, text: object, score: object) -> None:
        cleaned_text = self._clean_text(str(text)) if text is not None else None
        if not cleaned_text:
            return
        try:
            confidence = float(score) if score is not None else 1.0
        except (TypeError, ValueError):
            confidence = 1.0

        try:
            flat_box = self._to_bbox(bbox)
        except ValueError:
            logger.warning("跳过不支持的bbox格式: %s", bbox)
            return
        boxes.append(OCRBox(text=cleaned_text, confidence=confidence, bbox=flat_box))

    def _clean_text(self, text: str) -> Optional[str]:
        text = text.strip().upper()
        text = text.replace(" ", "")
        text = text.replace("O", "0") if text.isalpha() and "O" in text else text
        return text if ALLOWED_CHAR_PATTERN.match(text) else None

    @staticmethod
    def _to_bbox(points: object) -> List[float]:
        if points is None:
            return [0.0, 0.0, 0.0, 0.0]

        if hasattr(points, "tolist"):
            points = points.tolist()

        if isinstance(points, dict):
            points = points.get("points") or points.get("bbox") or points.get("box")

        if isinstance(points, (list, tuple)):
            if len(points) == 4 and all(isinstance(v, (int, float)) for v in points):
                x1, y1, x2, y2 = points
                return [float(x1), float(y1), float(x2), float(y2)]
            if len(points) == 8 and all(isinstance(v, (int, float)) for v in points):
                coords = list(zip(points[0::2], points[1::2]))
                return OCRService._to_bbox(coords)
            if points and all(isinstance(p, (list, tuple)) and len(p) >= 2 for p in points):
                xs = [float(p[0]) for p in points]
                ys = [float(p[1]) for p in points]
                return [min(xs), min(ys), max(xs), max(ys)]

        raise ValueError(f"Unsupported bbox format: {points}")
