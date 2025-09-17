import logging
from dataclasses import dataclass
from typing import List, Optional

import cv2

from paddleocr import PaddleOCR

try:
    import paddle  # type: ignore
except ImportError:  # pragma: no cover - paddle is an optional runtime dep
    paddle = None

logger = logging.getLogger(__name__)

ALLOWED_CHAR_SET = set("0123456789ABCDE-")


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
        enable_angle_cls: bool = False,
    ) -> None:
        init_kwargs = {
            "lang": lang,
            "use_angle_cls": enable_angle_cls,
        }
        if det_model_dir:
            init_kwargs["det_model_dir"] = det_model_dir
        if rec_model_dir:
            init_kwargs["rec_model_dir"] = rec_model_dir

        self.use_angle_cls = enable_angle_cls

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
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图片: {image_path}")

        try:
            dt_boxes, rec_res, _ = self.ocr.__call__(image, cls=self.use_angle_cls)
        except TypeError:
            dt_boxes, rec_res, _ = self.ocr.__call__(image)

        boxes: List[OCRBox] = []
        if dt_boxes and rec_res:
            for poly, rec in zip(dt_boxes, rec_res):
                if isinstance(rec, (list, tuple)) and len(rec) >= 2:
                    text, score = rec[0], rec[1]
                else:
                    text, score = rec, 1.0
                self._append_box(boxes, poly, text, score)

        logger.info("PaddleOCR 返回 %s 个候选框", len(boxes))
        for idx, box in enumerate(boxes, start=1):
            logger.info(
                "结果 #%d 文本=%s 置信度=%.3f 区域=%s",
                idx,
                box.text,
                box.confidence,
                box.bbox,
            )
        if not boxes:
            logger.warning("OCR 未返回有效检测框")
        return boxes

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
        if text.isalpha() and "O" in text:
            text = text.replace("O", "0")
        filtered = "".join(ch for ch in text if ch in ALLOWED_CHAR_SET)
        if not filtered:
            logger.debug("过滤后文本为空，原始文本: %s", text)
            return None
        if filtered != text:
            logger.debug("识别结果包含非受限字符，已过滤: %s -> %s", text, filtered)
        return filtered

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
