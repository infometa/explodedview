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
        boxes: List[OCRBox] = []
        for line in result:
            for bbox, (text, score) in line:
                cleaned_text = self._clean_text(text)
                if not cleaned_text:
                    continue
                flat_box = self._to_bbox(bbox)
                boxes.append(OCRBox(text=cleaned_text, confidence=float(score), bbox=flat_box))
        return boxes

    def _clean_text(self, text: str) -> Optional[str]:
        text = text.strip().upper()
        text = text.replace(" ", "")
        text = text.replace("O", "0") if text.isalpha() and "O" in text else text
        return text if ALLOWED_CHAR_PATTERN.match(text) else None

    @staticmethod
    def _to_bbox(points: List[List[float]]) -> List[float]:
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        x1, y1 = min(xs), min(ys)
        x2, y2 = max(xs), max(ys)
        return [float(x1), float(y1), float(x2), float(y2)]
