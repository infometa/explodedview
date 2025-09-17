import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from paddleocr import PaddleOCR

from ..config import OCR_CHAR_DICT

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
        use_gpu: bool = False,
    ) -> None:
        custom_dict = str(OCR_CHAR_DICT) if OCR_CHAR_DICT.exists() else None
        init_kwargs = {
            "lang": lang,
            "use_angle_cls": True,
            "det_model_dir": det_model_dir,
            "rec_model_dir": rec_model_dir,
            "rec_char_dict_path": custom_dict,
        }
        # Keep use_gpu only if the installed PaddleOCR supports it.
        if use_gpu is not None:
            init_kwargs["use_gpu"] = use_gpu

        def _create_ocr(**extra):
            kwargs = {**init_kwargs, **extra}
            try:
                return PaddleOCR(**kwargs)
            except (TypeError, ValueError) as exc:
                message = str(exc)
                if "Unknown argument" in message:
                    unknown = message.split(":", 1)[-1].strip()
                    unknown = unknown.strip(" '\"")
                    init_kwargs.pop(unknown, None)
                    kwargs.pop(unknown, None)
                    return PaddleOCR(**kwargs)
                raise

        try:
            self.ocr = _create_ocr(rec_algorithm="SVTR_LCNet")
        except (TypeError, ValueError):
            # rec_algorithm argument not supported; fallback to defaults.
            self.ocr = _create_ocr()

    def run(self, image_path: str) -> List[OCRBox]:
        result = self.ocr.ocr(image_path, cls=True)
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
