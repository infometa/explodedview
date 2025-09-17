import inspect
import logging
import re
from dataclasses import dataclass
from typing import Any, List, Optional

import numpy as np

from paddleocr import PaddleOCR

try:
    import paddle  # type: ignore
except ImportError:  # pragma: no cover - paddle is an optional runtime dep
    paddle = None

logger = logging.getLogger(__name__)

ALLOWED_CHAR_PATTERN = re.compile(r"^[0-9A-E-]+$")
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
        ocr_callable = getattr(self.ocr, "ocr")
        signature_params = set(inspect.signature(ocr_callable).parameters.keys())
        extra_kwargs = {}
        if "cls" in signature_params and self.use_angle_cls:
            extra_kwargs["cls"] = True

        result = ocr_callable(image_path, **extra_kwargs)
        boxes = self._parse_result(result)
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
            preview = repr(result)
            if isinstance(preview, str) and len(preview) > 1000:
                preview = preview[:1000] + "..."
            logger.warning("OCR 原始返回为空或无法解析: %s", preview)
        return boxes

    def _parse_result(self, result: object) -> List[OCRBox]:
        boxes: List[OCRBox] = []

        if not result:
            return boxes

        visited = set()

        def visit(node: Any) -> None:
            node_id = id(node)
            if node_id in visited:
                return
            visited.add(node_id)

            if node is None:
                return

            if isinstance(node, dict):
                if "dt_polys" in node and "rec_texts" in node:
                    self._append_dt_polys(node, boxes)
                    return
                for key, value in node.items():
                    if key in {"input_img", "rot_img", "output_img", "rot_mat"}:
                        continue
                    visit(value)
                return

            if isinstance(node, (list, tuple)):
                if len(node) == 2 and self._looks_like_bbox(node[0]):
                    bbox_candidate = node[0]
                    rec = node[1]
                    if isinstance(rec, (list, tuple)) and len(rec) >= 2:
                        text, score = rec[0], rec[1]
                    else:
                        text, score = rec, 1.0
                    self._append_box(boxes, bbox_candidate, text, score)
                    return

                for item in node:
                    visit(item)
                return

            if isinstance(node, np.ndarray):
                return

        visit(result)
        return boxes

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

    def _reverse_orientation(
        self, poly: object, angle: float, width: int, height: int
    ) -> List[List[float]]:
        """将文档预处理旋转后的多边形坐标还原到原始坐标系."""

        if hasattr(poly, "tolist"):
            poly = poly.tolist()

        arr = np.asarray(poly, dtype=float)
        if arr.ndim == 1:
            if arr.size == 4:
                arr = arr.reshape(2, 2)
            elif arr.size == 8:
                arr = arr.reshape(4, 2)
            else:
                raise ValueError(f"无法处理的多边形形状: {poly}")

        if arr.ndim != 2 or arr.shape[1] < 2:
            raise ValueError(f"无法处理的多边形形状: {poly}")

        if angle % 360 == 0:
            return arr.tolist()

        angle_norm = int(angle) % 360

        if angle_norm == 180:
            transformed = np.empty_like(arr)
            transformed[:, 0] = width - 1 - arr[:, 0]
            transformed[:, 1] = height - 1 - arr[:, 1]
        elif angle_norm == 90:
            transformed = np.empty_like(arr)
            transformed[:, 0] = arr[:, 1]
            transformed[:, 1] = width - 1 - arr[:, 0]
        elif angle_norm == 270:
            transformed = np.empty_like(arr)
            transformed[:, 0] = height - 1 - arr[:, 1]
            transformed[:, 1] = arr[:, 0]
        else:
            logger.debug("未处理的旋转角度 %s，返回原坐标", angle)
            return arr.tolist()

        return transformed.tolist()

    def _append_dt_polys(self, entry: dict, boxes: List[OCRBox]) -> None:
        polys = entry.get("dt_polys") or entry.get("det_polys")
        texts = entry.get("rec_texts") or entry.get("texts")
        scores = entry.get("rec_scores") or entry.get("scores") or []

        if not polys or not texts:
            return

        angle = None
        image_width = None
        image_height = None
        doc_res = entry.get("doc_preprocessor_res") or {}
        if isinstance(doc_res, dict):
            angle = doc_res.get("angle")
            img_ref = doc_res.get("input_img")
            if img_ref is None:
                img_ref = doc_res.get("output_img")
            if isinstance(img_ref, np.ndarray):
                image_height, image_width = img_ref.shape[:2]

        for idx, poly in enumerate(polys):
            text = texts[idx] if idx < len(texts) else ""
            score = scores[idx] if idx < len(scores) else 1.0
            if (
                angle is not None
                and image_width is not None
                and image_height is not None
                and angle % 360 != 0
            ):
                poly = self._reverse_orientation(poly, angle, image_width, image_height)
            self._append_box(boxes, poly, text, score)

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
