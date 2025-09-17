from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass
class ImageQualityReport:
    blur_score: float
    contrast_score: float
    width: int
    height: int
    recommended_scale: int
    needs_upscale: bool
    notes: Optional[str] = None


class ImageQualityAnalyzer:
    def __init__(self, blur_threshold: float = 120.0, contrast_threshold: float = 40.0):
        self.blur_threshold = blur_threshold
        self.contrast_threshold = contrast_threshold

    def analyze(self, image_path: str) -> ImageQualityReport:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Unable to read image at {image_path}")

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        contrast_score = float(np.std(gray))
        height, width = gray.shape

        recommended_scale = self._recommend_scale(width, height, blur_score, contrast_score)
        needs_upscale = recommended_scale > 1
        notes = None

        if blur_score < self.blur_threshold:
            notes = (notes or "") + "Low sharpness detected. "
        if contrast_score < self.contrast_threshold:
            notes = (notes or "") + "Low contrast detected. "
        if min(width, height) < 800:
            notes = (notes or "") + "Resolution is low. "

        if notes:
            notes = notes.strip()

        return ImageQualityReport(
            blur_score=blur_score,
            contrast_score=contrast_score,
            width=width,
            height=height,
            recommended_scale=recommended_scale,
            needs_upscale=needs_upscale,
            notes=notes,
        )

    def _recommend_scale(self, width: int, height: int, blur: float, contrast: float) -> int:
        scale = 1
        if min(width, height) < 800:
            scale = 2
        if blur < self.blur_threshold * 0.7:
            scale = max(scale, 2)
        if blur < self.blur_threshold * 0.4 or contrast < self.contrast_threshold * 0.5:
            scale = max(scale, 3)
        if min(width, height) < 500:
            scale = max(scale, 4)
        return min(scale, 4)
