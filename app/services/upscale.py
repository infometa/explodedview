from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2


@dataclass
class UpscaleResult:
    output_path: str
    scale: int
    engine: str


class UpscaleEngine:
    def __init__(self, preferred_model: str = "realesr-general-x4v3"):
        self.preferred_model = preferred_model
        self._realesrgan = None
        self._load_realesrgan()

    def _load_realesrgan(self) -> None:
        try:
            from realesrgan import RealESRGANer
            from basicsr.archs.rrdbnet_arch import RRDBNet
        except ImportError:
            self._realesrgan = None
            return

        model_name = self.preferred_model
        model_configs = {
            "realesr-general-x4v3": {
                "model_path": "weights/realesr-general-x4v3.pth",
                "netscale": 4,
                "rrdb_blocks": 23,
                "rrdb_filters": 64,
            },
            "realesr-anime-x4plus": {
                "model_path": "weights/RealESRGAN_x4plus_anime_6B.pth",
                "netscale": 4,
                "rrdb_blocks": 6,
                "rrdb_filters": 64,
            },
        }

        config = model_configs.get(model_name)
        if not config:
            self._realesrgan = None
            return

        model_path = Path(config["model_path"])
        if not model_path.exists():
            # Defer loading until weights are available.
            self._realesrgan = (RealESRGANer, config)
            return

        net = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=config["rrdb_filters"],
            num_block=config["rrdb_blocks"],
            num_grow_ch=32,
            scale=config["netscale"],
        )

        self._realesrgan = RealESRGANer(
            scale=config["netscale"],
            model_path=str(model_path),
            model=net,
            tile=0,
            tile_pad=10,
            pre_pad=0,
            half=False,
            device="cpu",
        )

    def upscale(self, image_path: str, output_path: str, scale: int) -> UpscaleResult:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Unable to read image at {image_path}")

        if self._supports_realesrgan():
            result = self._run_realesrgan(image, output_path, scale)
            if result:
                return result

        resized = self._run_opencv_resize(image, scale)
        cv2.imwrite(output_path, resized)
        return UpscaleResult(output_path=output_path, scale=scale, engine="opencv")

    def _supports_realesrgan(self) -> bool:
        return self._realesrgan is not None and not isinstance(self._realesrgan, tuple)

    def _run_realesrgan(self, image, output_path: str, scale: int) -> Optional[UpscaleResult]:
        if isinstance(self._realesrgan, tuple):
            RealESRGANer, config = self._realesrgan
            model_path = Path(config["model_path"])
            if not model_path.exists():
                return None
            try:
                from basicsr.archs.rrdbnet_arch import RRDBNet
            except ImportError:
                return None
            net = RRDBNet(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=config["rrdb_filters"],
                num_block=config["rrdb_blocks"],
                num_grow_ch=32,
                scale=config["netscale"],
            )
            self._realesrgan = RealESRGANer(
                scale=config["netscale"],
                model_path=str(model_path),
                model=net,
                tile=0,
                tile_pad=10,
                pre_pad=0,
                half=False,
                device="cpu",
            )
        engine = self._realesrgan

        engine = self._realesrgan
        current_scale = getattr(engine, "scale", 4)
        target_scale = min(scale, current_scale)
        output, _ = engine.enhance(image, outscale=target_scale)
        cv2.imwrite(output_path, output)
        return UpscaleResult(output_path=output_path, scale=target_scale, engine="realesrgan")

    @staticmethod
    def _run_opencv_resize(image, scale: int):
        interpolation = cv2.INTER_CUBIC if scale <= 2 else cv2.INTER_LANCZOS4
        height, width = image.shape[:2]
        return cv2.resize(image, (width * scale, height * scale), interpolation=interpolation)
