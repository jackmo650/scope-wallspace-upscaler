import logging
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from scope.core.pipelines.interface import Pipeline, Requirements

if TYPE_CHECKING:
    from scope.core.pipelines.base_schema import BasePipelineConfig

from .schema import WallspaceUpscalerConfig

logger = logging.getLogger(__name__)

RESOLUTION_MAP = {
    "720p": (720, 1280),
    "1080p": (1080, 1920),
    "2k": (1440, 2560),
    "4k": (2160, 3840),
}


class WallspaceUpscalerPipeline(Pipeline):
    """Resolution-targeting upscaler using Real-ESRGAN with optional RIFE."""

    @classmethod
    def get_config_class(cls) -> type["BasePipelineConfig"]:
        return WallspaceUpscalerConfig

    def __init__(self, device: torch.device | None = None, **kwargs):
        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.target_resolution = kwargs.get("target_resolution", "1080p")
        self.quality_mode = kwargs.get("quality_mode", "balanced")
        self.enable_rife = kwargs.get("enable_rife", False)

        self._esrgan_x2 = None
        self._esrgan_x4 = None
        self._rife_model = None
        self._prev_frame = None

        # Pre-compute target dims once at load time
        self._target_h, self._target_w = RESOLUTION_MAP.get(
            self.target_resolution, (1080, 1920)
        )

        self._load_models()

    # ── Model loading ───────────────────────────────────────────────────────

    def _get_model_path(self, filename: str) -> str:
        import os
        try:
            from scope.core.config import get_model_file_path
            return get_model_file_path("ai-forever/Real-ESRGAN", filename)
        except Exception:
            pass
        for base_dir in [
            os.environ.get("HF_HOME", ""),
            os.path.join(os.path.expanduser("~"), ".cache", "huggingface"),
            "/workspace/huggingface",
        ]:
            if not base_dir:
                continue
            snapshots = os.path.join(
                base_dir, "hub", "models--ai-forever--Real-ESRGAN", "snapshots"
            )
            if os.path.isdir(snapshots):
                for entry in os.listdir(snapshots):
                    candidate = os.path.join(snapshots, entry, filename)
                    if os.path.isfile(candidate):
                        return candidate
        logger.warning("Could not resolve model path for %s", filename)
        return filename

    def _load_models(self):
        if self.quality_mode == "fast":
            return
        try:
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer
        except ImportError:
            logger.warning("realesrgan/basicsr not installed — bicubic-only mode")
            self.quality_mode = "fast"
            return

        use_half = self.device.type == "cuda"
        net_x2 = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                         num_block=23, num_grow_ch=32, scale=2)
        self._esrgan_x2 = RealESRGANer(
            scale=2, model_path=self._get_model_path("RealESRGAN_x2.pth"),
            model=net_x2, device=self.device, half=use_half)

        if self.quality_mode == "quality" or self._target_h > 960:
            net_x4 = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                             num_block=23, num_grow_ch=32, scale=4)
            self._esrgan_x4 = RealESRGANer(
                scale=4, model_path=self._get_model_path("RealESRGAN_x4.pth"),
                model=net_x4, device=self.device, half=use_half)

        if self.enable_rife:
            self._load_rife()

    def _load_rife(self):
        for mod_path in ["scope_rife.ifnet", "rife.model"]:
            try:
                IFNet = __import__(mod_path, fromlist=["IFNet"]).IFNet
                self._rife_model = IFNet().to(self.device).eval()
                if self.device.type == "cuda":
                    self._rife_model = self._rife_model.half()
                return
            except (ImportError, AttributeError):
                continue
        logger.warning("RIFE not available — frame smoothing disabled")

    # ── Pipeline interface ──────────────────────────────────────────────────

    def prepare(self, **kwargs) -> Requirements:
        return Requirements(input_size=1)

    def __call__(self, **kwargs) -> dict:
        """Process video frames. Optimized to stay in tensor space for fast mode."""
        video = kwargs.get("video")
        if video is None:
            raise ValueError("WallspaceUpscalerPipeline requires video input")
        if not video:
            return {"video": torch.zeros(1, 1, 1, 3, device=self.device)}

        sharpness = kwargs.get("sharpness", 0.0)
        denoise_strength = kwargs.get("denoise_strength", 0.0)

        first = video[0]
        in_h, in_w = first.shape[1], first.shape[2]
        target_h, target_w = self._target_h, self._target_w

        # Determine if we need ESRGAN or can stay pure-tensor
        scale = max(target_h / max(in_h, 1), target_w / max(in_w, 1))
        use_esrgan = self.quality_mode != "fast" and scale > 1.0

        output_frames = []

        for frame_tensor in video:
            if use_esrgan:
                result = self._process_esrgan(frame_tensor, scale,
                                              target_h, target_w,
                                              denoise_strength, sharpness)
            else:
                result = self._process_fast(frame_tensor, target_h, target_w,
                                            denoise_strength, sharpness)

            # RIFE interpolation
            if self.enable_rife and self._rife_model is not None and self._prev_frame is not None:
                mid = self._interpolate_rife(result, self._prev_frame)
                output_frames.append(mid.squeeze(0).permute(1, 2, 0))

            self._prev_frame = result.detach()
            output_frames.append(result.squeeze(0).permute(1, 2, 0))

        return {"video": torch.stack(output_frames, dim=0)}

    # ── Fast path: pure tensor, no numpy, no model ─────────────────────────

    def _process_fast(self, frame: torch.Tensor, target_h: int, target_w: int,
                      denoise: float, sharpness: float) -> torch.Tensor:
        """Zero-copy bicubic upscale. Input: (1,H,W,C) [0,255] → BCHW [0,1]."""
        # (1, H, W, C) → (1, C, H, W), normalize
        t = frame.squeeze(0).permute(2, 0, 1).unsqueeze(0)
        t = t.to(device=self.device, dtype=torch.float32).mul_(1.0 / 255.0)

        # Denoise (blur blend) — all in tensor space
        if denoise > 0.0:
            k = int(3 + denoise * 4) | 1
            p = k // 2
            blurred = F.avg_pool2d(F.pad(t, [p, p, p, p], mode="reflect"), k, stride=1)
            t = t.lerp(blurred, denoise * 0.5)

        # Resize
        if t.shape[2] != target_h or t.shape[3] != target_w:
            t = F.interpolate(t, size=(target_h, target_w),
                              mode="bicubic", align_corners=False)

        # Sharpen
        if sharpness > 0.0:
            blurred = F.avg_pool2d(F.pad(t, [1, 1, 1, 1], mode="reflect"), 3, stride=1)
            t = t + sharpness * (t - blurred)

        return t.clamp_(0, 1)

    # ── ESRGAN path: requires numpy for model ──────────────────────────────

    def _process_esrgan(self, frame: torch.Tensor, scale: float,
                        target_h: int, target_w: int,
                        denoise: float, sharpness: float) -> torch.Tensor:
        """ESRGAN upscale. Falls through numpy for model inference only."""
        frame_np = frame.squeeze(0).clamp(0, 255).byte().cpu().numpy()

        # Denoise pre-pass
        if denoise > 0.0:
            t = torch.from_numpy(frame_np).permute(2, 0, 1).unsqueeze(0).float()
            t = t.to(self.device) / 255.0
            k = int(3 + denoise * 4) | 1
            p = k // 2
            blurred = F.avg_pool2d(F.pad(t, [p, p, p, p], mode="reflect"), k, stride=1)
            t = t.lerp(blurred, denoise * 0.5)
            frame_np = (t.squeeze(0).permute(1, 2, 0).clamp(0, 1) * 255).byte().cpu().numpy()

        # Choose ESRGAN strategy
        if scale <= 2.0:
            output, _ = self._esrgan_x2.enhance(frame_np, outscale=2)
        elif scale <= 4.0 and self.quality_mode == "quality" and self._esrgan_x4:
            output, _ = self._esrgan_x4.enhance(frame_np, outscale=4)
        elif scale > 4.0 and self._esrgan_x4:
            intermediate, _ = self._esrgan_x4.enhance(frame_np, outscale=4)
            output, _ = self._esrgan_x2.enhance(intermediate, outscale=2)
        else:
            output, _ = self._esrgan_x2.enhance(frame_np, outscale=2)

        # Back to tensor, resize to exact target
        t = torch.from_numpy(output).permute(2, 0, 1).unsqueeze(0).float()
        t = t.to(self.device) / 255.0
        if t.shape[2] != target_h or t.shape[3] != target_w:
            t = F.interpolate(t, size=(target_h, target_w),
                              mode="bicubic", align_corners=False)

        if sharpness > 0.0:
            blurred = F.avg_pool2d(F.pad(t, [1, 1, 1, 1], mode="reflect"), 3, stride=1)
            t = t + sharpness * (t - blurred)

        return t.clamp_(0, 1)

    # ── RIFE ────────────────────────────────────────────────────────────────

    def _interpolate_rife(self, current: torch.Tensor, prev: torch.Tensor) -> torch.Tensor:
        if self._rife_model is None:
            return current
        with torch.no_grad():
            dtype = torch.float16 if self.device.type == "cuda" else torch.float32
            mid = self._rife_model(prev.to(dtype=dtype), current.to(dtype=dtype))
        return mid.float().clamp_(0, 1)
