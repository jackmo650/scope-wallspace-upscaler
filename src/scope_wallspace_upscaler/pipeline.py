import logging
from typing import TYPE_CHECKING

import torch

from scope.core.pipelines.interface import Pipeline, Requirements

if TYPE_CHECKING:
    from scope.core.pipelines.base_schema import BasePipelineConfig

from .schema import WallspaceUpscalerConfig

logger = logging.getLogger(__name__)

# Target resolution lookup: name -> (height, width)
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

        # Load-time config
        self.target_resolution = kwargs.get("target_resolution", "1080p")
        self.quality_mode = kwargs.get("quality_mode", "balanced")
        self.enable_rife = kwargs.get("enable_rife", False)

        # Model handles (lazy-loaded)
        self._esrgan_x2 = None
        self._esrgan_x4 = None
        self._rife_model = None

        # RIFE temporal state
        self._prev_frame = None

        # Load models based on quality mode
        self._load_models()

    # ── Model loading ───────────────────────────────────────────────────────

    def _get_model_path(self, filename: str) -> str:
        """Resolve model file path via Scope's artifact system."""
        import os

        # Try Scope's built-in resolver first
        try:
            from scope.core.config import get_model_file_path
            return get_model_file_path("ai-forever/Real-ESRGAN", filename)
        except Exception:
            pass

        # Fallback: search common HuggingFace cache locations
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

        logger.warning(
            "Could not resolve model path for %s, using filename as fallback",
            filename,
        )
        return filename

    def _load_models(self):
        """Load ESRGAN and optionally RIFE models."""
        if self.quality_mode == "fast":
            # Bicubic only — no models needed
            return

        try:
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer
        except ImportError:
            logger.warning(
                "realesrgan/basicsr not installed — falling back to bicubic-only mode"
            )
            self.quality_mode = "fast"
            return

        use_half = self.device.type == "cuda"

        # Always load x2 (used in balanced mode and as second pass for large scales)
        net_x2 = RRDBNet(
            num_in_ch=3, num_out_ch=3, num_feat=64,
            num_block=23, num_grow_ch=32, scale=2,
        )
        self._esrgan_x2 = RealESRGANer(
            scale=2,
            model_path=self._get_model_path("RealESRGAN_x2.pth"),
            model=net_x2,
            device=self.device,
            half=use_half,
        )

        # Load x4 for quality mode or when input requires >2x scale
        target_h, _ = RESOLUTION_MAP.get(self.target_resolution, (1080, 1920))
        if self.quality_mode == "quality" or target_h > 960:
            net_x4 = RRDBNet(
                num_in_ch=3, num_out_ch=3, num_feat=64,
                num_block=23, num_grow_ch=32, scale=4,
            )
            self._esrgan_x4 = RealESRGANer(
                scale=4,
                model_path=self._get_model_path("RealESRGAN_x4.pth"),
                model=net_x4,
                device=self.device,
                half=use_half,
            )

        # Load RIFE if enabled
        if self.enable_rife:
            self._load_rife()

    def _load_rife(self):
        """Lazy-load RIFE model for frame interpolation."""
        try:
            from scope_rife.ifnet import IFNet

            self._rife_model = IFNet().to(self.device).eval()
            if self.device.type == "cuda":
                self._rife_model = self._rife_model.half()
        except ImportError:
            try:
                from rife.model import IFNet

                self._rife_model = IFNet().to(self.device).eval()
                if self.device.type == "cuda":
                    self._rife_model = self._rife_model.half()
            except ImportError:
                logger.warning("RIFE not available — frame smoothing disabled")
                self._rife_model = None

    # ── Upscale planning ────────────────────────────────────────────────────

    def _compute_plan(self, in_h: int, in_w: int):
        """Determine upscale strategy based on input dims and settings.

        Returns: (strategy, target_h, target_w)
        """
        target_h, target_w = RESOLUTION_MAP.get(
            self.target_resolution, (1080, 1920)
        )
        scale_h = target_h / max(in_h, 1)
        scale_w = target_w / max(in_w, 1)
        scale = max(scale_h, scale_w)

        if self.quality_mode == "fast" or scale <= 1.0:
            return "bicubic", target_h, target_w

        if scale <= 2.0:
            return "esrgan_x2", target_h, target_w

        if scale <= 4.0:
            if self.quality_mode == "quality" and self._esrgan_x4 is not None:
                return "esrgan_x4", target_h, target_w
            return "esrgan_x2", target_h, target_w

        # >4x: multi-pass (x4 then x2) for maximum quality
        if self._esrgan_x4 is not None:
            return "esrgan_multi", target_h, target_w
        return "esrgan_x2", target_h, target_w

    # ── Frame processing ────────────────────────────────────────────────────

    def _upscale_frame(self, frame_np, plan: str,
                       target_h: int, target_w: int) -> torch.Tensor:
        """Upscale a single HWC uint8 numpy frame. Returns BCHW float32 [0,1]."""
        import torch.nn.functional as F

        if plan == "bicubic":
            t = torch.from_numpy(frame_np).permute(2, 0, 1).unsqueeze(0).float()
            t = t.to(self.device) / 255.0
            return F.interpolate(t, size=(target_h, target_w),
                                 mode="bicubic", align_corners=False).clamp(0, 1)

        if plan == "esrgan_x2":
            output, _ = self._esrgan_x2.enhance(frame_np, outscale=2)
            t = torch.from_numpy(output).permute(2, 0, 1).unsqueeze(0).float()
            t = t.to(self.device) / 255.0
            return F.interpolate(t, size=(target_h, target_w),
                                 mode="bicubic", align_corners=False).clamp(0, 1)

        if plan == "esrgan_x4":
            output, _ = self._esrgan_x4.enhance(frame_np, outscale=4)
            t = torch.from_numpy(output).permute(2, 0, 1).unsqueeze(0).float()
            t = t.to(self.device) / 255.0
            return F.interpolate(t, size=(target_h, target_w),
                                 mode="bicubic", align_corners=False).clamp(0, 1)

        if plan == "esrgan_multi":
            intermediate, _ = self._esrgan_x4.enhance(frame_np, outscale=4)
            output, _ = self._esrgan_x2.enhance(intermediate, outscale=2)
            t = torch.from_numpy(output).permute(2, 0, 1).unsqueeze(0).float()
            t = t.to(self.device) / 255.0
            return F.interpolate(t, size=(target_h, target_w),
                                 mode="bicubic", align_corners=False).clamp(0, 1)

        # Fallback
        t = torch.from_numpy(frame_np).permute(2, 0, 1).unsqueeze(0).float()
        return (t.to(self.device) / 255.0).clamp(0, 1)

    def _apply_sharpening(self, tensor: torch.Tensor, strength: float) -> torch.Tensor:
        """Unsharp mask on BCHW tensor."""
        import torch.nn.functional as F

        if strength <= 0.0:
            return tensor
        blurred = F.avg_pool2d(
            F.pad(tensor, [1, 1, 1, 1], mode="reflect"),
            kernel_size=3, stride=1,
        )
        sharpened = tensor + strength * (tensor - blurred)
        return sharpened.clamp(0, 1)

    def _apply_denoise(self, frame_np, strength: float):
        """Simple pre-upscale denoise via weighted blur blend."""
        import numpy as np
        import torch.nn.functional as F

        if strength <= 0.0:
            return frame_np
        t = torch.from_numpy(frame_np).permute(2, 0, 1).unsqueeze(0).float()
        t = t.to(self.device) / 255.0
        k = int(3 + strength * 4) | 1
        pad = k // 2
        blurred = F.avg_pool2d(
            F.pad(t, [pad, pad, pad, pad], mode="reflect"),
            kernel_size=k, stride=1,
        )
        blend = strength * 0.5
        result = t * (1.0 - blend) + blurred * blend
        return (result.squeeze(0).permute(1, 2, 0).clamp(0, 1) * 255).byte().cpu().numpy()

    def _interpolate_rife(self, current_bchw: torch.Tensor,
                          prev_bchw: torch.Tensor) -> torch.Tensor:
        """Generate an interpolated mid-frame between prev and current using RIFE."""
        if self._rife_model is None:
            return current_bchw
        with torch.no_grad():
            dtype = torch.float16 if self.device.type == "cuda" else torch.float32
            mid = self._rife_model(
                prev_bchw.to(dtype=dtype),
                current_bchw.to(dtype=dtype),
            )
        return mid.float().clamp(0, 1)

    # ── Pipeline interface ──────────────────────────────────────────────────

    def prepare(self, **kwargs) -> Requirements:
        return Requirements(input_size=1)

    def __call__(self, **kwargs) -> dict:
        """Process video frames through the upscaler.

        Input:  kwargs["video"] = list of N tensors, each (1, H, W, C) float32 [0, 255]
        Output: {"video": tensor} in THWC format, float32, [0, 1]
        """
        import numpy as np

        video = kwargs.get("video")
        if video is None:
            raise ValueError("WallspaceUpscalerPipeline requires video input")
        if not video:
            return {"video": torch.zeros(1, 1, 1, 3, device=self.device)}

        # Runtime params from kwargs (never from self)
        sharpness = kwargs.get("sharpness", 0.0)
        denoise_strength = kwargs.get("denoise_strength", 0.0)

        # Determine upscale plan from first frame dimensions
        first = video[0]  # (1, H, W, C)
        in_h, in_w = first.shape[1], first.shape[2]
        plan, target_h, target_w = self._compute_plan(in_h, in_w)

        output_frames = []

        for frame_tensor in video:
            # Convert to numpy HWC uint8
            frame_np = frame_tensor.squeeze(0).clamp(0, 255).byte().cpu().numpy()

            # Optional pre-upscale denoise
            if denoise_strength > 0.0:
                frame_np = self._apply_denoise(frame_np, denoise_strength)

            # Upscale -> BCHW float32 [0,1]
            result_bchw = self._upscale_frame(frame_np, plan, target_h, target_w)

            # Optional post-upscale sharpening
            if sharpness > 0.0:
                result_bchw = self._apply_sharpening(result_bchw, sharpness)

            # Optional RIFE frame interpolation (insert mid-frame before current)
            if self.enable_rife and self._rife_model is not None and self._prev_frame is not None:
                mid_bchw = self._interpolate_rife(result_bchw, self._prev_frame)
                mid_hwc = mid_bchw.squeeze(0).permute(1, 2, 0)
                output_frames.append(mid_hwc)

            # Cache current frame for next RIFE call
            self._prev_frame = result_bchw.detach()

            # Convert current frame to HWC
            result_hwc = result_bchw.squeeze(0).permute(1, 2, 0)
            output_frames.append(result_hwc)

        # Stack to THWC: (T, H, W, C) float32 [0, 1]
        output = torch.stack(output_frames, dim=0)
        return {"video": output}
