from typing import Literal

from pydantic import Field
from scope.core.pipelines.base_schema import (
    BasePipelineConfig,
    ModeDefaults,
    UsageType,
    ui_field_config,
)
from scope.core.pipelines.artifacts import HuggingfaceRepoArtifact

TargetResolution = Literal["720p", "1080p", "2k", "4k"]
QualityMode = Literal["fast", "balanced", "quality"]


class WallspaceUpscalerConfig(BasePipelineConfig):
    """Resolution-targeting upscaler with optional RIFE frame smoothing."""

    pipeline_id = "wallspace-upscaler"
    pipeline_name = "WS Upscaler"
    pipeline_description = (
        "Upscales 360p/480p video to 720p, 1080p, 2K, or 4K using Real-ESRGAN "
        "with optional RIFE temporal smoothing"
    )
    pipeline_version = "0.1.4"
    estimated_vram_gb = 4.0
    supports_prompts = False
    supports_lora = False
    supports_vace = False
    supports_cache_management = False
    supports_quantization = False

    modes = {"video": ModeDefaults(default=True)}
    usage = [UsageType.POSTPROCESSOR]

    artifacts = [
        HuggingfaceRepoArtifact(
            repo_id="ai-forever/Real-ESRGAN",
            files=["RealESRGAN_x2.pth", "RealESRGAN_x4.pth"],
        ),
    ]

    # ── Load-time parameters (require pipeline reload) ──────────────────────

    target_resolution: TargetResolution = Field(
        default="1080p",
        description="Target output resolution for upscaled video",
        json_schema_extra=ui_field_config(
            order=1,
            label="Target Resolution",
            is_load_param=True,
            category="configuration",
        ),
    )

    quality_mode: QualityMode = Field(
        default="balanced",
        description="Fast = bicubic only, Balanced = ESRGAN x2, Quality = ESRGAN x4 multi-pass",
        json_schema_extra=ui_field_config(
            order=2,
            label="Quality Mode",
            is_load_param=True,
            category="configuration",
        ),
    )

    enable_rife: bool = Field(
        default=False,
        description="Enable RIFE 2x frame interpolation for smoother output",
        json_schema_extra=ui_field_config(
            order=3,
            label="RIFE Frame Smoothing",
            is_load_param=True,
            category="configuration",
        ),
    )

    # ── Runtime parameters (adjustable during streaming) ────────────────────

    sharpness: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Post-upscale sharpening intensity (unsharp mask)",
        json_schema_extra=ui_field_config(
            order=10,
            label="Sharpness",
            is_load_param=False,
            category="configuration",
        ),
    )

    denoise_strength: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Pre-upscale denoising to reduce ESRGAN artifacts on noisy input",
        json_schema_extra=ui_field_config(
            order=11,
            label="Denoise",
            is_load_param=False,
            category="configuration",
        ),
    )
