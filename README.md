# WS Upscaler

Resolution-targeting upscaler for [Daydream Scope](https://daydream.live). Upscales 360p/480p video to 720p, 1080p, 2K, or 4K using Real-ESRGAN with optional RIFE temporal smoothing.

## Features

- **Target resolution**: 720p, 1080p, 2K, 4K
- **Quality modes**: Fast (bicubic), Balanced (ESRGAN x2), Quality (ESRGAN x4 multi-pass)
- **RIFE frame smoothing**: Optional 2x frame interpolation
- **Runtime controls**: Sharpness and denoise sliders adjustable during streaming
- **Device support**: CUDA, Apple Silicon (MPS), CPU

## Install

```bash
uv pip install "scope-wallspace-upscaler @ git+https://github.com/jackmo650/scope-wallspace-upscaler"
```

Or install from the Daydream Scope plugin manager.

## Usage

Load as a **postprocessor** in Scope. Pair with any pipeline (Passthrough, LongLive, etc.) to upscale output to your target resolution.

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| Target Resolution | 720p / 1080p / 2k / 4k | 1080p | Output resolution |
| Quality Mode | fast / balanced / quality | balanced | Upscale strategy |
| RIFE Frame Smoothing | on / off | off | 2x frame interpolation |
| Sharpness | 0.0 - 1.0 | 0.0 | Post-upscale sharpening |
| Denoise | 0.0 - 1.0 | 0.0 | Pre-upscale denoising |

## Development Workflow

This project follows a **human-in-the-loop** development process:

1. **All requests start as GitHub Issues** — bugs, features, tasks, and experiments are logged using the provided issue templates.
2. **Issues are reviewed and triaged** — the maintainer reviews each issue, adjusts scope, and assigns priority.
3. **Only `approved` issues move forward** — no implementation begins until an issue is explicitly labeled `approved`.
4. **Implementation happens on explicit instruction** — coding agents and contributors only work on approved, assigned work.
5. **Pull requests reference an approved issue** — every PR must link back to the issue it addresses.

