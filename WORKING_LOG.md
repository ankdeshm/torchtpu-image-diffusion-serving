# Working Log — Image Diffusion Serving on TorchTPU

## Goal
Evaluate and compare serving performance of image diffusion models on Google Cloud TPU v7x and TPU v5p using PyTorch / TorchTPU across `vllm-torchtpu` (`v0.33.0` + `vllm-omni 0.33.0rc1`) and `sglang-torchtpu` (`main`).

## What We Tried & Found

### 1. Baseline Setup & Dual-Engine Bring-Up
- Deployed standard `vllm-torchtpu v0.33.0` + `vllm-omni 0.33.0rc1` (`OmniTpuPlatform`) and `sglang-torchtpu` (`main`) on 4-chip TPU v7x (`8` TensorCores) and 4-chip TPU v5p (`4` chips) using `torch 2.13.0`, `torch_tpu 0.2.0`, and `libtpu 0.0.50`.
- Enabled `tpu_splash_attn` on TPU v7x (`U=8`) and PyTorch SDPA on TPU v5p (`U=4`) in `sglang-torchtpu`, and Ulysses Sequence Parallelism (`U=4` / `U=2`) in `vllm-omni`.

### 2. Catching Silent Blank Outputs on `FLUX.2-klein-4B`
- Added automatic PNG decoding and per-channel RGB standard deviation (`pixel_std`) checks to every benchmark call (`valid_image = pixel_std > 2.0`).
- **Issue:** On `sglang-torchtpu` (`main`), passing `--enable-torch-compile` for `FLUX.2-klein-4B` returned `HTTP 200 OK` PNGs at `182–235 img/min`, but every image was completely black (`pixel_mean = 0.0`, `pixel_std = 0.0`) on both TPU v7x and TPU v5p.
- **Resolution:** Dropping `--enable-torch-compile` restored valid high-variance images (`pixel_std = 59.6–74.4`) on TPU v7x across `1024×1024` (`0.419 s`, `139.3 img/min` at `c=2`), `1536×1536` (`0.720 s`), and `2048×2048` (`1.207 s`). On TPU v5p, `sglang-torchtpu` without compile produced valid outputs at `1024×1024` (`122.4 img/min`) and `1536×1536` (`0.924 s`), while `2048×2048` still returned blank images; `vllm-omni 0.33.0rc1` produced valid `2048×2048` images on both TPU v7x (`1.485 s`) and TPU v5p (`1.826 s`).

### 3. Ulysses Head Divisibility on `Z-Image-Turbo`
- **Issue:** Launching `Z-Image-Turbo` on `vllm-omni 0.33.0rc1` with `--ulysses-degree 4` failed orchestrator startup because `Z-Image-Turbo` has 30 attention heads (`30 % 4 != 0`).
- **Resolution:** Switching to `--ulysses-degree 2` booted cleanly in `32.0 s` on TPU v7x and generated valid images at `1.902 s` (`31.0 img/min`) at `1024×1024` and `8.833 s` at `2048×2048`. In `sglang-torchtpu`, `--enable-torch-compile` works cleanly on `Z-Image-Turbo` (`pixel_std = 63.2–75.7`), lifting TPU v7x throughput from `44.6 img/min` eager to `115.9 img/min` compiled (`3.74×` faster than `vllm-omni U=2`).

### 4. `Qwen-Image (20B)` Prompt-Length Recompilation & Open Issues
- **Issue:** Variable prompt token lengths from the Qwen2.5-VL text encoder triggered ~550 s XLA recompiles under `c=1` on `vllm-omni 0.33.0rc1` (`559.6 s` median, `0.14 img/min`).
- **Resolution:** Fixing prompt token length eliminated recompiles after the one-time `584.4 s` cold compile, lifting `c=1` to **`4.921 s` median (`12.19 img/min`, `87×` higher than variable-length prompts)** and `c=4` to **`12.36 img/min`** (`pixel_std = 43.8–70.0`).
- **Open Issues:** On `vllm-omni 0.33.0rc1`, `1664×928` returned `HTTP 500` and 50-step cold compilation exceeded the 600 s timeout. On `sglang-torchtpu` (`main`), `Qwen-Image` returned `HTTP 503` on both TPU v7x and TPU v5p (with and without `--enable-torch-compile`).
