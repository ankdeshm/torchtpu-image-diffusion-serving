# FLUX.2-klein-4B on Google Cloud TPUs (TorchTPU)

- **Model:** [`black-forest-labs/FLUX.2-klein-4B`](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) (4B rectified-flow DiT with Qwen3-4B text encoder, 4 distilled steps, `bf16`)
- **License:** Apache-2.0
- **Engines Tested:** `sglang-torchtpu` (`main`, eager / no `--enable-torch-compile`) and `vllm-torchtpu v0.33.0` + `vllm-omni 0.33.0rc1` (`OmniTpuPlatform`)

## Measured Results (Verified `valid_image=True`)

All entries below passed PNG decoding and RGB pixel-variance verification (`pixel_std > 2.0`).

| Engine | TPU Hardware | Parallelism | Resolution | Steps | Concurrency | Warm Median Latency | Peak Throughput | Pixel Std |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`sglang-torchtpu` (`main`, no compile)** | **TPU v7x (4 chips / 8 cores)** | `U=8`, `tpu_splash_attn` | `1024×1024` | 4 | 1 | **0.419 s** | 118.1 img/min | 70.4 |
| **`sglang-torchtpu` (`main`, no compile)** | **TPU v7x (4 chips / 8 cores)** | `U=8`, `tpu_splash_attn` | `1024×1024` | 4 | 2 | 0.832 s | **139.3 img/min** | 70.4 |
| **`sglang-torchtpu` (`main`, no compile)** | **TPU v7x (4 chips / 8 cores)** | `U=8`, `tpu_splash_attn` | `1024×1024` | 4 | 4 | 1.636 s | 137.9 img/min | 59.6 |
| **`sglang-torchtpu` (`main`, no compile)** | **TPU v7x (4 chips / 8 cores)** | `U=8`, `tpu_splash_attn` | `1536×1536` | 4 | 1 | **0.720 s** | **79.3 img/min** | 74.4 |
| **`sglang-torchtpu` (`main`, no compile)** | **TPU v7x (4 chips / 8 cores)** | `U=8`, `tpu_splash_attn` | `2048×2048` | 4 | 1 | **1.207 s** | **43.2 img/min** | 69.0 |
| **`vllm-torchtpu v0.33.0` + `vllm-omni`** | **TPU v7x (4 chips)** | `U=4` | `1024×1024` | 4 | 1 | **0.450 s** | 120.7 img/min | 95.9 |
| `vllm-torchtpu v0.33.0` + `vllm-omni` | TPU v7x (4 chips) | `U=4` | `1536×1536` | 4 | 1 | 0.859 s | 64.5 img/min | 28.7 |
| `vllm-torchtpu v0.33.0` + `vllm-omni` | TPU v7x (4 chips) | `U=4` | `2048×2048` | 4 | 1 | 1.485 s | 36.9 img/min | 27.5 |
| **`vllm-torchtpu v0.33.0` + `vllm-omni`** | **TPU v5p (4 chips)** | `U=4` | `1024×1024` | 4 | 1 | **0.431 s** | **127.0 img/min** | 71.1 |
| `vllm-torchtpu v0.33.0` + `vllm-omni` | TPU v5p (4 chips) | `U=4` | `1536×1536` | 4 | 1 | 0.959 s | 57.4 img/min | 68.7 |
| `vllm-torchtpu v0.33.0` + `vllm-omni` | TPU v5p (4 chips) | `U=4` | `2048×2048` | 4 | 1 | 1.826 s | 30.4 img/min | 66.0 |
| `sglang-torchtpu` (`main`, no compile) | TPU v5p (4 chips) | `U=4` | `1024×1024` | 4 | 1 | 0.550 s | 103.6 img/min | 71.0 |
| `sglang-torchtpu` (`main`, no compile) | TPU v5p (4 chips) | `U=4` | `1024×1024` | 4 | 2 | 0.931 s | 122.4 img/min | 68.5 |
| `sglang-torchtpu` (`main`, no compile) | TPU v5p (4 chips) | `U=4` | `1536×1536` | 4 | 1 | 0.924 s | 61.5 img/min | 73.6 |

### Reference Comparison
- Published single-GPU NVIDIA H200 latency for `FLUX.2-klein-4B` (`1024×1024`, 4 steps, `bf16`) is **`~0.57 s`** (`~1.77 img/s`, [inferencebench.io](https://inferencebench.io)).
- On 4-chip Ulysses Sequence Parallelism, TPU v7x achieves **`0.419 s`** on `sglang-torchtpu` (`1.36×` lower latency vs. the published single-GPU H200 reference) and **`0.450 s`** on `vllm-omni` (`1.27×` lower latency); TPU v5p achieves **`0.431 s`** on `vllm-omni` (`1.32×` lower latency).

### Important Quality Note (`--enable-torch-compile`)
- Do **not** pass `--enable-torch-compile` to `sglang-torchtpu` (`main`) when serving `FLUX.2-klein-4B`: compiled runs returned HTTP `200 OK` blank images (`pixel_std = 0.0`, `valid_image = False`) on both TPU v7x and TPU v5p. Removing `--enable-torch-compile` restored valid high-variance images (`pixel_std = 59.6–73.6`). See [`../APPENDIX.md`](../APPENDIX.md) for the full audit.
- For `2048×2048` 4-megapixel generation, use **`vllm-omni 0.33.0rc1`** (`1.485 s` on TPU v7x, `1.826 s` on TPU v5p, both `valid_image=True`).

## Reproduce

### Option A: SGLang on TorchTPU (Highest `1024×1024` Throughput — Do Not Pass `--enable-torch-compile`)
```bash
sglang serve \
  --model-path /path/to/black-forest-labs/FLUX.2-klein-4B \
  --host 0.0.0.0 --port 8000 \
  --num-gpus 8 --ulysses-degree 8 \
  --attention-backend tpu_splash_attn

python3 ../scripts/benchmark_image_diffusion.py \
  --engine sglang --model black-forest-labs/FLUX.2-klein-4B \
  --width 1024 --height 1024 --steps 4 --concurrency 2 --num-requests 8
```

### Option B: vLLM-Omni on TorchTPU (Best for `1536×1536` and `2048×2048`)
```bash
vllm serve /path/to/black-forest-labs/FLUX.2-klein-4B \
  --omni --served-model-name black-forest-labs/FLUX.2-klein-4B \
  --host 0.0.0.0 --port 8000 --ulysses-degree 4

python3 ../scripts/benchmark_image_diffusion.py \
  --engine vllm-omni --model black-forest-labs/FLUX.2-klein-4B \
  --width 2048 --height 2048 --steps 4 --concurrency 1 --num-requests 6
```
