# FLUX.1-schnell on Google Cloud TPUs (TorchTPU)

- **Model:** [`black-forest-labs/FLUX.1-schnell`](https://huggingface.co/black-forest-labs/FLUX.1-schnell) (12B rectified-flow DiT, 4 distilled steps, `bf16`)
- **License:** Apache-2.0
- **Engines Tested:** `sglang-torchtpu` (`main`, `tpu_splash_attn`) and `vllm-torchtpu v0.33.0` + `vllm-omni 0.33.0rc1` (`OmniTpuPlatform`)

## Measured Results (Verified `valid_image=True`)

All outputs were decoded and verified for non-trivial pixel variance (`pixel_std` between `17.0` and `72.6`).

| Engine | TPU Hardware | Parallelism | Resolution | Steps | Concurrency | Warm Median Latency | Peak Throughput | Pixel Std |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`sglang-torchtpu` (`main`)** | **TPU v7x (4 chips / 8 cores)** | `U=8`, `tpu_splash_attn` | `1024×1024` | 4 | 1 | **0.830 s** | 68.5 img/min | 51.5 |
| **`sglang-torchtpu` (`main`)** | **TPU v7x (4 chips / 8 cores)** | `U=8`, `tpu_splash_attn` | `1024×1024` | 4 | 2 | 1.528 s | **77.5 img/min** | 51.5 |
| **`sglang-torchtpu` (`main`)** | **TPU v7x (4 chips / 8 cores)** | `U=8`, `tpu_splash_attn` | `1024×1024` | 4 | 4 | 3.093 s | 76.9 img/min | 46.6 |
| **`sglang-torchtpu` (`main`)** | **TPU v7x (4 chips / 8 cores)** | `U=8`, `tpu_splash_attn` | `2048×2048` | 4 | 1 | **2.008 s** | 29.0 img/min | 17.8 |
| **`sglang-torchtpu` (`main`)** | **TPU v7x (4 chips / 8 cores)** | `U=8`, `tpu_splash_attn` | `2048×2048` | 4 | 2 | 3.978 s | **29.5 img/min** | 17.8 |
| `sglang-torchtpu` (`main`) | TPU v5p (4 chips) | `U=4` | `1024×1024` | 4 | 1 | 1.346 s | 42.9 img/min | 51.7 |
| `sglang-torchtpu` (`main`) | TPU v5p (4 chips) | `U=4` | `1024×1024` | 4 | 4 | 3.724 s | 59.4 img/min | 44.0 |
| `sglang-torchtpu` (`main`) | TPU v5p (4 chips) | `U=4` | `2048×2048` | 4 | 1 | 2.877 s | 20.0 img/min | 26.9 |
| `sglang-torchtpu` (`main`) | TPU v5p (4 chips) | `U=4` | `2048×2048` | 4 | 2 | 5.669 s | 20.7 img/min | 26.9 |
| `vllm-torchtpu v0.33.0` + `vllm-omni` | TPU v7x (4 chips) | `U=4` | `1024×1024` | 4 | 1 | 1.510 s | 38.1 img/min | 51.5 |
| `vllm-torchtpu v0.33.0` + `vllm-omni` | TPU v7x (4 chips) | `U=4` | `1024×1024` | 4 | 2 | 2.766 s | 41.8 img/min | 51.5 |
| `vllm-torchtpu v0.33.0` + `vllm-omni` | TPU v7x (4 chips) | `U=4` | `2048×2048` | 4 | 1 | 9.445 s | 6.3 img/min | 17.0 |
| `vllm-torchtpu v0.33.0` + `vllm-omni` | TPU v5p (4 chips) | `U=4` | `1024×1024` | 4 | 1 | 2.679 s | 22.0 img/min | 62.7 |
| `vllm-torchtpu v0.33.0` + `vllm-omni` | TPU v5p (4 chips) | `U=4` | `1024×1024` | 4 | 2 | 5.065 s | 23.1 img/min | 58.5 |
| `vllm-torchtpu v0.33.0` + `vllm-omni` | TPU v5p (4 chips) | `U=4` | `2048×2048` | 4 | 1 | 21.887 s | 2.7 img/min | 54.4 |

### Key Takeaways
- **`sglang-torchtpu` (`tpu_splash_attn`) leads cleanly on `FLUX.1-schnell`:** On TPU v7x, SGLang achieves **1.82× lower latency** at `1024×1024` (`0.830 s` vs `1.510 s` on `vllm-omni`) and **4.70× lower latency** at `2048×2048` (`2.008 s` vs `9.445 s`).
- **Strong TPU v7x scaling over TPU v5p:** On SGLang, TPU v7x (`U=8`) delivers **1.62× lower `1024×1024` latency** (`0.830 s` vs `1.346 s`) and **1.43× lower `2048×2048` latency** (`2.008 s` vs `2.877 s`) than 4-chip TPU v5p (`U=4`).

## Reproduce

### Option A: SGLang on TorchTPU (Recommended for FLUX.1-schnell)
```bash
# On 4x TPU v7x (8 TensorCores)
sglang serve \
  --model-path /path/to/black-forest-labs/FLUX.1-schnell \
  --host 0.0.0.0 --port 8000 \
  --num-gpus 8 --ulysses-degree 8 \
  --attention-backend tpu_splash_attn

python3 ../scripts/benchmark_image_diffusion.py \
  --engine sglang --model black-forest-labs/FLUX.1-schnell \
  --width 1024 --height 1024 --steps 4 --concurrency 2 --num-requests 8
```

### Option B: vLLM-Omni on TorchTPU
```bash
vllm serve /path/to/black-forest-labs/FLUX.1-schnell \
  --omni --served-model-name black-forest-labs/FLUX.1-schnell \
  --host 0.0.0.0 --port 8000 --ulysses-degree 4

python3 ../scripts/benchmark_image_diffusion.py \
  --engine vllm-omni --model black-forest-labs/FLUX.1-schnell \
  --width 1024 --height 1024 --steps 4 --concurrency 1 --num-requests 6
```
