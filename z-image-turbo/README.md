# Z-Image-Turbo on Google Cloud TPUs (TorchTPU)

- **Model:** [`Tongyi-MAI/Z-Image-Turbo`](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) (6B single-stream DiT, 30 attention heads, 8 distilled steps, `bf16`)
- **License:** Apache-2.0
- **Engines Tested:** `sglang-torchtpu` (`main`, `tpu_splash_attn` + `--enable-torch-compile`) and `vllm-torchtpu v0.33.0` + `vllm-omni 0.33.0rc1`

## Measured Results (Verified `valid_image=True`)

All generated images passed PNG decode and pixel-variance checks (`pixel_std` between `58.0` and `88.8`).

| Engine | TPU Hardware | Parallelism / Flags | Resolution | Steps | Concurrency | Warm Min / Median | Peak Throughput | Pixel Std |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`sglang-torchtpu` (`main`)** | **TPU v7x (4 chips / 8 cores)** | `U=8`, `--enable-torch-compile` | `1024×1024` | 8 | 2 | 0.507 s / **1.014 s** | 115.7 img/min | 72.3 |
| **`sglang-torchtpu` (`main`)** | **TPU v7x (4 chips / 8 cores)** | `U=8`, `--enable-torch-compile` | `1024×1024` | 8 | 4 | 0.509 s / 2.046 s | **115.9 img/min** | 75.7 |
| **`sglang-torchtpu` (`main`)** | **TPU v7x (4 chips / 8 cores)** | `U=8`, `--enable-torch-compile` | `2048×2048` | 8 | 1 | **2.390 s** (steady) | — | 63.2 |
| `sglang-torchtpu` (`main`) | TPU v7x (4 chips / 8 cores) | `U=8`, eager (no compile) | `1024×1024` | 8 | 2 | 1.454 s / 2.663 s | 43.8 img/min | 88.4 |
| `sglang-torchtpu` (`main`) | TPU v7x (4 chips / 8 cores) | `U=8`, eager (no compile) | `1024×1024` | 8 | 4 | 1.334 s / 5.351 s | 44.6 img/min | 88.8 |
| `sglang-torchtpu` (`main`) | TPU v5p (4 chips) | `U=4`, `--enable-torch-compile` | `1024×1024` | 8 | 2 | 0.870 s / 1.537 s | 68.0 img/min | 72.4 |
| `sglang-torchtpu` (`main`) | TPU v5p (4 chips) | `U=4`, `--enable-torch-compile` | `1024×1024` | 8 | 4 | 0.870 s / 3.486 s | 68.2 img/min | 75.7 |
| `sglang-torchtpu` (`main`) | TPU v5p (4 chips) | `U=4`, `--enable-torch-compile` | `2048×2048` | 8 | 1 | 28.65 s (med, incl. recompile) | 2.1 img/min | 58.0 |
| `vllm-torchtpu v0.33.0` + `vllm-omni` | TPU v7x (4 chips) | `U=2` *(30 heads % 2 == 0)* | `1024×1024` | 8 | 1 | 1.893 s / **1.902 s** | **31.0 img/min** | 85.4 |
| `vllm-torchtpu v0.33.0` + `vllm-omni` | TPU v7x (4 chips) | `U=2` *(30 heads % 2 == 0)* | `1024×1024` | 8 | 2 | 1.903 s / 4.476 s | 24.7 img/min | 84.8 |
| `vllm-torchtpu v0.33.0` + `vllm-omni` | TPU v7x (4 chips) | `U=2` *(30 heads % 2 == 0)* | `1024×1024` | 8 | 4 | 1.888 s / 9.275 s | 25.3 img/min | 86.9 |
| `vllm-torchtpu v0.33.0` + `vllm-omni` | TPU v7x (4 chips) | `U=2` *(30 heads % 2 == 0)* | `2048×2048` | 8 | 1 | **8.833 s** | **6.7 img/min** | Valid |

### Key Takeaways & Engine Compatibility Notes
- **Dual-engine comparison on TPU v7x:** `sglang-torchtpu` (`U=8`, `--enable-torch-compile`) reaches **`115.9 img/min`** (`0.507 s` min), delivering **`3.74×` higher throughput** than `vllm-omni 0.33.0rc1` (`U=2`, `31.0 img/min` / `1.902 s`), and **`2.60×` higher throughput** than eager SGLang (`44.6 img/min`), with verified high pixel variance (`pixel_std = 75.7`).
- **TPU v7x outperforms TPU v5p by 1.70×:** `115.9 img/min` on 4-chip TPU v7x (`U=8`) vs `68.2 img/min` on 4-chip TPU v5p (`U=4`).
- **`vllm-omni` Ulysses head-divisibility fix verified live:** `Z-Image-Turbo` uses **30 attention heads** (`query_head_cnt=30`), which fails strict validation at `--ulysses-degree 4`. Switching to `--ulysses-degree 2` booted cleanly in `32.0 s` and served verified valid `1024×1024` images (`pixel_std = 84.8–86.9`) at `1.902 s` (`31.0 img/min`).
- **Prompt-length bucketing:** Under `c=1` with varying prompt lengths, each new prompt token length triggers a fresh XLA shape compilation (`24.8 s` median at `2048×2048` vs `2.39 s` steady-state on cached shapes). Pad or bucket prompt lengths in production.

## Reproduce

```bash
# Recommended: SGLang on TorchTPU with torch.compile (4x TPU v7x / 8 cores)
sglang serve \
  --model-path /path/to/Tongyi-MAI/Z-Image-Turbo \
  --host 0.0.0.0 --port 8000 \
  --num-gpus 8 --ulysses-degree 8 \
  --attention-backend tpu_splash_attn \
  --enable-torch-compile

python3 ../scripts/benchmark_image_diffusion.py \
  --engine sglang --model Tongyi-MAI/Z-Image-Turbo \
  --width 1024 --height 1024 --steps 8 --concurrency 4 --num-requests 8 --fixed-prompt
```
