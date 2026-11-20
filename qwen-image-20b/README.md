# Qwen-Image (20B) on Google Cloud TPUs (TorchTPU)

- **Model:** [`Qwen/Qwen-Image`](https://huggingface.co/Qwen/Qwen-Image) (20B MMDiT with Qwen2.5-VL text encoder, 20 steps, `bf16`)
- **License:** Apache-2.0
- **Engines Tested:** `vllm-torchtpu v0.33.0` + `vllm-omni 0.33.0rc1` (`OmniTpuPlatform`) and `sglang-torchtpu` (`main`)

## Measured Results (Verified `valid_image=True`)

| Engine | TPU Hardware | Parallelism & Prompt Mode | Resolution | Steps | Concurrency | Warm Median / p95 Latency | Peak Throughput | Pixel Std |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`vllm-torchtpu v0.33.0` + `vllm-omni`** | **TPU v7x (4 chips)** | **`CFG=2×U=2 + Cache-DiT`**, fixed prompt | `1024×1024` | 20 | 1 | **3.345 s** / 3.392 s | **17.75 img/min** | 41.2–52.4 |
| **`vllm-torchtpu v0.33.0` + `vllm-omni`** | **TPU v7x (4 chips)** | **`CFG=2×U=2 + Cache-DiT`**, fixed prompt | `1024×1024` | 20 | 2 | 5.948 s / 6.324 s | **19.48 img/min** | 40.0–51.9 |
| **`vllm-torchtpu v0.33.0` + `vllm-omni`** | **TPU v7x (4 chips)** | **`CFG=2×U=2 + Cache-DiT`**, fixed prompt | `1024×1024` | 20 | 4 | 8.101 s / 13.421 s | 17.84 img/min | 49.5–60.1 |
| `vllm-torchtpu v0.33.0` + `vllm-omni` | TPU v7x (4 chips) | `U=4 + Cache-DiT`, fixed prompt | `1024×1024` | 20 | 1 | 3.953 s / 6.091 s | 13.29 img/min | 40.1–55.2 |
| `vllm-torchtpu v0.33.0` + `vllm-omni` | TPU v7x (4 chips) | `U=4 + Cache-DiT`, fixed prompt | `1024×1024` | 20 | 2 | 7.379 s / 7.944 s | 15.93 img/min | 44.9–55.8 |
| `vllm-torchtpu v0.33.0` + `vllm-omni` | TPU v7x (4 chips) | `U=4` (exact), fixed prompt | `1024×1024` | 20 | 1 | 4.921 s / — | 12.19 img/min | 43.8 |
| `vllm-torchtpu v0.33.0` + `vllm-omni` | TPU v7x (4 chips) | `U=4` (exact), fixed prompt | `1024×1024` | 20 | 4 | 19.165 s / — | 12.36 img/min | 70.0 |
| `vllm-torchtpu v0.33.0` + `vllm-omni` | TPU v7x (4 chips) | `U=4` (exact), variable prompt | `1024×1024` | 20 | 1 | 559.6 s *(recompiles)* | 0.14 img/min | 36.8 |
| `sglang-torchtpu` (`main`) | TPU v7x / v5p | `U=8` / `U=4` | `1024×1024` | 20–28 | 1 | *Unsupported (`HTTP 503`)* | — | — |

### Key Takeaways & Caveats
- **`CFG=2 × U=2 + Cache-DiT` reaches `3.345 s` (`c=1`) and `19.48 img/min` (`c=2`):** Combining `--cfg-parallel-size 2 --ulysses-degree 2 --cache-backend cache_dit` reduces warm `c=1` latency by **`1.47×`** (`4.921 s -> 3.345 s`) and increases peak throughput by **`1.58×`** (`12.36 -> 19.48 img/min`) vs. exact `U=4`, with `20.5 s` cold compile and `42.0 s` warm server boot.
- **Prompt-length bucketing lifts `c=1` throughput by `87×–139×`:** With unbucketed variable prompts, each new Qwen2.5-VL prompt token length triggered a ~550 s XLA recompile (`559.6 s` median / `0.14 img/min` at `c=1`). Using fixed-length prompts removes recompiles completely.
- **Current engine limitations documented:**
  - `vllm-omni 0.33.0rc1` at `1664×928` returned `HTTP 500`, and 50-step cold compilation exceeded the 600 s client timeout.
  - `sglang-torchtpu` (`main`) returned `HTTP 503` on both TPU v7x and TPU v5p (with and without `--enable-torch-compile`).

## Reproduce

```bash
vllm serve /path/to/Qwen/Qwen-Image \
  --omni --served-model-name Qwen/Qwen-Image \
  --host 0.0.0.0 --port 8000 \
  --cfg-parallel-size 2 --ulysses-degree 2 --cache-backend cache_dit

python3 ../scripts/benchmark_image_diffusion.py \
  --engine vllm-omni --model Qwen/Qwen-Image \
  --width 1024 --height 1024 --steps 20 --concurrency 2 --num-requests 8 --fixed-prompt
```
