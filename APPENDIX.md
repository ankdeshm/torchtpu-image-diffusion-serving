# Appendix — Image Diffusion Optimization Progressions, Quality Audits & Extended Ablations

This appendix captures detailed dual-engine measurements, cold-vs-warm compilation behavior, pixel-variance quality audits, and failure/negative findings for image diffusion models on TPU v7x and TPU v5p.

---

## 1. Critical Quality Audit: Silent Blank Outputs Under `--enable-torch-compile` on `FLUX.2-klein-4B`

Every benchmark request decoded the returned PNG and computed channel-averaged RGB standard deviation (`pixel_std`), marking an output valid only when `pixel_std > 2.0`.

During `sglang-torchtpu` (`main`) testing of `black-forest-labs/FLUX.2-klein-4B`, passing `--enable-torch-compile` caused the server to return `HTTP 200 OK` PNG responses at very high apparent speed (`182.4–235.1 img/min`), **but every generated image was completely black (`pixel_mean = 0.0`, `pixel_std = 0.0`, `valid_image = False`)**:

| Hardware | Engine & Compile Flag | Resolution | Concurrency | Reported Throughput | `pixel_mean` | `pixel_std` | `valid_image` |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| TPU v7x (`U=8`) | `sglang-torchtpu` + `--enable-torch-compile` | `1024×1024` | 4 | 235.1 img/min *(invalid)* | 0.00 | 0.00 | **False** |
| TPU v7x (`U=8`) | `sglang-torchtpu` + `--enable-torch-compile` | `1536×1536` | 1 | 107.0 img/min *(invalid)* | 0.00 | 0.00 | **False** |
| TPU v7x (`U=8`) | `sglang-torchtpu` + `--enable-torch-compile` | `2048×2048` | 1 | 63.3 img/min *(invalid)* | 0.00 | 0.00 | **False** |
| TPU v5p (`U=4`) | `sglang-torchtpu` + `--enable-torch-compile` | `1024×1024` | 4 | 182.4 img/min *(invalid)* | 0.00 | 0.00 | **False** |
| **TPU v7x (`U=8`)** | **`sglang-torchtpu` (no compile)** | **`1024×1024`** | **2** | **139.3 img/min** | **105.65** | **70.43** | **True** |
| **TPU v5p (`U=4`)** | **`sglang-torchtpu` (no compile)** | **`1024×1024`** | **2** | **122.4 img/min** | **105.60** | **68.51** | **True** |
| **TPU v5p (`U=4`)** | **`sglang-torchtpu` (no compile)** | **`1536×1536`** | **1** | **61.5 img/min** | **113.10** | **73.59** | **True** |
| TPU v5p (`U=4`) | `sglang-torchtpu` (no compile) | `2048×2048` | 1 | 35.4 img/min *(invalid)* | 0.00 | 0.00 | **False** |
| **TPU v7x (`U=4`)** | **`vllm-torchtpu v0.33.0` + `vllm-omni`** | **`1024×1024`** | **1** | **120.7 img/min** | **118.40** | **95.87** | **True** |
| **TPU v7x (`U=4`)** | **`vllm-torchtpu v0.33.0` + `vllm-omni`** | **`2048×2048`** | **1** | **36.9 img/min** | **90.10** | **27.51** | **True** |
| **TPU v5p (`U=4`)** | **`vllm-torchtpu v0.33.0` + `vllm-omni`** | **`2048×2048`** | **1** | **30.4 img/min** | **107.20** | **65.97** | **True** |

**Takeaway:** Never benchmark diffusion servers on HTTP status codes alone. On `FLUX.2-klein-4B`, `sglang-torchtpu` must be run **without** `--enable-torch-compile` at `1024×1024` / `1536×1536`, and `vllm-omni 0.33.0rc1` should be used for `2048×2048` where `sglang` produced blank output on TPU v5p.

---

## 2. Concurrency & Batching Scaling Across Engines

### `FLUX.2-klein-4B` (`1024×1024`, 4 steps, `valid_image=True`)
- **Single-request latency (`c=1`):** Both engines achieve sub-half-second generation (`0.419 s` on TPU v7x `sglang` no-compile, `0.450 s` on TPU v7x `vllm-omni`, `0.431 s` on TPU v5p `vllm-omni`, `0.550 s` on TPU v5p `sglang` no-compile).
- **Multi-request concurrency (`c=2` and `c=4`):**
  - On `sglang-torchtpu` (no compile), pipelined HTTP/VAE overlap lifts throughput from `118.1` (`c=1`) to **`139.3 img/min` (`c=2`)** and **`137.9 img/min` (`c=4`)** on TPU v7x (`U=8`), and from `103.6` (`c=1`) to **`122.4 img/min` (`c=2`)** on TPU v5p (`U=4`).
  - On `vllm-omni 0.33.0rc1` (`U=4`), concurrent requests (`c=2/4`) experienced queueing and shape-dispatch overhead (`29.1 img/min` at `c=2` vs `120.7 img/min` at `c=1` on TPU v7x; `25.2–29.8 img/min` vs `127.0 img/min` on TPU v5p). For `vllm-omni` deployments, scale throughput horizontally across `c=1` worker replicas rather than queueing multiple concurrent requests on one replica.

### `FLUX.1-schnell` (`1024×1024` & `2048×2048`, 4 steps, `valid_image=True`)
- On `sglang-torchtpu` (`main`), TPU v7x (`U=8`, `tpu_splash_attn`) achieves `0.830 s` (`68.5 img/min`) at `c=1`, scaling smoothly to **`77.5 img/min`** at `c=2` and `76.9 img/min` at `c=4`. At `2048×2048`, latency is **`2.008 s`** (`29.0 img/min` at `c=1`, `29.5 img/min` at `c=2`).
- On `vllm-omni 0.33.0rc1` (`U=4`), TPU v7x achieves `1.510 s` (`38.1 img/min`) at `c=1` and `41.8 img/min` at `c=2`, and `9.445 s` (`6.25 img/min`) at `2048×2048`. At `c=4` on TPU v7x, an intermediate recompile raised median latency to `64.4 s` despite `1.518 s` warm minimum.

### Built-In `vllm-omni 0.33.0rc1` `--data-parallel-size` vs. `sglang --dp-size 8` (4-Chip TPU v7x, `1024×1024`, Negative Ablation)

Enabling built-in single-server `--data-parallel-size > 1` in `vllm-omni 0.33.0rc1` on 4-chip TPU v7x produced verified outputs (`valid_image=True`, `pixel_std = 52.4–95.8`) but regressed severely due to single-coordinator worker dispatch contention (`3.0×–9.8×` lower host throughput than `sglang-torchtpu --dp-size 8`):

| Model (Steps) | `vllm-omni 0.33` Built-In `DP=4 × U=1` | `vllm-omni 0.33` Built-In `DP=2 × U=2` | `sglang-torchtpu` `DP=8 × U=1` | Comparison (`sglang DP=8` vs. `vllm-omni DP=4`) |
| :--- | :---: | :---: | :---: | :--- |
| **`FLUX.2-klein-4B`** (4) | `50.9 img/min` (`c=16`, med `17.36 s`) | `96.3 img/min` (`c=2`, med `1.07 s`) | **373.8 img/min** (`c=16`, med `2.35 s`) | **`7.34×` higher** host throughput on `sglang DP=8` (`vllm-omni` single-replica `U=4` reached `120.7 img/min`) |
| **`Z-Image-Turbo`** (8) | `27.5 img/min` (`c=16`, med `34.04 s`) | `32.7 img/min` (`c=4`, med `7.18 s`) | **270.4 img/min** (`c=32`, med `6.22 s`) | **`9.83×` higher** host throughput on `sglang DP=8` |
| **`FLUX.1-schnell`** (4) | `64.3 img/min` (`c=16`, med `12.84 s`) | — | **192.3 img/min** (`c=32`, med `9.41 s`) | **`2.99×` higher** host throughput on `sglang DP=8` |

**Takeaway:** For 4-chip host data-parallel serving on TPU v7x, use `sglang-torchtpu --dp-size 8` (which scales linearly across 2 hosts to `730.43 img/min` on `FLUX.2-klein-4B` and `589.01 img/min` on `Z-Image-Turbo` at `DP=16`) or deploy independent single-replica `vllm-omni` server processes behind an external HTTP router rather than `vllm-omni`'s built-in `--data-parallel-size`.

#### External-Router `vllm-omni` Multi-Host Scaling (`16` TPU v7x Chips = `4` Hosts, `DP=4`)
- **`Qwen-Image (20B)` (`CFG=2×U=2 + Cache-DiT`):** Scales cleanly across all 4 hosts at `c=4` (**`63.32 img/min`**, `3.71 s` med, `3.94 s` p95), `c=8` (`57.27 img/min`, `7.20 s` med), and `c=16` (`61.88 img/min`, `13.37 s` med) with zero throughput collapse (`pixel_std = 31.2–64.2`).
- **`FLUX.2-klein-4B` (`U=4`):** Reaches **`310.07 img/min`** (`0.709 s` med, `0.793 s` p95, `6.09×` higher than built-in `--data-parallel-size 4`) when concurrency is kept at 1 in-flight request per replica (`c=4` across 4 hosts), but drops to `97.85 img/min` (`c=8`) and `131.98 img/min` (`c=16`) when requests queue inside a single `vllm-omni` worker. Keep per-replica concurrency at `c=1` in the external router for sub-second `vllm-omni` image models.

---

## 3. Compilation Overhead & Prompt-Length Bucketing

| Model | Engine | Hardware | Resolution | Cold First Request | Warm Steady Latency | Notes |
| :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| `FLUX.2-klein-4B` | `vllm-omni 0.33.0rc1` | TPU v7x (`U=4`) | `1024×1024` | 0.487 s *(pre-warmed)* | 0.450 s | Warmed during engine boot |
| `FLUX.2-klein-4B` | `vllm-omni 0.33.0rc1` | TPU v7x (`U=4`) | `1536×1536` | 287.7 s | 0.859 s | One-time shape compile |
| `FLUX.2-klein-4B` | `vllm-omni 0.33.0rc1` | TPU v7x (`U=4`) | `2048×2048` | 436.7 s | 1.485 s | One-time shape compile |
| `FLUX.2-klein-4B` | `vllm-omni 0.33.0rc1` | TPU v5p (`U=4`) | `1024×1024` | 15.8 s | 0.431 s | Fast shape compile |
| `FLUX.2-klein-4B` | `vllm-omni 0.33.0rc1` | TPU v5p (`U=4`) | `2048×2048` | 47.6 s | 1.826 s | Fast shape compile |
| `FLUX.1-schnell` | `sglang-torchtpu` | TPU v7x (`U=8`) | `1024×1024` | 113.9 s | 0.830 s | `tpu_splash_attn` compile |
| `FLUX.1-schnell` | `sglang-torchtpu` | TPU v5p (`U=4`) | `2048×2048` | 204.6 s | 2.877 s | One-time 4MP compile |
| `Z-Image-Turbo` | `sglang-torchtpu` (compile) | TPU v7x (`U=8`) | `1024×1024` | 65.5 s | 0.507 s (min) / 1.014 s (`c=2`) | Pad prompt lengths to avoid recompiles |
| `Z-Image-Turbo` | `sglang-torchtpu` (compile) | TPU v7x (`U=8`) | `2048×2048` | 127.2 s | 2.390 s (min) | Variable prompts triggered 24.8 s median |
| `Qwen-Image (20B)` | `vllm-omni 0.33.0rc1` | TPU v7x (`U=4`) | `1024×1024` | ~560 s | 4.638–4.805 s | Variable Qwen2.5-VL prompt lengths trigger ~550 s recompiles |

---

## 4. What Did Not Work / Upstream Compatibility Boundaries

1. **`Z-Image-Turbo` on `vllm-omni 0.33.0rc1` with `--ulysses-degree 4`:**
   - Failed at orchestrator startup with:
     `RuntimeError: Orchestrator initialization failed: Dummy run failed: Ulysses-SP strict mode requires head_cnt divisible by ulysses_degree. query_head_cnt=30, ulysses_degree=4. Try ulysses_degree in [1, 2, 3, 5, 6, 10, 15, 30], or set ulysses_mode='advanced_uaa'.`
   - **Resolution:** Pass `--ulysses-degree 2` (or `1, 3, 5, 6`) or `--ulysses-mode advanced_uaa`, or serve on `sglang-torchtpu` (`main`) which runs `U=8` cleanly at `115.9 img/min`.
2. **`Qwen-Image (20B)` on `sglang-torchtpu` (`main`):**
   - Returned `HTTP 503 Service Unavailable` on both TPU v7x (`U=8`) and TPU v5p (`U=4`), both with and without `--enable-torch-compile`. Use `vllm-omni 0.33.0rc1` for `Qwen-Image (20B)`.
3. **`Qwen-Image (20B)` on `vllm-omni 0.33.0rc1` at `1664×928` and 50-step cold runs:**
   - `1664×928` returned `HTTP 500 Internal Server Error` on both TPU v7x and TPU v5p; 50-step cold generation exceeded the 600 s client timeout. `1024×1024` (20 steps) ran cleanly and returned verified valid outputs (`pixel_std = 37.0–37.9`).
4. **`FLUX.2-klein-4B` on older `sglang-torchtpu` snapshot (`v0.2.0`):**
   - Failed at startup prior to upstream `Flux2KleinPipeline` updates merged into `sglang-torchtpu` `main`. Upgrading to `main` (without `--enable-torch-compile`) resolved startup cleanly (`38.3 s` boot on TPU v7x).

---

## 5. External GPU Reference Details

- **Source:** Published single-GPU NVIDIA H200 `FLUX.2-klein-4B` (`1024×1024`, 4 steps, `bf16`) latency is **`~0.57 s`** (`~1.77 img/s`, [inferencebench.io](https://inferencebench.io)).
- **Setup Differences:** The external H200 figure reflects a single-GPU deployment, whereas the TPU v7x (`0.419 s` on SGLang / `0.450 s` on vLLM-Omni) and TPU v5p (`0.431 s` on vLLM-Omni) numbers use 4-chip Ulysses Sequence Parallelism (`U=4` or `U=8`) on warm compiled graphs.
- For `FLUX.1-schnell`, `Z-Image-Turbo`, and `Qwen-Image (20B)`, external H200+ benchmarks matching the exact step count and resolution were omitted where no verified H200+ reference existed.
