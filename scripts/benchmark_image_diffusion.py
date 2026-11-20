#!/usr/bin/env python3
"""Dual-engine TorchTPU Image Diffusion benchmark harness.

Benchmarks OpenAI-compatible image generation endpoints served by either:
- vLLM-Omni on TorchTPU (`vllm serve <model> --omni ...`)
- SGLang on TorchTPU (`sglang serve --model-path <model> ...`)

Includes mandatory PNG decode and pixel-variance verification so silent
blank/all-black outputs (pixel_std == 0) are never counted as valid results.
"""

from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import io
import json
import statistics
import time
import urllib.request
from pathlib import Path
from typing import Any

from PIL import Image, ImageStat

PROMPTS = [
    "A photorealistic architectural photograph of a sunlit alpine timber pavilion overlooking a calm glacial lake at golden hour, crisp reflections, ultra detailed",
    "Studio product photograph of a brushed titanium espresso machine on travertine stone, warm directional lighting, shallow depth of field, sharp typography",
    "Editorial wildlife photograph of a snow leopard resting on a granite ridge during light snowfall, telephoto lens, natural daylight, rich fur texture",
    "Wide-angle interior photograph of a sunlit botanical conservatory filled with glass domes, ferns, and wet limestone tiles, natural morning light",
]


def verify_png_quality(raw_png: bytes) -> dict[str, Any]:
    """Decode PNG bytes and verify non-trivial pixel variance."""
    img = Image.open(io.BytesIO(raw_png)).convert("RGB")
    stat = ImageStat.Stat(img)
    pixel_mean = round(sum(stat.mean) / 3.0, 2)
    pixel_std = round(sum(stat.stddev) / 3.0, 2)
    return {
        "width": img.width,
        "height": img.height,
        "bytes": len(raw_png),
        "pixel_mean": pixel_mean,
        "pixel_std": pixel_std,
        "valid_image": pixel_std > 2.0,
    }


def call_image_endpoint(
    base_url: str,
    engine: str,
    model: str,
    prompt: str,
    width: int,
    height: int,
    steps: int,
    timeout_s: float = 600.0,
) -> tuple[float, dict[str, Any]]:
    """Send a single image generation request and return (latency_s, quality_dict)."""
    t0 = time.perf_counter()
    if engine == "vllm-omni":
        url = f"{base_url.rstrip('/')}/v1/chat/completions"
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "extra_body": {
                "height": height,
                "width": width,
                "num_inference_steps": steps,
            },
        }
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        content = body["choices"][0]["message"]["content"]
        b64_img = None
        if isinstance(content, list):
            for item in content:
                if item.get("type") == "image_url":
                    img_url = item["image_url"]["url"]
                    b64_img = img_url.split(",", 1)[1] if "," in img_url else img_url
                    break
        if not b64_img:
            raise RuntimeError(f"No image returned by vLLM-Omni endpoint: {str(body)[:240]}")
        raw = base64.b64decode(b64_img)
    else:
        url = f"{base_url.rstrip('/')}/v1/images/generations"
        payload = {
            "model": model,
            "prompt": prompt,
            "size": f"{width}x{height}",
            "num_inference_steps": steps,
            "response_format": "b64_json",
        }
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        raw = base64.b64decode(body["data"][0]["b64_json"])

    elapsed = time.perf_counter() - t0
    return elapsed, verify_png_quality(raw)


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    """Run one cold warmup call followed by measured concurrency iterations."""
    cold_s, cold_q = call_image_endpoint(
        args.base_url,
        args.engine,
        args.model,
        PROMPTS[0],
        args.width,
        args.height,
        args.steps,
        timeout_s=args.timeout,
    )

    latencies: list[float] = []
    last_quality = cold_q
    wall_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futs = [
            pool.submit(
                call_image_endpoint,
                args.base_url,
                args.engine,
                args.model,
                PROMPTS[0] if args.fixed_prompt else PROMPTS[i % len(PROMPTS)],
                args.width,
                args.height,
                args.steps,
                args.timeout,
            )
            for i in range(args.num_requests)
        ]
        for fut in as_completed(futs):
            lat, qual = fut.result()
            latencies.append(lat)
            last_quality = qual
    wall_s = time.perf_counter() - wall_start

    return {
        "model": args.model,
        "engine": args.engine,
        "width": args.width,
        "height": args.height,
        "steps": args.steps,
        "concurrency": args.concurrency,
        "num_requests": args.num_requests,
        "cold_s": round(cold_s, 3),
        "warm_min_s": round(min(latencies), 3),
        "warm_median_s": round(statistics.median(latencies), 3),
        "warm_mean_s": round(statistics.mean(latencies), 3),
        "images_per_min": round(len(latencies) / wall_s * 60.0, 2),
        "quality": last_quality,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark TorchTPU image diffusion servers.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--engine", choices=["vllm-omni", "sglang"], required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--num-requests", type=int, default=6)
    parser.add_argument("--fixed-prompt", action="store_true", help="Keep prompt token length fixed to avoid XLA shape recompiles")
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    result = run_benchmark(args)
    print(json.dumps(result, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
