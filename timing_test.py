#!/usr/bin/env python3
"""VLM timing test: same N images, reports seconds/image and tokens/second."""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import urllib.request

from immich_api import ImmichClient, load_dotenv
from vlm_backend import OllamaVLM, DEFAULT_CAPTION_PROMPT

load_dotenv()

base_url = os.environ.get("IMMICH_URL", "").strip()
api_key  = os.environ.get("IMMICH_API_KEY", "").strip()

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--count", type=int, default=10)
    p.add_argument("--seed",  type=int, default=7)
    p.add_argument("--vlm-model", default="qwen2.5vl:7b")
    p.add_argument("--vlm-url",   default="http://localhost:11434")
    return p.parse_args()


def timed_caption(vlm: OllamaVLM, image_bytes: bytes, people: list[str]) -> tuple[str, float, float]:
    """Returns (caption, elapsed_sec, gen_tokens_per_sec). Reads timing from Ollama eval_duration."""
    import base64
    b64 = base64.b64encode(image_bytes).decode()

    if people:
        names_str = ", ".join(people)
        people_clause = (
            f"The people in this photo are: {names_str}. "
            f"Include their name(s) naturally in your sentence — do NOT output just a name alone."
        )
    else:
        people_clause = ""

    prompt = DEFAULT_CAPTION_PROMPT.format(people_clause=people_clause).strip()
    prompt = "\n".join(line for line in prompt.splitlines() if line.strip())

    body = {
        "model": vlm.model,
        "messages": [{"role": "user", "content": prompt, "images": [b64]}],
        "stream": False,
        "options": {"temperature": 0.3},
    }
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        f"{vlm.base_url}/api/chat",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=300) as resp:
        result = json.loads(resp.read())
    elapsed = time.time() - t0

    caption = result.get("message", {}).get("content", "").strip()

    # Ollama returns eval_duration in nanoseconds
    eval_ns    = result.get("eval_duration", 0)
    eval_count = result.get("eval_count", 0)
    gen_tps = (eval_count / (eval_ns / 1e9)) if eval_ns and eval_count else 0.0

    return caption, elapsed, gen_tps


def main():
    args = parse_args()
    client = ImmichClient(base_url, api_key)
    vlm = OllamaVLM(args.vlm_model, base_url=args.vlm_url, timeout=300)

    if not vlm.is_available():
        print("ERROR: Ollama not reachable", file=sys.stderr)
        sys.exit(1)

    print(f"Fetching asset pool...", file=sys.stderr)
    assets = list(client.find_new_assets(since=None))
    rng = random.Random(args.seed)
    sample = rng.sample(assets, min(args.count, len(assets)))

    print(f"\nModel : {args.vlm_model}")
    print(f"Images: {len(sample)} (seed={args.seed})")
    print("-" * 60)

    times, tps_list = [], []
    for i, asset in enumerate(sample, 1):
        thumb = client.get_thumbnail(asset.id)
        caption, elapsed, gen_tps = timed_caption(vlm, thumb, asset.people_names)
        times.append(elapsed)
        if gen_tps:
            tps_list.append(gen_tps)
        print(f"[{i:02d}] {asset.file_name[:45]:<45}  {elapsed:5.1f}s  {gen_tps:5.1f} t/s")
        print(f"      → {caption[:80]}")

    print("-" * 60)
    avg = sum(times) / len(times)
    avg_tps = sum(tps_list) / len(tps_list) if tps_list else 0
    print(f"Avg: {avg:.1f}s/image   gen: {avg_tps:.1f} t/s   total: {sum(times):.0f}s")

if __name__ == "__main__":
    main()
