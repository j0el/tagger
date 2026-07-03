#!/usr/bin/env python3
"""
Demo: compare existing Immich captions/tags vs what v2 would produce for N random assets.

Usage:
    uv run python demo_comparison.py --count 50 --output demo_report.txt
    uv run python demo_comparison.py --count 20 --skip-captioning  # tags only, fast
"""
from __future__ import annotations

import argparse
import io
import os
import random
import sys
import time
from pathlib import Path
from typing import Optional

from PIL import Image

from immich_api import ImmichClient, load_dotenv
from vlm_backend import OllamaVLM

load_dotenv()
from immich_caption_and_tag_v2 import (
    ZeroShotRunner,
    caption_based_tags,
    choose_device,
    compute_ai_tags,
    filter_preds,
    normalize_image_for_model,
    read_labels,
    read_taxonomy_map,
)

DEFAULT_ZERO_SHOT_MODEL = "google/siglip-so400m-patch14-384"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Demo: old vs new captions and tags for random Immich assets.")
    p.add_argument("--count", type=int, default=50, help="Number of assets to demo (default: 50).")
    p.add_argument("--seed", type=int, default=42, help="Random seed for reproducible selection.")
    p.add_argument("--labels-file", default="labels_curated_hierarchical.txt")
    p.add_argument("--taxonomy-map", default="labels_taxonomy_map.csv")
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    p.add_argument("--zero-shot-model", default=DEFAULT_ZERO_SHOT_MODEL)
    p.add_argument("--vlm-model", default="qwen2.5vl:7b")
    p.add_argument("--vlm-url", default="http://localhost:11434")
    p.add_argument("--skip-captioning", action="store_true", help="Skip VLM captions (tags only).")
    p.add_argument("--output", default=None, help="Save report to this file (also prints to stdout).")
    p.add_argument("--pool-size", type=int, default=500,
                   help="Sample N assets from this pool before picking --count at random.")
    return p.parse_args()


def fetch_asset_pool(client: ImmichClient, pool_size: int) -> list[dict]:
    """Fetch a diverse pool of assets with exif+people+tags for random sampling."""
    # Spread across multiple pages to get diversity across the library
    all_items: list[dict] = []
    page_size = 100
    # Pull from 5 different spots in the library (first, last, and 3 middle pages)
    # We do this by fetching in reverse order and ascending order
    for order in ("desc", "asc"):
        pages_needed = pool_size // (2 * page_size) + 1
        body: dict = {
            "size": page_size,
            "withExif": True,
            "withPeople": True,
            "order": order,
        }
        for page in range(1, pages_needed + 1):
            body["page"] = page
            resp = client._request("POST", "/api/search/metadata", body=body)
            assert isinstance(resp, dict)
            items = resp.get("assets", {}).get("items", [])
            all_items.extend(items)
            if not resp.get("assets", {}).get("hasNextPage", False):
                break
            if len(all_items) >= pool_size:
                break

    # Deduplicate by id
    seen: set[str] = set()
    unique: list[dict] = []
    for item in all_items:
        if item["id"] not in seen:
            seen.add(item["id"])
            unique.append(item)
    return unique


def get_existing_tags(client: ImmichClient, asset_id: str) -> tuple[list[str], list[str]]:
    """Return (ai_tags, human_tags) currently on the asset."""
    data = client._request("GET", f"/api/assets/{asset_id}")
    assert isinstance(data, dict)
    ai_tags = [t["value"] for t in data.get("tags", []) if t.get("value", "").startswith("ai:")]
    human_tags = [t["value"] for t in data.get("tags", []) if not t.get("value", "").startswith("ai:")]
    return ai_tags, human_tags


def render_report(rows: list[dict], out) -> None:
    sep = "─" * 80
    for i, row in enumerate(rows, 1):
        print(sep, file=out)
        people_str = f"  👥 People: {', '.join(row['people'])}" if row["people"] else ""
        print(f"[{i:02d}] {row['filename']}{people_str}", file=out)
        print(f"     Asset ID: {row['asset_id']}", file=out)

        print("\n  ── BEFORE (existing in Immich) ──────────────────", file=out)
        caption_old = row.get("old_description") or "(none)"
        print(f"  Caption : {caption_old}", file=out)
        if row["old_ai_tags"]:
            for t in row["old_ai_tags"]:
                print(f"    tag: {t}", file=out)
        else:
            print("    tags: (none)", file=out)

        print("\n  ── AFTER  (v2 pipeline) ─────────────────────────", file=out)
        caption_new = row.get("new_description") or "(VLM not run)"
        if caption_new == caption_old:
            caption_new += "  [unchanged]"
        print(f"  Caption : {caption_new}", file=out)
        if row["new_ai_tags"]:
            for t in row["new_ai_tags"]:
                marker = "  NEW" if t not in row["old_ai_tags"] else ""
                print(f"    tag: {t}{marker}", file=out)
        else:
            print("    tags: (none — below threshold)", file=out)
        print(file=out)

    print(sep, file=out)


def main() -> int:
    args = parse_args()

    base_url = os.environ.get("IMMICH_URL", "").strip()
    api_key = os.environ.get("IMMICH_API_KEY", "").strip()
    if not base_url or not api_key:
        print("ERROR: IMMICH_URL and IMMICH_API_KEY must be set.", file=sys.stderr)
        return 1

    labels = read_labels(Path(args.labels_file))
    taxonomy = read_taxonomy_map(Path(args.taxonomy_map))
    device = choose_device(args.device)
    client = ImmichClient(base_url, api_key)

    # VLM
    vlm: Optional[OllamaVLM] = None
    if not args.skip_captioning:
        vlm = OllamaVLM(args.vlm_model, base_url=args.vlm_url)
        if vlm.is_available():
            print(f"VLM: {args.vlm_model} is available via Ollama ✓", file=sys.stderr)
        else:
            print(
                f"WARNING: Ollama not reachable at {args.vlm_url}. "
                "Running tags-only demo. Install Ollama + pull a model for caption comparison.",
                file=sys.stderr,
            )
            vlm = None

    print("Loading SigLIP model...", file=sys.stderr)
    runner = ZeroShotRunner(device, args.zero_shot_model, labels, verbose=False)

    # Asset pool
    print(f"Fetching asset pool (up to {args.pool_size})...", file=sys.stderr)
    pool = fetch_asset_pool(client, args.pool_size)
    print(f"  Pool: {len(pool)} assets", file=sys.stderr)

    # Prefer assets that already have a description (more interesting comparison)
    with_desc = [a for a in pool if a.get("exifInfo", {}) and a["exifInfo"].get("description")]
    without_desc = [a for a in pool if a not in with_desc]
    print(f"  With existing description: {len(with_desc)}", file=sys.stderr)

    rng = random.Random(args.seed)
    rng.shuffle(with_desc)
    rng.shuffle(without_desc)

    # Prefer those with descriptions; fill up with others if needed
    selected = (with_desc + without_desc)[: args.count]
    rng.shuffle(selected)

    print(f"\nProcessing {len(selected)} assets...\n", file=sys.stderr)

    rows: list[dict] = []
    t_start = time.time()

    for idx, item in enumerate(selected, 1):
        asset_id = item["id"]
        fname = item.get("originalFileName", asset_id)
        people = [p["name"] for p in item.get("people", []) if p.get("name")]
        old_description = (item.get("exifInfo") or {}).get("description") or None

        elapsed = time.time() - t_start
        eta = (elapsed / idx) * (len(selected) - idx) if idx > 1 else 0
        print(
            f"[{idx:02d}/{len(selected)}] {fname}"
            + (f" (people: {', '.join(people)})" if people else "")
            + (f"  — ETA {eta:.0f}s" if idx > 1 else ""),
            file=sys.stderr,
        )

        # Get existing ai: tags
        try:
            old_ai_tags, _ = get_existing_tags(client, asset_id)
        except Exception as exc:
            print(f"  Warning: could not fetch tags: {exc}", file=sys.stderr)
            old_ai_tags = []

        # Download thumbnail
        try:
            thumb_bytes = client.get_thumbnail(asset_id)
            img = Image.open(io.BytesIO(thumb_bytes))
            img = normalize_image_for_model(img, 768)
        except Exception as exc:
            print(f"  Error: thumbnail failed: {exc}", file=sys.stderr)
            continue

        # SigLIP classification
        try:
            pairs = runner.scores_batch([img])[0]
            preds = filter_preds(pairs, 6, 0.32, 0.70)
            if not preds:
                # fallback thresholds
                preds = filter_preds(pairs, 3, 0.20, 0.50)
            new_ai_tags = compute_ai_tags(preds, taxonomy, 6)
        except Exception as exc:
            print(f"  Error: classification failed: {exc}", file=sys.stderr)
            new_ai_tags = []

        # VLM caption
        new_description = old_description
        if vlm:
            try:
                caption = vlm.caption(thumb_bytes, people)
                if caption:
                    new_description = caption
            except Exception as exc:
                print(f"  VLM error: {exc}", file=sys.stderr)

        # Fallback: synthesize tags from caption nouns when zero-shot found nothing
        if not new_ai_tags and new_description:
            new_ai_tags = caption_based_tags(new_description, taxonomy)

        rows.append({
            "asset_id": asset_id,
            "filename": fname,
            "people": people,
            "old_description": old_description,
            "old_ai_tags": old_ai_tags,
            "new_description": new_description,
            "new_ai_tags": new_ai_tags,
        })

    total = time.time() - t_start
    print(f"\nCompleted {len(rows)} assets in {total:.0f}s ({total/max(len(rows),1):.1f}s/asset)\n",
          file=sys.stderr)

    # Write report
    outputs = [sys.stdout]
    if args.output:
        f = open(args.output, "w", encoding="utf-8")
        outputs.append(f)

    header = (
        f"IMMICH TAGGER DEMO — {len(rows)} random assets\n"
        f"SigLIP model : {args.zero_shot_model}\n"
        f"VLM model    : {args.vlm_model if vlm else 'not available (tags only)'}\n"
        f"Generated    : {time.strftime('%Y-%m-%d %H:%M')}\n"
    )

    for out in outputs:
        print(header, file=out)
        render_report(rows, out)

        # Summary stats
        captioned = sum(1 for r in rows if r["new_description"] and r["new_description"] != r["old_description"])
        improved_tags = sum(1 for r in rows if r["new_ai_tags"] and not r["old_ai_tags"])
        print(f"Summary:", file=out)
        print(f"  Captions updated   : {captioned}/{len(rows)}", file=out)
        print(f"  Tags added (was 0) : {improved_tags}/{len(rows)}", file=out)
        total_new = sum(len(r["new_ai_tags"]) for r in rows)
        total_old = sum(len(r["old_ai_tags"]) for r in rows)
        print(f"  Avg old ai: tags   : {total_old/max(len(rows),1):.1f}", file=out)
        print(f"  Avg new ai: tags   : {total_new/max(len(rows),1):.1f}", file=out)

    if args.output:
        f.close()
        print(f"\nReport saved to: {args.output}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
