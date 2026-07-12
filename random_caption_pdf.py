#!/usr/bin/env python3
"""Build a PDF contact sheet of random Immich images with their captions.

Pulls N random image assets via the Immich API and lays them out in a
3-column grid (2 rows per letter page), each image with its caption
(exifInfo.description) underneath.

Usage:
    uv run python random_caption_pdf.py 30 sample.pdf
    uv run python random_caption_pdf.py 30 sample.pdf --tag ai:water/marine/boat
    uv run python random_caption_pdf.py 30 sample.pdf --tag ai:nature --tag ai:people/family
"""
from __future__ import annotations

import argparse
import io
import os
import sys

from PIL import Image
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader, simpleSplit
from reportlab.pdfgen import canvas

from immich_api import AssetInfo, ImmichClient, load_dotenv

# Layout (points; letter page = 612 x 792)
PAGE_W, PAGE_H = letter
MARGIN = 36
COLS = 3
GUTTER = 14
CELL_W = (PAGE_W - 2 * MARGIN - (COLS - 1) * GUTTER) / COLS
IMG_BOX_H = 160
CAPTION_GAP = 6
CAPTION_FONT = ("Helvetica", 7.5)
CAPTION_LEADING = 9
CAPTION_MAX_LINES = 6
FILENAME_FONT = ("Helvetica-Oblique", 6)
ROW_H = IMG_BOX_H + CAPTION_GAP + CAPTION_MAX_LINES * CAPTION_LEADING + 14
ROWS = max(1, int((PAGE_H - 2 * MARGIN) // ROW_H))


def fetch_random_images(
    client: ImmichClient,
    count: int,
    tag_ids: list[str] | None = None,
) -> list[tuple[AssetInfo, Image.Image]]:
    """Random assets paired with their preview thumbnails, skipping unfetchable ones."""
    picked: list[tuple[AssetInfo, Image.Image]] = []
    seen: set[str] = set()
    attempts = 0
    while len(picked) < count and attempts < 5:
        attempts += 1
        for asset in client.search_random(count - len(picked), tag_ids=tag_ids):
            if asset.id in seen:
                continue
            seen.add(asset.id)
            try:
                raw = client.get_thumbnail(asset.id)
                img = Image.open(io.BytesIO(raw))
                img.load()
            except Exception as exc:
                print(f"  skipping {asset.file_name} ({asset.id}): {exc}", file=sys.stderr)
                continue
            picked.append((asset, img.convert("RGB")))
            if len(picked) == count:
                break
    return picked


def draw_cell(c: canvas.Canvas, x: float, y_top: float, asset: AssetInfo, img: Image.Image) -> None:
    """Draw one image + caption cell with its top-left corner at (x, y_top)."""
    # Image, scaled to fit the box, centered horizontally
    scale = min(CELL_W / img.width, IMG_BOX_H / img.height)
    w, h = img.width * scale, img.height * scale
    c.drawImage(
        ImageReader(img),
        x + (CELL_W - w) / 2,
        y_top - IMG_BOX_H + (IMG_BOX_H - h) / 2,
        width=w,
        height=h,
    )

    text_y = y_top - IMG_BOX_H - CAPTION_GAP - FILENAME_FONT[1]
    c.setFont(*FILENAME_FONT)
    c.setFillGray(0.45)
    c.drawString(x, text_y, asset.file_name[:60])

    caption = asset.description or "(no caption)"
    c.setFont(*CAPTION_FONT)
    c.setFillGray(0.0)
    lines = simpleSplit(caption, CAPTION_FONT[0], CAPTION_FONT[1], CELL_W)
    if len(lines) > CAPTION_MAX_LINES:
        lines = lines[:CAPTION_MAX_LINES]
        lines[-1] = lines[-1].rstrip() + " …"
    text_y -= CAPTION_LEADING
    for line in lines:
        c.drawString(x, text_y, line)
        text_y -= CAPTION_LEADING


def build_pdf(items: list[tuple[AssetInfo, Image.Image]], out_path: str) -> None:
    c = canvas.Canvas(out_path, pagesize=letter)
    per_page = COLS * ROWS
    for i, (asset, img) in enumerate(items):
        slot = i % per_page
        if i and slot == 0:
            c.showPage()
        col, row = slot % COLS, slot // COLS
        x = MARGIN + col * (CELL_W + GUTTER)
        y_top = PAGE_H - MARGIN - row * ROW_H
        draw_cell(c, x, y_top, asset, img)
    c.save()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("count", type=int, help="Number of random images to include")
    parser.add_argument("output", help="Output PDF file name")
    parser.add_argument(
        "--tag",
        action="append",
        dest="tags",
        metavar="TAG_VALUE",
        help="Restrict to assets with this tag (full hierarchical value, e.g. "
        "ai:water/marine/boat). Repeatable; assets must carry all given tags.",
    )
    args = parser.parse_args()

    load_dotenv()
    base_url = os.environ.get("IMMICH_URL", "").strip()
    api_key = os.environ.get("IMMICH_API_KEY", "").strip()
    if not base_url or not api_key:
        print("ERROR: IMMICH_URL and IMMICH_API_KEY must be set (see .env).", file=sys.stderr)
        sys.exit(1)

    client = ImmichClient(base_url, api_key)

    tag_ids: list[str] | None = None
    if args.tags:
        tag_ids = []
        for value in args.tags:
            tag_id = client.find_tag_id(value)
            if tag_id is None:
                print(f"ERROR: no such tag: {value!r}", file=sys.stderr)
                sys.exit(1)
            tag_ids.append(tag_id)

    print(f"Fetching {args.count} random images from {base_url} ...")
    items = fetch_random_images(client, args.count, tag_ids=tag_ids)
    if not items:
        print("ERROR: no images could be fetched.", file=sys.stderr)
        sys.exit(1)
    if len(items) < args.count:
        print(f"WARNING: only {len(items)}/{args.count} images fetched.", file=sys.stderr)

    build_pdf(items, args.output)
    print(f"Wrote {len(items)} images to {args.output}")


if __name__ == "__main__":
    main()
