#!/usr/bin/env python3
"""
API-based caption + tagging tool (v2).

Finds new (or all) Immich assets via the API, runs SigLIP zero-shot classification
for hierarchical tags, generates captions via a local Ollama VLM, and writes results
back through the Immich API — no sidecar files touched.

People already identified by Immich face recognition are injected into the caption
prompt so the VLM can name them instead of saying "a man" or "two boys".
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import queue
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from PIL import Image, ImageOps
from tqdm import tqdm

from immich_api import AssetInfo, ImmichClient, load_dotenv
from vlm_backend import DEFAULT_CAPTION_PROMPT, OllamaVLM

load_dotenv()

try:
    import torch
    from transformers import AutoModel, AutoProcessor
except Exception:
    torch = None
    AutoModel = None
    AutoProcessor = None

# ------------------------------------------------------------------ #
# Constants                                                            #
# ------------------------------------------------------------------ #

DEFAULT_ZERO_SHOT_MODEL = "google/siglip-so400m-patch14-384"
DEFAULT_VLM_MODEL = "qwen2.5vl:7b"
AI_PREFIX = "ai:"


# ------------------------------------------------------------------ #
# Argument parsing                                                     #
# ------------------------------------------------------------------ #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Tag and caption Immich assets via the API using SigLIP + a local VLM."
    )

    # --- Data sources ---
    p.add_argument("--labels-file", required=True, help="Newline-separated candidate labels.")
    p.add_argument("--taxonomy-map", default=None, help="CSV mapping label → hierarchical tag.")
    p.add_argument("--db-path", default=".immich_tagger_v2_cache.sqlite3", help="SQLite cache path.")

    # --- Asset selection ---
    g = p.add_mutually_exclusive_group()
    g.add_argument("--reprocess-all", action="store_true",
                   help="Process every asset in the library (ignores last-run date; respects model-sig cache).")
    g.add_argument("--since", default=None,
                   help="Only process assets created after this ISO date (overrides saved last-run date).")
    g.add_argument("--asset-id", action="append", default=None,
                   help="Process only this specific asset ID (repeatable). For manual testing/debugging.")
    p.add_argument("--reprocess-captions", action="store_true",
                   help="Re-generate captions even when the asset already has a description.")
    p.add_argument("--force", action="store_true",
                   help="Re-classify even if result is already cached for this model signature.")
    p.add_argument("--limit", type=int, default=0, help="Stop after processing this many assets.")

    # --- Models ---
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    p.add_argument("--zero-shot-model", default=DEFAULT_ZERO_SHOT_MODEL)
    p.add_argument("--skip-captioning", action="store_true",
                   help="Skip VLM caption generation entirely.")
    p.add_argument("--vlm-model", default=DEFAULT_VLM_MODEL,
                   help="Ollama model name for captioning (default: qwen2.5vl:7b).")
    p.add_argument("--vlm-url", default="http://localhost:11434",
                   help="VLM server base URL.")
    p.add_argument("--vlm-api", choices=["ollama", "openai"], default="ollama",
                   help="VLM protocol: 'ollama' (/api/chat) or 'openai' "
                        "(/v1/chat/completions, e.g. the llama-vlm service on :8082).")
    p.add_argument("--vlm-timeout", type=int, default=120,
                   help="Seconds to wait for a VLM response per image.")
    p.add_argument("--vlm-workers", type=int, default=2,
                   help="Concurrent VLM caption requests. Should match the Ollama "
                        "server's OLLAMA_NUM_PARALLEL; extra requests just queue "
                        "inside Ollama. Set 1 to restore serial captioning.")
    p.add_argument("--caption-prompt", default=None,
                   help="Caption prompt template. Use {people_clause} for name injection. "
                        "Defaults to the built-in template in vlm_backend.py.")

    # --- Classification thresholds ---
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--top-k", type=int, default=6)
    p.add_argument("--threshold", type=float, default=0.32)
    p.add_argument("--relative-threshold", type=float, default=0.70)
    p.add_argument("--fallback-threshold", type=float, default=0.20)
    p.add_argument("--fallback-relative-threshold", type=float, default=0.50)
    p.add_argument("--fallback-top-k", type=int, default=3)
    p.add_argument("--max-side", type=int, default=768)
    p.add_argument("--vlm-max-side", type=int, default=1024,
                   help="Long-side cap (px) for images sent to the VLM. qwen2.5-vl's "
                        "image-token count grows linearly with pixels, so full-size "
                        "1440px+ previews caption 3-4x slower than a ~1024px copy.")

    # --- Output ---
    p.add_argument("--dry-run", action="store_true",
                   help="Print what would change; write nothing to Immich.")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--no-caption-tags", action="store_true",
                   help="Disable caption-noun fallback tagging when zero-shot finds nothing.")
    p.add_argument("--stop-at", default=None, metavar="HH:MM",
                   help="Stop gracefully when local time reaches HH:MM (for overnight windows).")

    return p.parse_args()


# ------------------------------------------------------------------ #
# Label / taxonomy utilities (self-contained, no v1 import)           #
# ------------------------------------------------------------------ #

def read_labels(labels_file: Path) -> List[str]:
    labels: List[str] = []
    for line in labels_file.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s and not s.startswith("#"):
            labels.append(s)
    if not labels:
        raise ValueError(f"No labels found in {labels_file}")
    return labels


def normalize_tag(s: str) -> str:
    s = s.strip().lower().replace("_", " ")
    s = re.sub(r"[^a-z0-9:+#&' /-]+", "", s)
    s = re.sub(r"\s+", " ", s).strip(" -/")
    return s


def read_taxonomy_map(path: Optional[Path]) -> Dict[str, List[str]]:
    if not path or not path.exists():
        return {}
    mapping: Dict[str, List[str]] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        sample = f.read(4096)
        f.seek(0)
        has_header = "label" in sample.lower() and ("tag" in sample.lower() or "taxonomy" in sample.lower())
        reader = csv.DictReader(f) if has_header else None
        if reader:
            cols = [c or "" for c in (reader.fieldnames or [])]
            label_col = next(
                (c for c in cols if c.lower() in ("label", "raw_label", "source", "source_label")), cols[0]
            )
            tag_col = next(
                (c for c in cols if c.lower() in ("tags", "tag", "hierarchical_tag", "taxonomy_tag", "mapped_tag")),
                cols[-1],
            )
            for row in reader:
                label = normalize_tag(row.get(label_col, ""))
                raw_tags = row.get(tag_col, "") or ""
                tags = [normalize_tag(p) for p in re.split(r"[|;]", raw_tags) if normalize_tag(p)]
                if label and tags:
                    mapping[label] = _dedupe(tags)
        else:
            f.seek(0)
            for row in csv.reader(f):
                if len(row) >= 2:
                    label = normalize_tag(row[0])
                    tags = [normalize_tag(p) for p in re.split(r"[|;]", row[1]) if normalize_tag(p)]
                    if label and tags:
                        mapping[label] = _dedupe(tags)
    return mapping


def map_labels(label: str, taxonomy: Dict[str, List[str]]) -> List[str]:
    return taxonomy.get(normalize_tag(label), [normalize_tag(label)])


def _dedupe(items: Sequence[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for item in items:
        key = item.casefold()
        if key not in seen:
            seen.add(key)
            out.append(item)
    return out


def make_ai_tag(raw: str) -> str:
    """Normalize a raw taxonomy tag and prepend ai: if not already present."""
    t = normalize_tag(raw)
    return t if t.startswith(AI_PREFIX) else f"{AI_PREFIX}{t}"


# ------------------------------------------------------------------ #
# SigLIP zero-shot runner                                             #
# ------------------------------------------------------------------ #

def choose_device(device_arg: str) -> str:
    if device_arg != "auto":
        return device_arg
    if torch is not None and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def normalize_image_for_model(img: Image.Image, max_side: int) -> Image.Image:
    img = ImageOps.exif_transpose(img).convert("RGB")
    w, h = img.size
    long_side = max(w, h)
    if long_side > max_side:
        scale = max_side / float(long_side)
        img = img.resize(
            (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
            Image.Resampling.LANCZOS,
        )
    return img


def extract_video_frame(video_bytes: bytes) -> bytes:
    """Extract the first frame of a video as JPEG bytes via ffmpeg.

    Used for assets Immich has no thumbnail for (e.g. MVIMG motion-photo
    videos). ffmpeg needs a seekable input for mp4, hence the temp file.
    """
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not installed")
    with tempfile.NamedTemporaryFile(suffix=".mp4") as tmp:
        tmp.write(video_bytes)
        tmp.flush()
        proc = subprocess.run(
            [
                "ffmpeg", "-v", "error", "-i", tmp.name,
                "-frames:v", "1", "-f", "image2pipe", "-c:v", "mjpeg", "-q:v", "2", "-",
            ],
            capture_output=True,
            timeout=120,
        )
    if proc.returncode != 0 or not proc.stdout:
        raise RuntimeError(
            f"ffmpeg failed: {proc.stderr.decode(errors='replace').strip() or 'no output'}"
        )
    return proc.stdout


class ZeroShotRunner:
    """SigLIP zero-shot scorer with text embeddings precomputed once.

    The candidate labels never change during a run, so their embeddings are
    computed a single time here; each batch then only encodes the images.
    (The HF zero-shot pipeline this replaces re-encoded all labels through the
    text tower on every call.) Scoring matches the pipeline exactly:
    sigmoid(img @ txt.T * logit_scale.exp() + logit_bias) with the pipeline's
    default "This is a photo of {}." template and max_length padding.
    """

    def __init__(self, device: str, model_name: str, labels: Sequence[str], verbose: bool = False):
        if torch is None or AutoModel is None:
            raise RuntimeError("transformers not available — install with: uv add transformers torch")
        if verbose:
            print(f"Loading zero-shot model: {model_name} on {device}", file=sys.stderr)
        self._device = device
        self._model = AutoModel.from_pretrained(model_name).to(device).eval()
        self._processor = AutoProcessor.from_pretrained(model_name, backend="torchvision")
        self._labels = [normalize_tag(str(l)) for l in labels]
        texts = [f"This is a photo of {l}." for l in labels]
        text_inputs = self._processor.tokenizer(
            texts, padding="max_length", truncation=True, return_tensors="pt"
        ).to(device)
        with torch.inference_mode():
            text_embeds = self._as_tensor(self._model.get_text_features(**text_inputs))
            self._text_embeds = torch.nn.functional.normalize(text_embeds, dim=-1)
            self._logit_scale = self._model.logit_scale.exp()
            self._logit_bias = self._model.logit_bias

    @staticmethod
    def _as_tensor(out):
        """get_*_features returns a plain tensor in older transformers, a model
        output object with pooler_output in newer ones."""
        return out if torch.is_tensor(out) else out.pooler_output

    def scores_batch(self, images: Sequence[Image.Image]) -> List[List[Tuple[str, float]]]:
        """Return, per image, ALL (label, score) pairs sorted by score descending."""
        inputs = self._processor(images=list(images), return_tensors="pt").to(self._device)
        with torch.inference_mode():
            image_embeds = self._as_tensor(self._model.get_image_features(**inputs))
            image_embeds = torch.nn.functional.normalize(image_embeds, dim=-1)
            logits = image_embeds @ self._text_embeds.T * self._logit_scale + self._logit_bias
            probs = torch.sigmoid(logits).cpu()
        results: List[List[Tuple[str, float]]] = []
        for row in probs:
            pairs = sorted(zip(self._labels, row.tolist()), key=lambda x: x[1], reverse=True)
            results.append(pairs)
        return results


def filter_preds(
    pairs: List[Tuple[str, float]],
    top_k: int,
    threshold: float,
    relative_threshold: float,
) -> List[Tuple[str, float]]:
    """Apply score thresholds to already-sorted (label, score) pairs — no inference."""
    best = pairs[0][1] if pairs else 0.0
    return [
        (lab, sc) for lab, sc in pairs
        if sc >= threshold and sc >= best * relative_threshold
    ][:top_k]


# ------------------------------------------------------------------ #
# SQLite cache                                                         #
# ------------------------------------------------------------------ #

def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS asset_cache (
            asset_id   TEXT PRIMARY KEY,
            model_sig  TEXT NOT NULL DEFAULT '',
            tagged_at  REAL NOT NULL DEFAULT 0,
            data_json  TEXT NOT NULL DEFAULT ''
        );
        CREATE TABLE IF NOT EXISTS run_state (
            key   TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        """
    )
    conn.commit()


def model_signature(args: argparse.Namespace, labels: Sequence[str], taxonomy: Dict[str, List[str]]) -> str:
    payload = {
        "zero_shot_model": args.zero_shot_model,
        "vlm_model": args.vlm_model,
        "labels": list(labels),
        "taxonomy": taxonomy,
        "threshold": args.threshold,
        "relative_threshold": args.relative_threshold,
        "top_k": args.top_k,
        "fallback_threshold": args.fallback_threshold,
        "fallback_relative_threshold": args.fallback_relative_threshold,
        "fallback_top_k": args.fallback_top_k,
        "skip_captioning": args.skip_captioning,
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()[:16]


def is_cached(conn: sqlite3.Connection, asset_id: str, model_sig: str) -> bool:
    row = conn.execute(
        "SELECT model_sig FROM asset_cache WHERE asset_id = ?", (asset_id,)
    ).fetchone()
    return row is not None and row[0] == model_sig


def put_cache(conn: sqlite3.Connection, asset_id: str, model_sig: str, data: dict) -> None:
    conn.execute(
        """
        INSERT INTO asset_cache(asset_id, model_sig, tagged_at, data_json)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(asset_id) DO UPDATE SET
            model_sig=excluded.model_sig,
            tagged_at=excluded.tagged_at,
            data_json=excluded.data_json
        """,
        (asset_id, model_sig, time.time(), json.dumps(data)),
    )
    conn.commit()


def get_last_run_date(conn: sqlite3.Connection) -> Optional[datetime]:
    row = conn.execute("SELECT value FROM run_state WHERE key = 'last_run_at'").fetchone()
    if not row:
        return None
    try:
        return datetime.fromisoformat(row[0])
    except Exception:
        return None


def set_last_run_date(conn: sqlite3.Connection, dt: datetime) -> None:
    conn.execute(
        "INSERT INTO run_state(key, value) VALUES ('last_run_at', ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (dt.isoformat(),),
    )
    conn.commit()


# ------------------------------------------------------------------ #
# Tag helpers                                                          #
# ------------------------------------------------------------------ #

def compute_ai_tags(
    preds: List[Tuple[str, float]],
    taxonomy: Dict[str, List[str]],
    top_k: int,
) -> List[str]:
    """Map raw SigLIP predictions to deduplicated ai: taxonomy tags."""
    seen: set[str] = set()
    tags: List[str] = []
    for label, _score in preds[:top_k]:
        for raw in map_labels(label, taxonomy):
            t = make_ai_tag(raw)
            key = t.casefold()
            if key not in seen:
                seen.add(key)
                tags.append(t)
    return tags


_STOP_WORDS = frozenset(
    "a an the is are was were be been being have has had do does did "
    "will would could should may might must shall to of in on at by "
    "for with as from into and or but not no this that these those "
    "it he she they we you i me my his her its our your their "
    "what where when who how there here up out".split()
)


def _simple_stems(text: str) -> set[str]:
    """Lowercase words from text, with crude suffix removal for matching."""
    words = set()
    for w in re.split(r"[^a-z]+", text.lower()):
        if len(w) < 3 or w in _STOP_WORDS:
            continue
        words.add(w)
        # Strip common suffixes so "hiking" → "hike", "dogs" → "dog"
        for suffix, min_root in (("ing", 3), ("tion", 4), ("ers", 3), ("ies", 3), ("es", 3), ("s", 3)):
            if w.endswith(suffix) and len(w) - len(suffix) >= min_root:
                words.add(w[: -len(suffix)])
                break
    return words


def caption_based_tags(
    caption: str,
    taxonomy: Dict[str, List[str]],
    max_tags: int = 5,
) -> List[str]:
    """Fallback: match taxonomy labels against caption words when zero-shot produced nothing."""
    if not caption:
        return []
    caption_stems = _simple_stems(caption)
    seen: set[str] = set()
    tags: List[str] = []
    for label, raw_tags in taxonomy.items():
        label_stems = _simple_stems(label)
        if not label_stems:
            continue
        if label_stems.issubset(caption_stems):
            for raw in raw_tags:
                # Skip the legacy Prospective/Nouns holding area — too generic
                if raw.lower().startswith("prospective/"):
                    continue
                t = make_ai_tag(raw)
                key = t.casefold()
                if key not in seen:
                    seen.add(key)
                    tags.append(t)
        if len(tags) >= max_tags:
            break
    return tags[:max_tags]


# ------------------------------------------------------------------ #
# Main                                                                 #
# ------------------------------------------------------------------ #

def main() -> int:
    args = parse_args()

    # Credentials from environment (set by run script via .env)
    base_url = os.environ.get("IMMICH_URL", "").strip()
    api_key = os.environ.get("IMMICH_API_KEY", "").strip()
    if not base_url or not api_key:
        print("ERROR: IMMICH_URL and IMMICH_API_KEY must be set in the environment.", file=sys.stderr)
        return 1

    # --- Time window ---
    stop_dt: Optional[datetime] = None
    if args.stop_at:
        h, m = map(int, args.stop_at.split(":"))
        now = datetime.now()
        stop_dt = now.replace(hour=h, minute=m, second=0, microsecond=0)
        if stop_dt <= now:
            stop_dt += timedelta(days=1)
        print(f"Will stop at {stop_dt.strftime('%a %Y-%m-%d %H:%M')} local time", file=sys.stderr)

    labels_file = Path(args.labels_file).expanduser().resolve()
    labels = read_labels(labels_file)
    taxonomy = read_taxonomy_map(Path(args.taxonomy_map).expanduser().resolve() if args.taxonomy_map else None)
    model_sig = model_signature(args, labels, taxonomy)
    device = choose_device(args.device)

    if args.verbose:
        print(f"Device: {device}", file=sys.stderr)
        print(f"Labels: {len(labels)} from {labels_file}", file=sys.stderr)
        print(f"Taxonomy entries: {len(taxonomy)}", file=sys.stderr)

    # --- Immich client ---
    client = ImmichClient(base_url, api_key)
    print("Loading Immich tag catalogue...", file=sys.stderr)
    client.load_all_tags()
    if args.verbose:
        print(f"  {len(client._tag_cache)} tags loaded", file=sys.stderr)

    # --- VLM ---
    vlm: Optional[OllamaVLM] = None
    caption_prompt = args.caption_prompt or DEFAULT_CAPTION_PROMPT
    if not args.skip_captioning:
        vlm = OllamaVLM(args.vlm_model, base_url=args.vlm_url, timeout=args.vlm_timeout,
                        api=args.vlm_api)
        if not vlm.is_available():
            print(
                f"WARNING: VLM server not reachable at {args.vlm_url}. "
                "Captions will be skipped. Start the server or pass --skip-captioning.",
                file=sys.stderr,
            )
            vlm = None
        elif args.verbose:
            print(f"VLM: {args.vlm_model} via {args.vlm_url}", file=sys.stderr)

    # --- Zero-shot model ---
    print("Loading zero-shot classification model...", file=sys.stderr)
    runner = ZeroShotRunner(device, args.zero_shot_model, labels, verbose=args.verbose)

    # --- SQLite cache ---
    db_path = Path(args.db_path).expanduser().resolve()
    # check_same_thread: the caption/write worker thread commits cache entries;
    # all access is serialized through db_lock below.
    conn = sqlite3.connect(db_path, check_same_thread=False)
    init_db(conn)

    # --- Asset list ---
    if args.asset_id:
        print(f"Mode: specific asset ID(s): {args.asset_id}", file=sys.stderr)
        assets = [client.get_asset_by_id(aid) for aid in args.asset_id]
    elif args.reprocess_all:
        since: Optional[datetime] = None
        print("Mode: reprocess ALL assets", file=sys.stderr)
    elif args.since:
        since = datetime.fromisoformat(args.since).replace(tzinfo=timezone.utc)
        print(f"Mode: assets created after {since.date()}", file=sys.stderr)
    else:
        since = get_last_run_date(conn)
        if since:
            print(f"Mode: assets created after last run ({since.date()})", file=sys.stderr)
        else:
            print("Mode: no last-run date found — processing ALL assets (first run)", file=sys.stderr)

    if not args.asset_id:
        print("Fetching asset list from Immich...", file=sys.stderr)
        assets = list(tqdm(client.find_new_assets(since=since), desc="Fetching", unit="asset"))

    if args.limit > 0:
        assets = assets[: args.limit]

    if not assets:
        print("No assets to process.", file=sys.stderr)
        if not args.reprocess_all and not args.asset_id:
            set_last_run_date(conn, datetime.now(timezone.utc))
        conn.close()
        return 0

    print(f"Assets to process: {len(assets)}", file=sys.stderr)

    written = 0
    skipped = 0
    skipped_no_media = 0
    errors = 0
    caption_count = 0

    batch_size = args.batch_size
    db_lock = threading.Lock()

    # --- Three-stage pipeline ---------------------------------------- #
    # main thread: fetch (pooled) + SigLIP classify (CPU)
    #   -> caption_q -> N caption workers (only talk to Ollama, keep iGPU fed)
    #   -> write_q   -> 1 writer thread (Immich API + sqlite; off the GPU path)
    # Queues are bounded so the main thread runs just far enough ahead to
    # keep the VLM busy without holding the whole library in memory.
    vlm_workers = max(1, args.vlm_workers)
    caption_q: queue.Queue = queue.Queue(maxsize=max(8, 4 * vlm_workers))
    write_q: queue.Queue = queue.Queue(maxsize=64)
    # written/captions/write-errors are owned by the writer thread alone;
    # main-thread errors (thumbnails, classification) stay in `errors`.
    writer_totals = {"written": 0, "captions": 0, "errors": 0}

    def caption_worker() -> None:
        while True:
            item = caption_q.get()
            if item is None:
                break
            asset, vlm_bytes, ai_tags, preds = item
            new_description = asset.description
            captioned = False
            if vlm and (not asset.description or args.reprocess_captions):
                try:
                    caption = vlm.caption(vlm_bytes, asset.people_names, caption_prompt)
                    if caption:
                        new_description = caption
                        captioned = True
                except Exception as exc:
                    if args.verbose:
                        print(f"VLM error for {asset.id}: {exc}", file=sys.stderr)
            write_q.put((asset, ai_tags, preds, new_description, captioned))

    def writer() -> None:
        while True:
            item = write_q.get()
            if item is None:
                break
            asset, ai_tags, preds, new_description, captioned = item

            # Fallback: synthesize tags from caption nouns when zero-shot found nothing
            if not ai_tags and new_description and not args.no_caption_tags:
                ai_tags = caption_based_tags(new_description, taxonomy)

            if args.verbose or args.dry_run:
                people_str = f", people={asset.people_names}" if asset.people_names else ""
                print(f"\n{asset.file_name} ({asset.id}){people_str}")
                print(f"  tags: {ai_tags}")
                if new_description:
                    print(f"  caption: {new_description}")
                if preds:
                    print(f"  scores: {[(l, round(s, 3)) for l, s in preds[:5]]}")

            if args.dry_run:
                writer_totals["written"] += 1
                writer_totals["captions"] += int(captioned)
                continue

            try:
                # In reprocess mode, remove stale ai: tags before assigning new ones
                if args.reprocess_all:
                    old_tag_ids = client.get_asset_ai_tag_ids(asset.id)
                    for old_id in old_tag_ids:
                        client.remove_tag_from_assets(old_id, [asset.id])

                # Assign new ai: tags
                for tag_value in ai_tags:
                    tag_id = client.ensure_tag(tag_value)
                    client.assign_tag_to_assets(tag_id, [asset.id])

                # Update description if it changed
                if new_description and new_description != asset.description:
                    client.update_description(asset.id, new_description)

                with db_lock:
                    put_cache(conn, asset.id, model_sig, {
                        "tags": ai_tags,
                        "description": new_description,
                    })
                writer_totals["written"] += 1
                writer_totals["captions"] += int(captioned)
            except Exception as exc:
                writer_totals["errors"] += 1
                print(f"API write error for {asset.id}: {exc}", file=sys.stderr)

    def fetch_source_bytes(asset: AssetInfo) -> Optional[bytes]:
        """Image bytes for an asset: its Immich thumbnail, else the original
        (photos as-is, videos via an ffmpeg-extracted first frame).

        Returns None when no pixels can be had at all — the asset is then
        skipped, not counted as an error (e.g. MVIMG motion-photo videos on a
        host without ffmpeg)."""
        try:
            return client.get_thumbnail(asset.id)
        except Exception as thumb_exc:
            try:
                orig_bytes = client.get_original(asset.id)
                if asset.asset_type == "VIDEO":
                    return extract_video_frame(orig_bytes)
                return orig_bytes
            except Exception as orig_exc:
                if args.verbose:
                    print(
                        f"No usable media for {asset.id} ({asset.file_name}): "
                        f"thumbnail: {thumb_exc}; original: {orig_exc} — skipping",
                        file=sys.stderr,
                    )
                return None

    # Sentinel from fetch_one: asset has no fetchable pixels — skip, not an error.
    SKIP_NO_MEDIA = "skip_no_media"

    def fetch_one(asset: AssetInfo) -> Tuple[AssetInfo, Image.Image, bytes] | str | None:
        src_bytes = fetch_source_bytes(asset)
        if src_bytes is None:
            return SKIP_NO_MEDIA
        try:
            orig = Image.open(io.BytesIO(src_bytes))
            img = normalize_image_for_model(orig, args.max_side)
            vlm_img = normalize_image_for_model(orig, args.vlm_max_side)
            vlm_buf = io.BytesIO()
            vlm_img.save(vlm_buf, format="JPEG", quality=90)
            return (asset, img, vlm_buf.getvalue())
        except Exception as exc:
            print(f"Image decode error {asset.id} ({asset.file_name}): {exc}", file=sys.stderr)
            return None

    caption_threads = [
        threading.Thread(target=caption_worker, name=f"caption-{i}", daemon=True)
        for i in range(vlm_workers)
    ]
    writer_thread = threading.Thread(target=writer, name="writer", daemon=True)
    for t in caption_threads:
        t.start()
    writer_thread.start()
    fetch_pool = ThreadPoolExecutor(max_workers=4)

    stopped_early = False
    for batch_start in tqdm(range(0, len(assets), batch_size), desc="Batches"):
        if stop_dt and datetime.now() >= stop_dt:
            print(f"\nTime window ended at {args.stop_at}. Stopping — resume tomorrow night.", file=sys.stderr)
            stopped_early = True
            break

        batch = assets[batch_start : batch_start + batch_size]

        # Filter cached
        uncached: List[AssetInfo] = []
        for asset in batch:
            with db_lock:
                cached = not args.force and is_cached(conn, asset.id, model_sig)
            if cached:
                skipped += 1
            else:
                uncached.append(asset)
        if not uncached:
            continue

        fetched = list(fetch_pool.map(fetch_one, uncached))
        skipped_no_media += sum(1 for f in fetched if f == SKIP_NO_MEDIA)
        errors += sum(1 for f in fetched if f is None)
        to_process = [f for f in fetched if isinstance(f, tuple)]
        if not to_process:
            continue

        # SigLIP classification (CPU): raw scores once, both threshold passes on
        # the same scores — the old fallback re-ran inference for empty results.
        images = [img for _, img, _ in to_process]
        try:
            raw_scores = runner.scores_batch(images)
        except Exception as exc:
            errors += len(to_process)
            print(f"Classification batch failed: {exc}", file=sys.stderr)
            continue

        for (asset, _img, vlm_bytes), pairs in zip(to_process, raw_scores):
            preds = filter_preds(pairs, args.top_k, args.threshold, args.relative_threshold)
            if not preds:
                preds = filter_preds(
                    pairs, args.fallback_top_k, args.fallback_threshold,
                    args.fallback_relative_threshold,
                )
            ai_tags = compute_ai_tags(preds, taxonomy, args.top_k)
            # Blocks when the queue is full — backpressure on the main thread
            caption_q.put((asset, vlm_bytes, ai_tags, preds))

        # Clear GPU cache if using CUDA
        if device == "cuda" and torch is not None:
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    if stopped_early:
        # Drop anything not yet captioning — uncached assets are simply
        # reprocessed on the next run (same as the old mid-batch break).
        try:
            while True:
                caption_q.get_nowait()
        except queue.Empty:
            pass

    for _ in caption_threads:
        caption_q.put(None)
    for t in caption_threads:
        t.join()
    write_q.put(None)
    writer_thread.join()
    fetch_pool.shutdown(wait=True)

    written += writer_totals["written"]
    caption_count += writer_totals["captions"]
    errors += writer_totals["errors"]

    # Update last-run date for future incremental runs
    if not args.reprocess_all and not args.asset_id and not args.dry_run:
        set_last_run_date(conn, datetime.now(timezone.utc))

    conn.close()

    print(
        f"\nDone. assets={len(assets)} written={written} skipped_cached={skipped} "
        f"skipped_no_media={skipped_no_media} captions={caption_count} errors={errors}"
    )
    if args.dry_run:
        print("Dry run — nothing written to Immich.")
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
