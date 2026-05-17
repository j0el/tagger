#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Set, Tuple
from xml.etree import ElementTree as ET

from PIL import Image, ImageOps, UnidentifiedImageError
from tqdm import tqdm

try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
    HEIF_ENABLED = True
except Exception:
    HEIF_ENABLED = False

try:
    import torch
    from transformers import BlipForConditionalGeneration, BlipProcessor
except Exception:
    torch = None
    BlipForConditionalGeneration = None
    BlipProcessor = None


NAMESPACES = {
    "x": "adobe:ns:meta/",
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "dc": "http://purl.org/dc/elements/1.1/",
}
for prefix, uri in NAMESPACES.items():
    ET.register_namespace(prefix, uri)

AI_PREFIX = "ai:"
LOW_QUALITY_TAG = "ai:caption/low-quality"
DEFAULT_MODEL = "Salesforce/blip-image-captioning-large"

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".heic", ".heif"
}

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "else", "when", "while",
    "of", "in", "on", "at", "to", "with", "without", "for", "from", "by", "as",
    "into", "onto", "over", "under", "near", "next", "beside", "behind", "front",
    "back", "between", "through", "across", "around", "inside", "outside",
    "is", "are", "was", "were", "be", "been", "being", "am", "has", "have",
    "had", "do", "does", "did", "can", "could", "would", "should", "will",
    "may", "might", "must", "there", "here", "this", "that", "these", "those",
    "it", "its", "they", "them", "their", "he", "she", "his", "her", "hers",
    "him", "you", "your", "we", "our", "i", "me", "my",
    "photo", "picture", "image", "view", "scene", "shows", "showing", "depicts",
    "contains", "featuring", "featured", "visible", "appears", "looking",
}

GENERIC = {
    "thing", "things", "object", "objects", "stuff", "item", "items", "part", "parts",
    "area", "place", "places", "background", "foreground", "something", "someone",
    "person", "people", "man", "woman", "boy", "girl", "child", "children", "group",
    "room", "wall", "floor", "hand", "hands", "face", "head", "side", "line", "set",
}

BAD_TAG_FRAGMENTS = {
    "tor", "torta", "che", "sae", "tuxed", "chandel", "residence", "radio",
}


def eprint(*args, **kwargs) -> None:
    print(*args, file=sys.stderr, **kwargs)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Version 4 BLIP repair tool: process sidecars currently marked "
            "ai:caption/low-quality, generate a better caption with BLIP-large, "
            "and replace the marker with ai:prospective/... tags."
        )
    )
    p.add_argument("root", help="Folder containing .xmp sidecars.")
    p.add_argument("--recurse", action="store_true", help="Scan recursively.")
    p.add_argument("--apply", action="store_true", help="Actually write sidecars. Default is dry run.")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--model", default=DEFAULT_MODEL, help="Caption model. Default: Salesforce/blip-image-captioning-large")
    p.add_argument("--device", choices=["auto", "mps", "cuda", "cpu"], default="auto")
    p.add_argument("--max-new-tokens", type=int, default=50)
    p.add_argument("--max-side", type=int, default=1024)
    p.add_argument("--heic-fallback", choices=["auto", "sips", "ffmpeg", "none"], default="auto")
    p.add_argument("--low-quality-tag", default=LOW_QUALITY_TAG)
    p.add_argument("--prospective-prefix", default="prospective")
    p.add_argument("--max-tags", type=int, default=4)
    p.add_argument("--min-term-len", type=int, default=3)
    p.add_argument("--update-caption", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--keep-low-quality-tag", action="store_true")
    p.add_argument("--limit", type=int, default=0)
    return p.parse_args()


def ensure_deps() -> None:
    if torch is None or BlipForConditionalGeneration is None or BlipProcessor is None:
        raise RuntimeError(
            "Missing dependencies. Install with:\n"
            "  uv add torch torchvision transformers pillow tqdm accelerate sentencepiece protobuf pillow-heif"
        )


def choose_device(device_arg: str) -> str:
    ensure_deps()
    if device_arg != "auto":
        return device_arg
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def iter_sidecars(root: Path, recurse: bool) -> Iterable[Path]:
    yield from (root.rglob("*.xmp") if recurse else root.glob("*.xmp"))


def media_path_for_sidecar(xmp_path: Path) -> Optional[Path]:
    if xmp_path.suffix.lower() != ".xmp":
        return None
    media = xmp_path.with_suffix("")
    if media.exists() and media.suffix.lower() in IMAGE_EXTENSIONS:
        return media
    return None


def read_tags_caption_tree(path: Path) -> Tuple[List[str], Optional[str], Optional[ET.ElementTree]]:
    try:
        tree = ET.parse(path)
    except ET.ParseError:
        return [], None, None

    root = tree.getroot()
    tags: List[str] = []

    for li in root.findall(".//dc:subject/rdf:Bag/rdf:li", NAMESPACES):
        if li.text and li.text.strip():
            tags.append(li.text.strip())
    if not tags:
        for li in root.findall(".//{*}subject//{*}li"):
            if li.text and li.text.strip():
                tags.append(li.text.strip())

    caption: Optional[str] = None
    desc_node = root.find(".//dc:description/rdf:Alt/rdf:li", NAMESPACES)
    if desc_node is not None and desc_node.text and desc_node.text.strip():
        caption = desc_node.text.strip()
    if caption is None:
        desc_node = root.find(".//{*}description//{*}li")
        if desc_node is not None and desc_node.text and desc_node.text.strip():
            caption = desc_node.text.strip()

    return tags, caption, tree


def has_target_tag(tags: Sequence[str], target: str) -> bool:
    target_cf = target.casefold()
    return any(t.casefold() == target_cf for t in tags)


def is_heic_path(path: Path) -> bool:
    return path.suffix.lower() in {".heic", ".heif"}


def normalize_image_for_model(img: Image.Image, max_side: int) -> Image.Image:
    img = ImageOps.exif_transpose(img).convert("RGB")
    w, h = img.size
    long_side = max(w, h)
    if long_side <= max_side:
        return img
    scale = max_side / float(long_side)
    new_size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
    return img.resize(new_size, Image.Resampling.LANCZOS)


def _open_converted_temp_image(temp_path: Path, max_side: int) -> Image.Image:
    with Image.open(temp_path) as img:
        prepared = normalize_image_for_model(img, max_side)
        prepared.load()
        return prepared


def prepare_heic_with_sips(path: Path, max_side: int) -> Optional[Image.Image]:
    if shutil.which("sips") is None:
        return None
    with tempfile.TemporaryDirectory(prefix="blip_large_heic_sips_") as td:
        out = Path(td) / "converted.jpg"
        res = subprocess.run(
            ["sips", "-s", "format", "jpeg", str(path), "--out", str(out)],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        if res.returncode != 0 or not out.exists() or out.stat().st_size == 0:
            return None
        return _open_converted_temp_image(out, max_side)


def prepare_heic_with_ffmpeg(path: Path, max_side: int) -> Optional[Image.Image]:
    if shutil.which("ffmpeg") is None:
        return None
    with tempfile.TemporaryDirectory(prefix="blip_large_heic_ffmpeg_") as td:
        out = Path(td) / "converted.jpg"
        res = subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error",
                "-i", str(path), "-frames:v", "1", "-q:v", "2", "-y", str(out),
            ],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        if res.returncode != 0 or not out.exists() or out.stat().st_size == 0:
            return None
        return _open_converted_temp_image(out, max_side)


def prepare_heic_fallback(path: Path, max_side: int, mode: str) -> Optional[Image.Image]:
    if mode == "none":
        return None
    if mode in ("auto", "sips"):
        img = prepare_heic_with_sips(path, max_side)
        if img is not None:
            return img
    if mode in ("auto", "ffmpeg"):
        img = prepare_heic_with_ffmpeg(path, max_side)
        if img is not None:
            return img
    return None


def prepare_image(path: Path, max_side: int, heic_fallback: str) -> Image.Image:
    try:
        with Image.open(path) as img:
            prepared = normalize_image_for_model(img, max_side)
            prepared.load()
            return prepared
    except (UnidentifiedImageError, OSError, ValueError):
        if is_heic_path(path):
            fallback = prepare_heic_fallback(path, max_side, heic_fallback)
            if fallback is not None:
                return fallback
        raise


class BlipLargeRunner:
    def __init__(self, model_name: str, device: str, verbose: bool):
        ensure_deps()
        self.model_name = model_name
        self.device = device
        self.verbose = verbose
        if verbose:
            eprint(f"Loading BLIP caption model: {model_name} on {device}")
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()

    def caption(self, image: Image.Image, max_new_tokens: int) -> str:
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.inference_mode():
            generated = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        text = self.processor.batch_decode(generated, skip_special_tokens=True)[0]
        return clean_model_text(text)


def clean_model_text(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.strip(" .,:;-")


def words_from_text(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+(?:'[a-z]+)?", text.lower())


def simple_singular(word: str) -> str:
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 4 and word.endswith("es") and not word.endswith(("ses", "xes", "ches", "shes")):
        return word[:-2]
    if len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]
    return word


def normalize_label(s: str) -> str:
    s = s.strip().lower()
    s = s.replace("_", " ")
    s = re.sub(r"https?://\S+", "", s)
    s = re.sub(r"[^a-z0-9&' /-]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip(" -/")
    return s


def is_bad_candidate(term: str, min_len: int) -> bool:
    term = normalize_label(term)
    if not term or len(term) < min_len:
        return True
    parts = term.split()
    if any(p in STOPWORDS for p in parts):
        return True
    if all(p in GENERIC for p in parts):
        return True
    if len(parts) == 1 and parts[0] in GENERIC:
        return True
    if term in BAD_TAG_FRAGMENTS:
        return True
    if re.fullmatch(r"[0-9x]+", term):
        return True
    return False


def collapse_repetition(words: List[str]) -> List[str]:
    out: List[str] = []
    counts: Counter[str] = Counter()
    prev = None
    for word in words:
        if word == prev:
            counts[word] += 1
            if counts[word] >= 2:
                continue
        else:
            counts[word] += 1

        if len(word) <= 4 and counts[word] >= 4:
            break
        if counts[word] >= 5:
            break

        out.append(word)
        prev = word
    return out


def candidate_terms_from_text(text: str, min_len: int) -> List[str]:
    text = normalize_label(text)
    words = collapse_repetition([simple_singular(w) for w in words_from_text(text)])

    cleaned: List[str] = []
    for word in words:
        word = normalize_label(word)
        if not word or word in STOPWORDS:
            cleaned.append("")
            continue
        cleaned.append(word)

    candidates: List[str] = []
    for n in (2, 3, 1):
        for i in range(len(cleaned) - n + 1):
            chunk = cleaned[i:i+n]
            if any(not x for x in chunk):
                continue
            phrase = " ".join(chunk)
            if is_bad_candidate(phrase, min_len):
                continue
            candidates.append(phrase)

    seen: Set[str] = set()
    out: List[str] = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def score_candidate(term: str) -> Tuple[int, int, int, str]:
    words = term.split()
    phrase_pref = 3 if len(words) == 2 else 2 if len(words) == 3 else 1
    length_score = sum(len(w) for w in words)
    return (phrase_pref, length_score, -len(words), term)


def pick_tags(model_text: str, max_tags: int, prefix: str, min_len: int) -> List[str]:
    terms = candidate_terms_from_text(model_text, min_len)
    terms.sort(key=score_candidate, reverse=True)

    selected_terms: List[str] = []
    for term in terms:
        if any(term in chosen.split() for chosen in selected_terms):
            continue
        selected_terms.append(term)
        if len(selected_terms) >= max_tags:
            break

    return [AI_PREFIX + prefix.strip(" /") + "/" + t for t in selected_terms]


def ensure_description_node(tree: ET.ElementTree) -> ET.Element:
    root = tree.getroot()
    rdf = root.find("rdf:RDF", NAMESPACES)
    if rdf is None:
        rdf = root.find(".//{*}RDF")
    if rdf is None:
        rdf = ET.SubElement(root, f"{{{NAMESPACES['rdf']}}}RDF")

    desc = rdf.find("rdf:Description", NAMESPACES)
    if desc is None:
        desc = rdf.find("{http://www.w3.org/1999/02/22-rdf-syntax-ns#}Description")
    if desc is None:
        desc = ET.SubElement(rdf, f"{{{NAMESPACES['rdf']}}}Description")
    return desc


def replace_subject_tags(tree: ET.ElementTree, tags: Sequence[str]) -> None:
    desc = ensure_description_node(tree)

    for node in list(desc.findall("dc:subject", NAMESPACES)):
        desc.remove(node)
    for node in list(desc.findall("{http://purl.org/dc/elements/1.1/}subject")):
        if node in list(desc):
            desc.remove(node)

    subject = ET.SubElement(desc, f"{{{NAMESPACES['dc']}}}subject")
    bag = ET.SubElement(subject, f"{{{NAMESPACES['rdf']}}}Bag")
    for tag in tags:
        li = ET.SubElement(bag, f"{{{NAMESPACES['rdf']}}}li")
        li.text = tag


def upsert_description(tree: ET.ElementTree, description: str) -> None:
    desc = ensure_description_node(tree)

    for node in list(desc.findall("dc:description", NAMESPACES)):
        desc.remove(node)
    for node in list(desc.findall("{http://purl.org/dc/elements/1.1/}description")):
        if node in list(desc):
            desc.remove(node)

    dc_desc = ET.SubElement(desc, f"{{{NAMESPACES['dc']}}}description")
    alt = ET.SubElement(dc_desc, f"{{{NAMESPACES['rdf']}}}Alt")
    li = ET.SubElement(alt, f"{{{NAMESPACES['rdf']}}}li")
    li.set("{http://www.w3.org/XML/1998/namespace}lang", "x-default")
    li.text = description


def merge_tags(existing: Sequence[str], generated: Sequence[str], low_quality_tag: str, keep_low_quality: bool) -> List[str]:
    out: List[str] = []
    for tag in existing:
        if not keep_low_quality and tag.casefold() == low_quality_tag.casefold():
            continue
        if tag.casefold().startswith("ai:prospective/"):
            continue
        if tag not in out:
            out.append(tag)
    for tag in generated:
        if tag not in out:
            out.append(tag)
    return out


def main() -> int:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    device = choose_device(args.device)

    candidates: List[Tuple[Path, Path, List[str], Optional[str], ET.ElementTree]] = []
    invalid = 0
    missing_media = 0
    skipped_no_tag = 0

    for xmp in iter_sidecars(root, args.recurse):
        tags, caption, tree = read_tags_caption_tree(xmp)
        if tree is None:
            invalid += 1
            continue
        if not has_target_tag(tags, args.low_quality_tag):
            skipped_no_tag += 1
            continue
        media = media_path_for_sidecar(xmp)
        if media is None:
            missing_media += 1
            if args.verbose:
                print(f"MISSING MEDIA for sidecar: {xmp}")
            continue
        candidates.append((xmp, media, tags, caption, tree))
        if args.limit and len(candidates) >= args.limit:
            break

    print("Scan summary")
    print("------------")
    print(f"sidecars_matching_low_quality_tag: {len(candidates)}")
    print(f"invalid_xml: {invalid}")
    print(f"missing_media: {missing_media}")
    print(f"skipped_without_target_tag: {skipped_no_tag}")
    print(f"mode: {'APPLY' if args.apply else 'DRY RUN'}")

    if not candidates:
        return 0

    runner = BlipLargeRunner(args.model, device, args.verbose)

    repaired = 0
    errors = 0
    no_generated_tags = 0

    for xmp, media, existing_tags, old_caption, tree in tqdm(candidates, desc="BLIP-large repair"):
        try:
            image = prepare_image(media, args.max_side, args.heic_fallback)
            model_text = runner.caption(image, args.max_new_tokens)
            generated_tags = pick_tags(model_text, args.max_tags, args.prospective_prefix, args.min_term_len)

            if not generated_tags:
                no_generated_tags += 1
                if args.verbose:
                    print(f"NO TAGS: {xmp}")
                    print(f"  model_text: {model_text}")
                continue

            final_tags = merge_tags(existing_tags, generated_tags, args.low_quality_tag, args.keep_low_quality_tag)

            if args.verbose or not args.apply:
                print(f"{'REPAIR' if args.apply else 'WOULD REPAIR'} {xmp}")
                print(f"  image: {media}")
                print(f"  old_caption: {old_caption}")
                print(f"  blip_large_caption: {model_text}")
                print(f"  new_tags: {', '.join(generated_tags)}")

            if args.apply:
                replace_subject_tags(tree, final_tags)
                if args.update_caption and model_text:
                    upsert_description(tree, model_text)
                tree.write(xmp, encoding="utf-8", xml_declaration=True)
                repaired += 1

            if device == "mps":
                try:
                    torch.mps.empty_cache()
                except Exception:
                    pass

        except Exception as exc:
            errors += 1
            if args.verbose:
                print(f"ERROR {xmp}: {exc}", file=sys.stderr)

    print("\nDone")
    print("----")
    print(f"matched: {len(candidates)}")
    print(f"repaired: {repaired}")
    print(f"no_generated_tags: {no_generated_tags}")
    print(f"errors: {errors}")
    print(f"mode: {'APPLY' if args.apply else 'DRY RUN'}")
    if not args.apply:
        print("\nDry run only. Rerun with --apply to write changes.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
