#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path
from typing import Iterable, List, Optional, Tuple
from xml.etree import ElementTree as ET

NAMESPACES = {
    "x": "adobe:ns:meta/",
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "dc": "http://purl.org/dc/elements/1.1/",
}

for prefix, uri in NAMESPACES.items():
    ET.register_namespace(prefix, uri)

AI_PREFIX = "ai:"

STOPWORDS_FOR_TAG_START = {
    "a", "an", "the", "i", "it", "this", "that", "there", "here"
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Repair XMP sidecars that have captions but no tags by adding one "
            "last-resort ai:caption/... tag. Skips low-quality repeated captions by default."
        )
    )
    p.add_argument("root", help="Folder containing .xmp sidecars.")
    p.add_argument("--recurse", action="store_true", help="Scan recursively.")
    p.add_argument("--apply", action="store_true", help="Actually write changes. Default is dry run.")
    p.add_argument("--max-words", type=int, default=10, help="Maximum words kept from caption for fallback tag.")
    p.add_argument("--max-chars", type=int, default=90, help="Maximum characters kept from caption for fallback tag.")
    p.add_argument("--min-quality", type=float, default=0.55, help="Minimum caption quality score, 0..1.")
    p.add_argument("--allow-low-quality", action="store_true", help="Allow tags even from low-quality/repetitive captions.")
    p.add_argument("--bad-caption-tag", default="", help="Optional fixed tag for low-quality captions, e.g. ai:caption/low-quality. Default skips them.")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def iter_sidecars(root: Path, recurse: bool) -> Iterable[Path]:
    yield from (root.rglob("*.xmp") if recurse else root.glob("*.xmp"))


def read_tags_and_caption(path: Path) -> Tuple[List[str], Optional[str], Optional[ET.ElementTree]]:
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


def words_from_text(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+(?:'[a-z]+)?", text.lower())


def repeated_ngram_score(words: List[str], n: int) -> float:
    if len(words) < n * 2:
        return 0.0
    chunks = [" ".join(words[i:i+n]) for i in range(len(words)-n+1)]
    counts = Counter(chunks)
    most = counts.most_common(1)[0][1] if counts else 0
    return most / max(1, len(chunks))


def caption_quality(caption: str) -> Tuple[float, List[str]]:
    """Return a rough quality score and reasons for rejection.

    This catches common caption-model loops:
      residence residence residence...
      tor tor tor...
      Google Earth - Google Earth - ...
      don't's don't's don't's...
    """
    raw = caption.strip()
    words = words_from_text(raw)
    reasons: List[str] = []

    if not words:
        return 0.0, ["empty"]

    unique_ratio = len(set(words)) / len(words)
    counts = Counter(words)
    top_word, top_count = counts.most_common(1)[0]
    top_ratio = top_count / len(words)

    if len(words) >= 6 and unique_ratio < 0.35:
        reasons.append(f"low unique ratio {unique_ratio:.2f}")
    if len(words) >= 6 and top_count >= 4 and top_ratio >= 0.35:
        reasons.append(f"repeated word {top_word!r} x{top_count}")
    if repeated_ngram_score(words, 2) >= 0.30:
        reasons.append("repeated bigram")
    if repeated_ngram_score(words, 3) >= 0.25:
        reasons.append("repeated trigram")

    # Short fragments repeated many times are a strong sign of decode-loop garbage.
    short_repeats = [w for w, c in counts.items() if len(w) <= 4 and c >= 4]
    if short_repeats:
        reasons.append("short fragment loop: " + ", ".join(short_repeats[:5]))

    # Lots of isolated apostrophe spacing usually means OCR/caption garbage.
    if raw.count(" ' ") >= 3:
        reasons.append("apostrophe fragments")

    # Build score.
    score = 1.0
    score -= max(0.0, 0.45 - unique_ratio)
    score -= max(0.0, top_ratio - 0.25)
    score -= 0.20 * min(2, len(short_repeats))
    if "apostrophe fragments" in reasons:
        score -= 0.20
    if repeated_ngram_score(words, 2) >= 0.30:
        score -= 0.25
    if repeated_ngram_score(words, 3) >= 0.25:
        score -= 0.25

    return max(0.0, min(1.0, score)), reasons


def collapse_repetition(words: List[str]) -> List[str]:
    """Remove immediate repeated words and stop at obvious decode loops."""
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

        # Stop if a short token is clearly looping.
        if len(word) <= 4 and counts[word] >= 4:
            break
        # Stop if any token is clearly looping.
        if counts[word] >= 5:
            break

        out.append(word)
        prev = word

    return out


def normalize_caption_for_tag(caption: str, max_words: int, max_chars: int) -> str:
    s = caption.strip().lower()
    s = re.sub(r"^describe this image.*?:\s*", "", s, flags=re.I)
    s = re.sub(r"https?://\S+", "", s)
    s = re.sub(r"\b(photo|picture|image)\b", "", s)
    s = s.replace("_", " ")
    s = re.sub(r"[^a-z0-9&' /-]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip(" -/")

    words = words_from_text(s)
    words = collapse_repetition(words)

    while words and words[0] in STOPWORDS_FOR_TAG_START:
        words.pop(0)

    if max_words > 0 and len(words) > max_words:
        words = words[:max_words]

    s = " ".join(words)

    if max_chars > 0 and len(s) > max_chars:
        s = s[:max_chars].rsplit(" ", 1)[0].strip(" -/")

    return s


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


def add_subject_tag(tree: ET.ElementTree, tag: str) -> None:
    desc = ensure_description_node(tree)

    for node in list(desc.findall("dc:subject", NAMESPACES)):
        desc.remove(node)
    for node in list(desc.findall("{http://purl.org/dc/elements/1.1/}subject")):
        if node in list(desc):
            desc.remove(node)

    subject = ET.SubElement(desc, f"{{{NAMESPACES['dc']}}}subject")
    bag = ET.SubElement(subject, f"{{{NAMESPACES['rdf']}}}Bag")
    li = ET.SubElement(bag, f"{{{NAMESPACES['rdf']}}}li")
    li.text = tag


def main() -> int:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()

    scanned = 0
    invalid = 0
    captioned_no_tags = 0
    repaired = 0
    skipped_no_caption = 0
    skipped_has_tags = 0
    skipped_low_quality = 0

    for xmp in iter_sidecars(root, args.recurse):
        scanned += 1
        tags, caption, tree = read_tags_and_caption(xmp)

        if tree is None:
            invalid += 1
            if args.verbose:
                print(f"INVALID XML: {xmp}")
            continue

        if tags:
            skipped_has_tags += 1
            continue

        if not caption:
            skipped_no_caption += 1
            continue

        captioned_no_tags += 1
        quality, reasons = caption_quality(caption)

        if quality < args.min_quality and not args.allow_low_quality:
            skipped_low_quality += 1
            fixed_tag = args.bad_caption_tag.strip()
            if fixed_tag:
                tag = fixed_tag if fixed_tag.startswith(AI_PREFIX) else AI_PREFIX + fixed_tag
            else:
                if args.verbose or not args.apply:
                    print(f"SKIP LOW QUALITY q={quality:.2f}: {xmp}")
                    print(f"  reasons: {', '.join(reasons)}")
                    print(f"  caption: {caption}")
                continue
        else:
            clean = normalize_caption_for_tag(caption, args.max_words, args.max_chars)
            if not clean:
                skipped_no_caption += 1
                continue
            tag = AI_PREFIX + "caption/" + clean

        if args.verbose or not args.apply:
            print(f"{'ADD' if args.apply else 'WOULD ADD'} {tag}")
            print(f"  quality: {quality:.2f}" + (f" ({', '.join(reasons)})" if reasons else ""))
            print(f"  {xmp}")
            print(f"  caption: {caption}")

        if args.apply:
            add_subject_tag(tree, tag)
            tree.write(xmp, encoding="utf-8", xml_declaration=True)
            repaired += 1

    print("\nSummary")
    print("-------")
    print(f"sidecars_scanned: {scanned}")
    print(f"invalid_xml: {invalid}")
    print(f"skipped_has_tags: {skipped_has_tags}")
    print(f"skipped_no_caption: {skipped_no_caption}")
    print(f"captioned_without_tags: {captioned_no_tags}")
    print(f"skipped_low_quality: {skipped_low_quality}")
    print(f"repaired: {repaired}")
    print(f"mode: {'APPLY' if args.apply else 'DRY RUN'}")

    if not args.apply:
        print("\nDry run only. Rerun with --apply to write tags.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
