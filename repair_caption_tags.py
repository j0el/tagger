#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Repair XMP sidecars that have captions but no tags by adding one "
            "last-resort ai:caption/... tag."
        )
    )
    p.add_argument("root", help="Folder containing .xmp sidecars.")
    p.add_argument("--recurse", action="store_true", help="Scan recursively.")
    p.add_argument("--apply", action="store_true", help="Actually write changes. Default is dry run.")
    p.add_argument("--max-words", type=int, default=10, help="Maximum words kept from caption for fallback tag.")
    p.add_argument("--max-chars", type=int, default=90, help="Maximum characters kept from caption for fallback tag.")
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


def normalize_caption_for_tag(caption: str, max_words: int, max_chars: int) -> str:
    s = caption.strip().lower()
    s = re.sub(r"^describe this image.*?:\s*", "", s, flags=re.I)
    s = re.sub(r"\b(photo|picture|image)\b", "", s)
    s = s.replace("_", " ")
    s = re.sub(r"[^a-z0-9&' /-]+", "", s)
    s = re.sub(r"\s+", " ", s).strip(" -/")

    words = s.split()
    if max_words > 0 and len(words) > max_words:
        s = " ".join(words[:max_words])
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
        clean = normalize_caption_for_tag(caption, args.max_words, args.max_chars)
        if not clean:
            skipped_no_caption += 1
            continue

        tag = AI_PREFIX + "caption/" + clean

        if args.verbose or not args.apply:
            print(f"{'ADD' if args.apply else 'WOULD ADD'} {tag}")
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
    print(f"repaired: {repaired}")
    print(f"mode: {'APPLY' if args.apply else 'DRY RUN'}")

    if not args.apply:
        print("\nDry run only. Rerun with --apply to write tags.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
