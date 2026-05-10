#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from xml.etree import ElementTree as ET

NAMESPACES = {
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "dc": "http://purl.org/dc/elements/1.1/",
}
AI_PREFIX = "ai:"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarize tags and captions found in XMP sidecars.")
    p.add_argument("root", help="Root folder containing XMP sidecars.")
    p.add_argument("--recurse", action="store_true", help="Scan sidecars recursively.")
    p.add_argument("--taxonomy-map", default=None, help="Optional labels_taxonomy_map.csv for validation.")
    p.add_argument("--top", type=int, default=30, help="Number of top tags to print.")
    p.add_argument("--csv-prefix", default=None, help="Write detailed CSV reports using this filename prefix.")
    p.add_argument("--json", dest="json_path", default=None, help="Write summary JSON to this path.")
    return p.parse_args()


def iter_sidecars(root: Path, recurse: bool) -> Iterable[Path]:
    if recurse:
        yield from root.rglob("*.xmp")
    else:
        yield from root.glob("*.xmp")


def read_xmp(path: Path) -> Tuple[List[str], Optional[str], bool]:
    try:
        tree = ET.parse(path)
        root = tree.getroot()
    except ET.ParseError:
        return [], None, False

    tags: List[str] = []
    for li in root.findall(".//dc:subject/rdf:Bag/rdf:li", NAMESPACES):
        if li.text and li.text.strip():
            tags.append(li.text.strip())
    if not tags:
        for li in root.findall(".//{*}subject//{*}li"):
            if li.text and li.text.strip():
                tags.append(li.text.strip())

    description: Optional[str] = None
    desc_node = root.find(".//dc:description/rdf:Alt/rdf:li", NAMESPACES)
    if desc_node is not None and desc_node.text:
        description = desc_node.text.strip()
    if description is None:
        desc_node = root.find(".//{*}description//{*}li")
        if desc_node is not None and desc_node.text:
            description = desc_node.text.strip()

    return tags, description, True


def normalize_tag(tag: str) -> str:
    return tag.strip()


def strip_ai_prefix(tag: str) -> str:
    return tag[len(AI_PREFIX):] if tag.startswith(AI_PREFIX) else tag


def read_taxonomy_paths(path: Optional[Path]) -> set[str]:
    if path is None or not path.exists():
        return set()
    out: set[str] = set()
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tags = (row.get("tags") or "").strip()
            for tag in tags.split("|"):
                tag = tag.strip()
                if tag:
                    out.add(tag)
    return out


def hierarchy_levels(tag: str) -> List[str]:
    parts = [p for p in tag.split("/") if p]
    return ["/".join(parts[:i]) for i in range(1, len(parts) + 1)]


def write_csv(prefix: Path, tag_counter: Counter[str], hierarchy_counter: Counter[str], per_sidecar: List[dict]) -> None:
    with prefix.with_name(prefix.name + "_tags.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tag", "count"])
        for tag, count in tag_counter.most_common():
            w.writerow([tag, count])

    with prefix.with_name(prefix.name + "_hierarchy.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["hierarchy_path", "count"])
        for tag, count in hierarchy_counter.most_common():
            w.writerow([tag, count])

    with prefix.with_name(prefix.name + "_sidecars.csv").open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["sidecar", "valid_xml", "caption_present", "tag_count", "ai_tag_count", "human_tag_count", "tags"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in per_sidecar:
            w.writerow(row)


def main() -> int:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    taxonomy_paths = read_taxonomy_paths(Path(args.taxonomy_map).expanduser().resolve() if args.taxonomy_map else None)

    sidecars = list(iter_sidecars(root, args.recurse))
    tag_counter: Counter[str] = Counter()
    ai_tag_counter: Counter[str] = Counter()
    human_tag_counter: Counter[str] = Counter()
    hierarchy_counter: Counter[str] = Counter()
    unmapped_ai_tags: Counter[str] = Counter()

    valid_xml = 0
    invalid_xml = 0
    captioned = 0
    sidecars_with_tags = 0
    sidecars_without_tags = 0
    total_assignments = 0
    per_sidecar: List[dict] = []

    for sidecar in sidecars:
        tags, description, ok = read_xmp(sidecar)
        if ok:
            valid_xml += 1
        else:
            invalid_xml += 1

        if description:
            captioned += 1

        tags = [normalize_tag(t) for t in tags if normalize_tag(t)]
        ai_tags = [t for t in tags if t.startswith(AI_PREFIX)]
        human_tags = [t for t in tags if not t.startswith(AI_PREFIX)]

        if tags:
            sidecars_with_tags += 1
        else:
            sidecars_without_tags += 1

        total_assignments += len(tags)
        tag_counter.update(tags)
        ai_tag_counter.update(ai_tags)
        human_tag_counter.update(human_tags)

        for tag in tags:
            raw = strip_ai_prefix(tag)
            if "/" in raw:
                hierarchy_counter.update(hierarchy_levels(raw))
            if tag.startswith(AI_PREFIX) and taxonomy_paths and raw not in taxonomy_paths:
                unmapped_ai_tags[raw] += 1

        per_sidecar.append({
            "sidecar": str(sidecar),
            "valid_xml": ok,
            "caption_present": bool(description),
            "tag_count": len(tags),
            "ai_tag_count": len(ai_tags),
            "human_tag_count": len(human_tags),
            "tags": "|".join(tags),
        })

    summary = {
        "root": str(root),
        "sidecars_scanned": len(sidecars),
        "valid_xml": valid_xml,
        "invalid_xml": invalid_xml,
        "captioned_sidecars": captioned,
        "uncaptioned_sidecars": len(sidecars) - captioned,
        "sidecars_with_tags": sidecars_with_tags,
        "sidecars_without_tags": sidecars_without_tags,
        "total_tag_assignments": total_assignments,
        "unique_tags": len(tag_counter),
        "unique_ai_tags": len(ai_tag_counter),
        "unique_human_tags": len(human_tag_counter),
        "unmapped_ai_tag_count": sum(unmapped_ai_tags.values()),
        "unique_unmapped_ai_tags": len(unmapped_ai_tags),
    }

    print("\nSummary")
    print("-------")
    for key, value in summary.items():
        if key != "root":
            print(f"{key}: {value}")

    print(f"\nTop {args.top} tags")
    print("-" * (4 + len(str(args.top)) + 5))
    for tag, count in tag_counter.most_common(args.top):
        print(f"{count:>7}  {tag}")

    if hierarchy_counter:
        print(f"\nTop {args.top} hierarchy paths")
        print("-" * (4 + len(str(args.top)) + 16))
        for tag, count in hierarchy_counter.most_common(args.top):
            print(f"{count:>7}  {tag}")

    if taxonomy_paths:
        print(f"\nAI tags not present as exact hierarchy paths in taxonomy ({len(unmapped_ai_tags)} unique)")
        print("-" * 64)
        for tag, count in unmapped_ai_tags.most_common(args.top):
            print(f"{count:>7}  {tag}")

    if args.csv_prefix:
        write_csv(Path(args.csv_prefix), tag_counter, hierarchy_counter, per_sidecar)
        print(f"\nWrote CSV reports with prefix: {args.csv_prefix}")

    if args.json_path:
        payload = {
            "summary": summary,
            "top_tags": tag_counter.most_common(),
            "top_ai_tags": ai_tag_counter.most_common(),
            "top_human_tags": human_tag_counter.most_common(),
            "hierarchy_paths": hierarchy_counter.most_common(),
            "unmapped_ai_tags": unmapped_ai_tags.most_common(),
        }
        Path(args.json_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote JSON summary: {args.json_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
