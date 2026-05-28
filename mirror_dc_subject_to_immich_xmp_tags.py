#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple
from xml.etree import ElementTree as ET

NAMESPACES = {
    "x": "adobe:ns:meta/",
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "dc": "http://purl.org/dc/elements/1.1/",
    "digiKam": "http://www.digikam.org/ns/1.0/",
    "lr": "http://ns.adobe.com/lightroom/1.0/",
}

for prefix, uri in NAMESPACES.items():
    ET.register_namespace(prefix, uri)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Mirror XMP dc:subject tags into digiKam:TagsList and lr:HierarchicalSubject for Immich."
    )
    p.add_argument("root", help="Folder containing .xmp sidecars.")
    p.add_argument("--recurse", action="store_true")
    p.add_argument("--apply", action="store_true", help="Write changes. Default is dry run.")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--write-lightroom", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--strip-ai-prefix", action="store_true", help="Write values without leading ai:. Default preserves tags exactly.")
    p.add_argument("--only-missing", action="store_true", help="Only fill empty digiKam/lr fields.")
    p.add_argument("--limit", type=int, default=0)
    return p.parse_args()


def iter_sidecars(root: Path, recurse: bool) -> Iterable[Path]:
    yield from (root.rglob("*.xmp") if recurse else root.glob("*.xmp"))


def dedupe(items: Sequence[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for item in items:
        item = (item or "").strip()
        if not item:
            continue
        key = item.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def read_dc_subject(root: ET.Element) -> List[str]:
    tags: List[str] = []
    for li in root.findall(".//dc:subject/rdf:Bag/rdf:li", NAMESPACES):
        if li.text and li.text.strip():
            tags.append(li.text.strip())

    if not tags:
        for subject in root.findall(".//{http://purl.org/dc/elements/1.1/}subject"):
            for li in subject.findall(".//{*}li"):
                if li.text and li.text.strip():
                    tags.append(li.text.strip())

    return dedupe(tags)


def read_existing_immich_fields(root: ET.Element) -> Tuple[List[str], List[str]]:
    digikam: List[str] = []
    lightroom: List[str] = []

    for li in root.findall(".//digiKam:TagsList/rdf:Seq/rdf:li", NAMESPACES):
        if li.text and li.text.strip():
            digikam.append(li.text.strip())
    if not digikam:
        for node in root.findall(".//{http://www.digikam.org/ns/1.0/}TagsList"):
            for li in node.findall(".//{*}li"):
                if li.text and li.text.strip():
                    digikam.append(li.text.strip())

    for li in root.findall(".//lr:HierarchicalSubject/rdf:Bag/rdf:li", NAMESPACES):
        if li.text and li.text.strip():
            lightroom.append(li.text.strip())
    if not lightroom:
        for node in root.findall(".//{http://ns.adobe.com/lightroom/1.0/}HierarchicalSubject"):
            for li in node.findall(".//{*}li"):
                if li.text and li.text.strip():
                    lightroom.append(li.text.strip())

    return dedupe(digikam), dedupe(lightroom)


def ensure_primary_description(root: ET.Element) -> ET.Element:
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


def remove_child(desc: ET.Element, namespace: str, local: str) -> None:
    tag = f"{{{namespace}}}{local}"
    for node in list(desc):
        if node.tag == tag:
            desc.remove(node)


def add_seq(parent: ET.Element, tags: Sequence[str]) -> None:
    seq = ET.SubElement(parent, f"{{{NAMESPACES['rdf']}}}Seq")
    for tag in tags:
        li = ET.SubElement(seq, f"{{{NAMESPACES['rdf']}}}li")
        li.text = tag


def add_bag(parent: ET.Element, tags: Sequence[str]) -> None:
    bag = ET.SubElement(parent, f"{{{NAMESPACES['rdf']}}}Bag")
    for tag in tags:
        li = ET.SubElement(bag, f"{{{NAMESPACES['rdf']}}}li")
        li.text = tag


def write_immich_fields(tree: ET.ElementTree, tags: Sequence[str], write_lightroom: bool) -> None:
    root = tree.getroot()
    desc = ensure_primary_description(root)

    remove_child(desc, NAMESPACES["digiKam"], "TagsList")
    dk = ET.SubElement(desc, f"{{{NAMESPACES['digiKam']}}}TagsList")
    add_seq(dk, tags)

    if write_lightroom:
        remove_child(desc, NAMESPACES["lr"], "HierarchicalSubject")
        lr = ET.SubElement(desc, f"{{{NAMESPACES['lr']}}}HierarchicalSubject")
        add_bag(lr, tags)


def convert_tags(tags: Sequence[str], strip_ai_prefix: bool) -> List[str]:
    out = []
    for tag in tags:
        tag = tag.strip()
        if strip_ai_prefix and tag.startswith("ai:"):
            tag = tag[3:]
        out.append(tag)
    return dedupe(out)


def main() -> int:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()

    scanned = invalid = no_dc = matching = would_update = updated = 0

    for xmp in iter_sidecars(root, args.recurse):
        scanned += 1
        if args.limit and scanned > args.limit:
            break

        try:
            tree = ET.parse(xmp)
        except ET.ParseError:
            invalid += 1
            if args.verbose:
                print(f"INVALID XML: {xmp}")
            continue

        root_el = tree.getroot()
        dc_tags = read_dc_subject(root_el)
        if not dc_tags:
            no_dc += 1
            continue

        desired = convert_tags(dc_tags, args.strip_ai_prefix)
        dk_existing, lr_existing = read_existing_immich_fields(root_el)

        if args.only_missing and (dk_existing or (args.write_lightroom and lr_existing)):
            matching += 1
            continue

        lr_ok = (not args.write_lightroom) or (lr_existing == desired)
        if dk_existing == desired and lr_ok:
            matching += 1
            continue

        would_update += 1
        if args.verbose or not args.apply:
            print(f"{'UPDATE' if args.apply else 'WOULD UPDATE'} {xmp}")
            print(f"  dc_subject_count: {len(dc_tags)}")
            print(f"  existing_digikam_count: {len(dk_existing)}")
            print(f"  existing_lightroom_count: {len(lr_existing)}")
            print(f"  tags: {' | '.join(desired[:8])}" + (" ..." if len(desired) > 8 else ""))

        if args.apply:
            write_immich_fields(tree, desired, args.write_lightroom)
            tree.write(xmp, encoding="utf-8", xml_declaration=True)
            updated += 1

    print("\nSummary")
    print("-------")
    print(f"sidecars_scanned: {scanned}")
    print(f"invalid_xml: {invalid}")
    print(f"sidecars_without_dc_subject: {no_dc}")
    print(f"already_matching_or_skipped: {matching}")
    print(f"would_update: {would_update}")
    print(f"updated: {updated}")
    print(f"mode: {'APPLY' if args.apply else 'DRY RUN'}")
    if not args.apply:
        print("\nDry run only. Rerun with --apply to write digiKam/lr fields.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
