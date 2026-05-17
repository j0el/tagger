#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple
from xml.etree import ElementTree as ET

NAMESPACES = {
    "x": "adobe:ns:meta/",
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "dc": "http://purl.org/dc/elements/1.1/",
    "xmp": "http://ns.adobe.com/xap/1.0/",
    "digiKam": "http://www.digikam.org/ns/1.0/",
    "lr": "http://ns.adobe.com/lightroom/1.0/",
}

for prefix, uri in NAMESPACES.items():
    try:
        ET.register_namespace(prefix, uri)
    except ValueError:
        pass


@dataclass
class CleanupResult:
    path: Path
    changed: bool
    removed: List[str]
    error: str = ""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Remove unwanted tags such as takeout and immich-go from XMP sidecars. "
            "Dry-run by default; add --apply to write changes."
        )
    )
    p.add_argument("root", help="Folder containing .xmp sidecars, or one .xmp file.")
    p.add_argument("--recurse", action="store_true", help="Scan subfolders recursively.")
    p.add_argument(
        "--terms",
        nargs="+",
        default=["takeout", "immich-go", "immich go", "immichgo"],
        help="Tag text to remove. Default: takeout immich-go immich go immichgo",
    )
    p.add_argument(
        "--exact",
        action="store_true",
        help=(
            "Only remove exact tag matches or exact path-segment matches. "
            "Default removes tags containing one of the terms, case-insensitive."
        ),
    )
    p.add_argument("--apply", action="store_true", help="Actually edit sidecars. Without this, no files are changed.")
    p.add_argument("--no-backup", action="store_true", help="Do not create timestamped .bak backups before editing.")
    p.add_argument("--csv", default=None, help="Optional CSV report path.")
    p.add_argument("--quiet", action="store_true", help="Only print summary lines.")
    return p.parse_args()


def iter_sidecars(root: Path, recurse: bool) -> Iterable[Path]:
    if root.is_file():
        if root.suffix.lower() == ".xmp":
            yield root
        return
    yield from (root.rglob("*.xmp") if recurse else root.glob("*.xmp"))


def local_name(tag: str) -> str:
    if tag.startswith("{"):
        return tag.rsplit("}", 1)[-1]
    if ":" in tag:
        return tag.rsplit(":", 1)[-1]
    return tag


def normalize_for_match(value: str) -> str:
    value = value.casefold().strip()
    value = value.replace("_", " ")
    value = re.sub(r"\s+", " ", value)
    return value


def split_path_segments(tag: str) -> List[str]:
    return [seg.strip() for seg in re.split(r"[/\\|>]", tag) if seg.strip()]


def should_remove(tag: str, terms: Sequence[str], exact: bool) -> bool:
    raw = tag.strip()
    if not raw:
        return False

    tag_norm = normalize_for_match(raw)
    terms_norm = [normalize_for_match(t) for t in terms if normalize_for_match(t)]

    if exact:
        segments = [normalize_for_match(s) for s in split_path_segments(raw)]
        return any(tag_norm == term or term in segments for term in terms_norm)

    # Default: case-insensitive contains. Also compare hyphen/space variants so
    # immich-go, immich go, and immichgo all match each other.
    tag_loose = tag_norm.replace("-", " ").replace(" ", "")
    for term in terms_norm:
        term_loose = term.replace("-", " ").replace(" ", "")
        if term in tag_norm or term_loose in tag_loose:
            return True
    return False


def tag_container_elements(root: ET.Element) -> List[ET.Element]:
    containers: List[ET.Element] = []

    # Standard XMP keyword tags used by this project: dc:subject/rdf:Bag/rdf:li.
    for elem in root.iter():
        if local_name(elem.tag).lower() == "subject":
            containers.append(elem)

    # Common digiKam sidecar tag list location: XMP-digiKam:TagsList.
    for elem in root.iter():
        if local_name(elem.tag).lower() == "tagslist":
            containers.append(elem)

    # De-duplicate while preserving order.
    seen = set()
    out = []
    for elem in containers:
        key = id(elem)
        if key not in seen:
            out.append(elem)
            seen.add(key)
    return out


def remove_matching_li(container: ET.Element, terms: Sequence[str], exact: bool) -> Tuple[bool, List[str]]:
    removed: List[str] = []
    changed = False

    for parent in list(container.iter()):
        for child in list(parent):
            if local_name(child.tag).lower() != "li":
                continue
            text = (child.text or "").strip()
            if should_remove(text, terms, exact):
                parent.remove(child)
                removed.append(text)
                changed = True

    return changed, removed


def split_tag_list_text(value: str) -> Tuple[List[str], str]:
    # exiftool often prints TagsList as comma-separated, but tolerate semicolon/pipe too.
    if "|" in value:
        delimiter = "|"
    elif ";" in value:
        delimiter = ";"
    else:
        delimiter = ","
    parts = [p.strip() for p in value.split(delimiter) if p.strip()]
    return parts, delimiter


def rewrite_text_tag_list(elem: ET.Element, terms: Sequence[str], exact: bool) -> Tuple[bool, List[str]]:
    text = elem.text or ""
    if not text.strip():
        return False, []

    parts, delimiter = split_tag_list_text(text)
    if len(parts) <= 1:
        single = text.strip()
        if should_remove(single, terms, exact):
            elem.text = ""
            return True, [single]
        return False, []

    kept = []
    removed = []
    for part in parts:
        if should_remove(part, terms, exact):
            removed.append(part)
        else:
            kept.append(part)

    if removed:
        elem.text = f"{delimiter} ".join(kept)
        return True, removed
    return False, []


def rewrite_tagslist_attributes(root: ET.Element, terms: Sequence[str], exact: bool) -> Tuple[bool, List[str]]:
    changed = False
    removed_all: List[str] = []

    for elem in root.iter():
        for attr, value in list(elem.attrib.items()):
            if local_name(attr).lower() != "tagslist":
                continue
            parts, delimiter = split_tag_list_text(value)
            kept = []
            removed = []
            for part in parts:
                if should_remove(part, terms, exact):
                    removed.append(part)
                else:
                    kept.append(part)
            if removed:
                if kept:
                    elem.set(attr, f"{delimiter} ".join(kept))
                else:
                    del elem.attrib[attr]
                removed_all.extend(removed)
                changed = True

    return changed, removed_all


def cleanup_sidecar(path: Path, terms: Sequence[str], exact: bool) -> CleanupResult:
    try:
        tree = ET.parse(path)
    except ET.ParseError as exc:
        return CleanupResult(path=path, changed=False, removed=[], error=f"XML parse error: {exc}")
    except OSError as exc:
        return CleanupResult(path=path, changed=False, removed=[], error=str(exc))

    root = tree.getroot()
    changed = False
    removed_all: List[str] = []

    for container in tag_container_elements(root):
        c_changed, removed = remove_matching_li(container, terms, exact)
        if c_changed:
            changed = True
            removed_all.extend(removed)

        # Covers direct text forms such as <digiKam:TagsList>foo, bar</digiKam:TagsList>.
        if local_name(container.tag).lower() == "tagslist":
            t_changed, t_removed = rewrite_text_tag_list(container, terms, exact)
            if t_changed:
                changed = True
                removed_all.extend(t_removed)

    a_changed, a_removed = rewrite_tagslist_attributes(root, terms, exact)
    if a_changed:
        changed = True
        removed_all.extend(a_removed)

    return CleanupResult(path=path, changed=changed, removed=dedupe_preserve_order(removed_all), error="")


def dedupe_preserve_order(values: Sequence[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for value in values:
        key = value.casefold()
        if key not in seen:
            out.append(value)
            seen.add(key)
    return out


def backup_path(path: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return path.with_name(f"{path.name}.{stamp}.bak")


def write_tree(tree_path: Path, terms: Sequence[str], exact: bool, make_backup: bool) -> CleanupResult:
    # Re-parse and clean immediately before writing so dry-run and apply use the same logic.
    try:
        tree = ET.parse(tree_path)
    except ET.ParseError as exc:
        return CleanupResult(tree_path, False, [], f"XML parse error: {exc}")
    except OSError as exc:
        return CleanupResult(tree_path, False, [], str(exc))

    root = tree.getroot()
    changed = False
    removed_all: List[str] = []

    for container in tag_container_elements(root):
        c_changed, removed = remove_matching_li(container, terms, exact)
        if c_changed:
            changed = True
            removed_all.extend(removed)

        if local_name(container.tag).lower() == "tagslist":
            t_changed, t_removed = rewrite_text_tag_list(container, terms, exact)
            if t_changed:
                changed = True
                removed_all.extend(t_removed)

    a_changed, a_removed = rewrite_tagslist_attributes(root, terms, exact)
    if a_changed:
        changed = True
        removed_all.extend(a_removed)

    removed_all = dedupe_preserve_order(removed_all)

    if changed:
        if make_backup:
            shutil.copy2(tree_path, backup_path(tree_path))
        tmp = tree_path.with_name(tree_path.name + ".tmp")
        tree.write(tmp, encoding="utf-8", xml_declaration=True)
        tmp.replace(tree_path)

    return CleanupResult(tree_path, changed, removed_all, "")


def write_csv_report(path: Path, results: Sequence[CleanupResult]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sidecar", "changed", "removed_tags", "error"])
        for r in results:
            w.writerow([str(r.path), r.changed, "|".join(r.removed), r.error])


def main() -> int:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()

    if not root.exists():
        print(f"Not found: {root}", file=sys.stderr)
        return 2

    sidecars = list(iter_sidecars(root, args.recurse))
    if not sidecars:
        print(f"No .xmp sidecars found under: {root}", file=sys.stderr)
        return 1

    dry_results = [cleanup_sidecar(path, args.terms, args.exact) for path in sidecars]
    to_change = [r.path for r in dry_results if r.changed and not r.error]

    final_results: List[CleanupResult]
    if args.apply:
        final_results = [
            write_tree(path, args.terms, args.exact, make_backup=not args.no_backup)
            if path in set(to_change)
            else next(r for r in dry_results if r.path == path)
            for path in sidecars
        ]
    else:
        final_results = dry_results

    changed_results = [r for r in final_results if r.changed]
    error_results = [r for r in final_results if r.error]
    removed_count = sum(len(r.removed) for r in changed_results)

    mode = "APPLY" if args.apply else "DRY RUN"
    print(f"\nMode: {mode}")
    print(f"Root: {root}")
    print(f"Sidecars scanned: {len(sidecars)}")
    print(f"Sidecars with matching tags: {len(changed_results)}")
    print(f"Unique removed tag entries across changed sidecars: {removed_count}")
    print(f"Errors: {len(error_results)}")

    if not args.quiet:
        for r in changed_results[:100]:
            print(f"\n{r.path}")
            for tag in r.removed:
                print(f"  remove: {tag}")
        if len(changed_results) > 100:
            print(f"\n... {len(changed_results) - 100} more changed sidecars not shown")

        for r in error_results[:20]:
            print(f"\nERROR {r.path}: {r.error}", file=sys.stderr)
        if len(error_results) > 20:
            print(f"\n... {len(error_results) - 20} more errors not shown", file=sys.stderr)

    if args.csv:
        write_csv_report(Path(args.csv).expanduser().resolve(), final_results)
        print(f"\nWrote CSV report: {args.csv}")

    if not args.apply and changed_results:
        print("\nDry run only. Re-run with --apply to edit files.")
        print("Backups are created by default. Use --no-backup only if you already have a backup.")

    return 0 if not error_results else 1


if __name__ == "__main__":
    raise SystemExit(main())
