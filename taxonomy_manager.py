#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import shutil
from collections import Counter, OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

DEFAULT_LABELS_FILE = "labels_curated_hierarchical.txt"
DEFAULT_TAXONOMY_FILE = "labels_taxonomy_map.csv"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Maintain curated labels and their hierarchy mappings.")
    p.add_argument("--labels-file", default=DEFAULT_LABELS_FILE)
    p.add_argument("--taxonomy-map", default=DEFAULT_TAXONOMY_FILE)
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("show", help="Show one label and its mapped hierarchy paths.")
    s.add_argument("label")

    s = sub.add_parser("add", help="Add a label to both files.")
    s.add_argument("label")
    s.add_argument("--tags", required=True, help="One or more hierarchy paths separated by |.")
    s.add_argument("--section", default="Prospective", help="Section heading in labels file for a new label.")

    s = sub.add_parser("remove", help="Remove a label from both files.")
    s.add_argument("label")

    s = sub.add_parser("rename", help="Rename a label in both files while preserving its hierarchy paths.")
    s.add_argument("old_label")
    s.add_argument("new_label")

    s = sub.add_parser("set-tags", help="Replace all hierarchy paths for a label.")
    s.add_argument("label")
    s.add_argument("--tags", required=True, help="One or more hierarchy paths separated by |.")

    s = sub.add_parser("add-path", help="Add one hierarchy path to an existing label.")
    s.add_argument("label")
    s.add_argument("--tag", required=True)

    s = sub.add_parser("remove-path", help="Remove one hierarchy path from an existing label.")
    s.add_argument("label")
    s.add_argument("--tag", required=True)

    sub.add_parser("audit", help="Check for duplicates and mismatches.")
    return p.parse_args()


def norm(s: str) -> str:
    return s.strip().casefold()


def clean_label(s: str) -> str:
    return " ".join(s.strip().split())


def split_tags(value: str) -> List[str]:
    out: List[str] = []
    seen = set()
    for tag in value.split("|"):
        tag = "/".join(part.strip() for part in tag.strip().split("/") if part.strip())
        if tag and tag.casefold() not in seen:
            out.append(tag)
            seen.add(tag.casefold())
    return out


def read_taxonomy(path: Path) -> "OrderedDict[str, List[str]]":
    rows: "OrderedDict[str, List[str]]" = OrderedDict()
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != ["label", "tags"]:
            raise ValueError(f"Expected CSV header label,tags in {path}")
        for row in reader:
            label = clean_label(row.get("label", ""))
            if not label:
                continue
            rows[label] = split_tags(row.get("tags", ""))
    return rows


def read_taxonomy_raw(path: Path) -> List[Tuple[str, List[str]]]:
    rows: List[Tuple[str, List[str]]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != ["label", "tags"]:
            raise ValueError(f"Expected CSV header label,tags in {path}")
        for row in reader:
            label = clean_label(row.get("label", ""))
            if not label:
                continue
            raw_tags = ["/".join(part.strip() for part in tag.strip().split("/") if part.strip())
                        for tag in (row.get("tags", "") or "").split("|")
                        if tag.strip()]
            rows.append((label, raw_tags))
    return rows


def write_taxonomy(path: Path, rows: "OrderedDict[str, List[str]]") -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label", "tags"])
        for label, tags in rows.items():
            w.writerow([label, "|".join(tags)])


def read_label_lines(path: Path) -> List[str]:
    if not path.exists():
        return []
    return path.read_text(encoding="utf-8").splitlines()


def labels_from_lines(lines: Iterable[str]) -> List[str]:
    labels = []
    for line in lines:
        s = clean_label(line)
        if s and not s.startswith("#"):
            labels.append(s)
    return labels


def backup(path: Path) -> None:
    if path.exists():
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        shutil.copy2(path, path.with_name(path.name + f".{stamp}.bak"))


def find_label_key(rows: "OrderedDict[str, List[str]]", label: str) -> Optional[str]:
    target = norm(label)
    for key in rows:
        if norm(key) == target:
            return key
    return None


def label_exists_in_lines(lines: List[str], label: str) -> bool:
    target = norm(label)
    return any(norm(line) == target for line in labels_from_lines(lines))


def add_label_to_section(lines: List[str], label: str, section: str) -> List[str]:
    heading = f"# --- {section} ---"
    if label_exists_in_lines(lines, label):
        return lines

    if heading in lines:
        idx = lines.index(heading) + 1
        while idx < len(lines) and (not lines[idx].startswith("# --- ")):
            idx += 1
        insert_at = idx
        if insert_at > 0 and lines[insert_at - 1] == "":
            insert_at -= 1
        return lines[:insert_at] + [label] + lines[insert_at:]

    if lines and lines[-1] != "":
        lines = lines + [""]
    return lines + [heading, label]


def remove_label_from_lines(lines: List[str], label: str) -> List[str]:
    target = norm(label)
    return [line for line in lines if norm(line) != target]


def rename_label_in_lines(lines: List[str], old_label: str, new_label: str) -> List[str]:
    target = norm(old_label)
    out = []
    for line in lines:
        out.append(new_label if norm(line) == target else line)
    return out


def write_label_lines(path: Path, lines: List[str]) -> None:
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def require_label(rows: "OrderedDict[str, List[str]]", label: str) -> str:
    key = find_label_key(rows, label)
    if key is None:
        raise SystemExit(f"Label not found in taxonomy: {label}")
    return key


def save_both(labels_path: Path, taxonomy_path: Path, lines: List[str], rows: "OrderedDict[str, List[str]]") -> None:
    backup(labels_path)
    backup(taxonomy_path)
    write_label_lines(labels_path, lines)
    write_taxonomy(taxonomy_path, rows)


def audit(lines: List[str], rows: "OrderedDict[str, List[str]]", raw_rows: List[Tuple[str, List[str]]]) -> int:
    labels = labels_from_lines(lines)
    line_counts = Counter(norm(x) for x in labels)
    tax_counts = Counter(norm(label) for label, _tags in raw_rows)

    duplicate_labels = sorted([label for label, count in line_counts.items() if count > 1])
    duplicate_taxonomy_labels = sorted([label for label, count in tax_counts.items() if count > 1])
    labels_only = sorted(set(line_counts) - set(tax_counts))
    taxonomy_only = sorted(set(tax_counts) - set(line_counts))
    empty_mappings = [label for label, tags in rows.items() if not tags]
    duplicate_paths = {
        label: [tag for tag, count in Counter(t.casefold() for t in tags).items() if count > 1]
        for label, tags in raw_rows
    }
    duplicate_paths = {label: tags for label, tags in duplicate_paths.items() if tags}

    print(f"labels_file entries: {len(labels)}")
    print(f"taxonomy rows: {len(raw_rows)}")
    print(f"unique taxonomy labels: {len(rows)}")
    print(f"duplicate labels in labels file: {len(duplicate_labels)}")
    print(f"duplicate labels in taxonomy CSV: {len(duplicate_taxonomy_labels)}")
    print(f"labels with no taxonomy row: {len(labels_only)}")
    print(f"taxonomy rows missing from labels file: {len(taxonomy_only)}")
    print(f"empty taxonomy mappings: {len(empty_mappings)}")
    print(f"labels with duplicate hierarchy paths: {len(duplicate_paths)}")

    if duplicate_labels:
        print("\nDuplicate labels in labels file:")
        for item in duplicate_labels:
            print(f"  {item}")
    if duplicate_taxonomy_labels:
        print("\nDuplicate labels in taxonomy CSV:")
        for item in duplicate_taxonomy_labels:
            print(f"  {item}")
    if labels_only:
        print("\nLabels with no taxonomy row:")
        for item in labels_only:
            print(f"  {item}")
    if taxonomy_only:
        print("\nTaxonomy rows missing from labels file:")
        for item in taxonomy_only:
            print(f"  {item}")
    if empty_mappings:
        print("\nLabels with empty taxonomy mapping:")
        for item in empty_mappings:
            print(f"  {item}")
    if duplicate_paths:
        print("\nLabels with duplicate hierarchy paths:")
        for label, tags in duplicate_paths.items():
            print(f"  {label}: {', '.join(tags)}")

    return 1 if any([duplicate_labels, duplicate_taxonomy_labels, labels_only, taxonomy_only, empty_mappings, duplicate_paths]) else 0


def main() -> int:
    args = parse_args()
    labels_path = Path(args.labels_file)
    taxonomy_path = Path(args.taxonomy_map)
    lines = read_label_lines(labels_path)
    rows = read_taxonomy(taxonomy_path)
    raw_rows = read_taxonomy_raw(taxonomy_path)

    if args.command == "show":
        key = find_label_key(rows, args.label)
        in_labels = label_exists_in_lines(lines, args.label)
        if key is None and not in_labels:
            print(f"Not found: {args.label}")
            return 1
        print(f"label: {key or clean_label(args.label)}")
        print(f"in labels file: {in_labels}")
        print(f"taxonomy paths: {' | '.join(rows.get(key, [])) if key else '(none)'}")
        return 0

    if args.command == "audit":
        return audit(lines, rows, raw_rows)

    if args.command == "add":
        label = clean_label(args.label)
        if find_label_key(rows, label) is not None or label_exists_in_lines(lines, label):
            raise SystemExit(f"Label already exists: {label}")
        rows[label] = split_tags(args.tags)
        if not rows[label]:
            raise SystemExit("At least one hierarchy path is required.")
        lines = add_label_to_section(lines, label, args.section)
        save_both(labels_path, taxonomy_path, lines, rows)
        print(f"Added: {label} -> {' | '.join(rows[label])}")
        return 0

    if args.command == "remove":
        key = find_label_key(rows, args.label)
        if key is None and not label_exists_in_lines(lines, args.label):
            raise SystemExit(f"Label not found: {args.label}")
        if key is not None:
            del rows[key]
        lines = remove_label_from_lines(lines, args.label)
        save_both(labels_path, taxonomy_path, lines, rows)
        print(f"Removed: {args.label}")
        return 0

    if args.command == "rename":
        old_key = require_label(rows, args.old_label)
        new_label = clean_label(args.new_label)
        if find_label_key(rows, new_label) is not None:
            raise SystemExit(f"New label already exists: {new_label}")
        new_rows: "OrderedDict[str, List[str]]" = OrderedDict()
        for label, tags in rows.items():
            new_rows[new_label if label == old_key else label] = tags
        lines = rename_label_in_lines(lines, old_key, new_label)
        save_both(labels_path, taxonomy_path, lines, new_rows)
        print(f"Renamed: {old_key} -> {new_label}")
        return 0

    if args.command == "set-tags":
        key = require_label(rows, args.label)
        tags = split_tags(args.tags)
        if not tags:
            raise SystemExit("At least one hierarchy path is required.")
        rows[key] = tags
        save_both(labels_path, taxonomy_path, lines, rows)
        print(f"Set: {key} -> {' | '.join(tags)}")
        return 0

    if args.command == "add-path":
        key = require_label(rows, args.label)
        tags = rows[key]
        candidate = split_tags(args.tag)
        if len(candidate) != 1:
            raise SystemExit("--tag must contain exactly one hierarchy path.")
        tag = candidate[0]
        if tag.casefold() not in {x.casefold() for x in tags}:
            rows[key] = tags + [tag]
            save_both(labels_path, taxonomy_path, lines, rows)
        print(f"{key} -> {' | '.join(rows[key])}")
        return 0

    if args.command == "remove-path":
        key = require_label(rows, args.label)
        candidate = split_tags(args.tag)
        if len(candidate) != 1:
            raise SystemExit("--tag must contain exactly one hierarchy path.")
        target = candidate[0].casefold()
        new_tags = [tag for tag in rows[key] if tag.casefold() != target]
        if not new_tags:
            raise SystemExit("Refusing to remove the last hierarchy path; use remove if you want to delete the label.")
        rows[key] = new_tags
        save_both(labels_path, taxonomy_path, lines, rows)
        print(f"{key} -> {' | '.join(rows[key])}")
        return 0

    raise SystemExit("Unknown command")


if __name__ == "__main__":
    raise SystemExit(main())
