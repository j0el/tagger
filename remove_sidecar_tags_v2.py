#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import html
import re
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple
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
    method: str = "xml"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Remove unwanted takeout / immich-go tags from XMP sidecars. Dry-run by default."
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
        help="Only remove exact tag matches or exact path-segment matches. Default is contains-match.",
    )
    p.add_argument("--apply", action="store_true", help="Actually edit sidecars. Without this, no files are changed.")
    p.add_argument("--no-backup", action="store_true", help="Do not create timestamped .bak backups before editing.")
    p.add_argument("--csv", default=None, help="Optional CSV report path.")
    p.add_argument("--quiet", action="store_true", help="Only print summary lines.")
    p.add_argument(
        "--strict",
        action="store_true",
        help="Return exit code 1 if any sidecar has an error. Default reports errors but exits 0.",
    )
    p.add_argument(
        "--no-text-fallback",
        action="store_true",
        help="Do not try regex/text cleanup when XML parsing fails.",
    )
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
    value = value.casefold().strip().replace("_", " ")
    value = re.sub(r"\s+", " ", value)
    return value


def split_path_segments(tag: str) -> List[str]:
    return [seg.strip() for seg in re.split(r"[/\\|>]", tag) if seg.strip()]


def should_remove(tag: str, terms: Sequence[str], exact: bool) -> bool:
    raw = html.unescape(tag).strip()
    if not raw:
        return False

    tag_norm = normalize_for_match(raw)
    terms_norm = [normalize_for_match(t) for t in terms if normalize_for_match(t)]

    if exact:
        segments = [normalize_for_match(s) for s in split_path_segments(raw)]
        return any(tag_norm == term or term in segments for term in terms_norm)

    tag_loose = tag_norm.replace("-", " ").replace(" ", "")
    for term in terms_norm:
        term_loose = term.replace("-", " ").replace(" ", "")
        if term in tag_norm or term_loose in tag_loose:
            return True
    return False


def dedupe_preserve_order(values: Sequence[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for value in values:
        key = value.casefold()
        if key not in seen:
            out.append(value)
            seen.add(key)
    return out


def split_tag_list_text(value: str) -> Tuple[List[str], str]:
    if "|" in value:
        delimiter = "|"
    elif ";" in value:
        delimiter = ";"
    else:
        delimiter = ","
    return [p.strip() for p in value.split(delimiter) if p.strip()], delimiter


def rewrite_list_value(value: str, terms: Sequence[str], exact: bool) -> Tuple[str, List[str]]:
    parts, delimiter = split_tag_list_text(html.unescape(value))
    if not parts:
        return value, []
    kept: List[str] = []
    removed: List[str] = []
    for part in parts:
        if should_remove(part, terms, exact):
            removed.append(part)
        else:
            kept.append(part)
    if not removed:
        return value, []
    return f"{delimiter} ".join(kept), removed


def tag_container_elements(root: ET.Element) -> List[ET.Element]:
    containers: List[ET.Element] = []
    for elem in root.iter():
        lname = local_name(elem.tag).lower()
        if lname in {"subject", "tagslist"}:
            containers.append(elem)
    seen = set()
    out: List[ET.Element] = []
    for elem in containers:
        if id(elem) not in seen:
            out.append(elem)
            seen.add(id(elem))
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


def rewrite_text_tag_list(elem: ET.Element, terms: Sequence[str], exact: bool) -> Tuple[bool, List[str]]:
    text = elem.text or ""
    if not text.strip():
        return False, []
    new_text, removed = rewrite_list_value(text, terms, exact)
    if removed:
        elem.text = new_text
        return True, removed
    return False, []


def rewrite_tagslist_attributes(root: ET.Element, terms: Sequence[str], exact: bool) -> Tuple[bool, List[str]]:
    changed = False
    removed_all: List[str] = []
    for elem in root.iter():
        for attr, value in list(elem.attrib.items()):
            if local_name(attr).lower() != "tagslist":
                continue
            new_value, removed = rewrite_list_value(value, terms, exact)
            if removed:
                if new_value:
                    elem.set(attr, new_value)
                else:
                    del elem.attrib[attr]
                removed_all.extend(removed)
                changed = True
    return changed, removed_all


def cleanup_tree(tree: ET.ElementTree, terms: Sequence[str], exact: bool) -> Tuple[bool, List[str]]:
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

    return changed, dedupe_preserve_order(removed_all)


LI_PATTERN = re.compile(
    r"(?P<full><(?P<tag>(?:[A-Za-z_][\w.-]*:)?li)\b(?P<attrs>[^>]*)>"
    r"(?P<body>.*?)"
    r"</(?P=tag)>)",
    flags=re.DOTALL,
)

TAGSLIST_ELEM_PATTERN = re.compile(
    r"(?P<open><(?P<tag>(?:[A-Za-z_][\w.-]*:)?TagsList)\b[^>]*>)"
    r"(?P<body>.*?)"
    r"(?P<close></(?P=tag)>)",
    flags=re.DOTALL,
)

TAGSLIST_ATTR_PATTERN = re.compile(
    r"(?P<name>(?:[A-Za-z_][\w.-]*:)?TagsList)=(?P<quote>['\"])(?P<value>.*?)(?P=quote)",
    flags=re.DOTALL,
)


def cleanup_text(raw: str, terms: Sequence[str], exact: bool) -> Tuple[str, List[str]]:
    removed_all: List[str] = []

    def li_repl(match: re.Match[str]) -> str:
        body = match.group("body")
        plain = re.sub(r"<[^>]+>", "", body).strip()
        plain = html.unescape(plain)
        if should_remove(plain, terms, exact):
            removed_all.append(plain)
            return ""
        return match.group("full")

    raw = LI_PATTERN.sub(li_repl, raw)

    def tagslist_elem_repl(match: re.Match[str]) -> str:
        body = html.unescape(match.group("body").strip())
        new_body, removed = rewrite_list_value(body, terms, exact)
        if removed:
            removed_all.extend(removed)
            return match.group("open") + html.escape(new_body, quote=False) + match.group("close")
        return match.group(0)

    raw = TAGSLIST_ELEM_PATTERN.sub(tagslist_elem_repl, raw)

    def attr_repl(match: re.Match[str]) -> str:
        value = html.unescape(match.group("value"))
        new_value, removed = rewrite_list_value(value, terms, exact)
        if removed:
            removed_all.extend(removed)
            q = match.group("quote")
            return f"{match.group('name')}={q}{html.escape(new_value, quote=True)}{q}"
        return match.group(0)

    raw = TAGSLIST_ATTR_PATTERN.sub(attr_repl, raw)
    return raw, dedupe_preserve_order(removed_all)


def backup_path(path: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return path.with_name(f"{path.name}.{stamp}.bak")


def cleanup_sidecar(path: Path, terms: Sequence[str], exact: bool, apply: bool, make_backup: bool, text_fallback: bool) -> CleanupResult:
    try:
        tree = ET.parse(path)
        changed, removed = cleanup_tree(tree, terms, exact)
        if changed and apply:
            if make_backup:
                shutil.copy2(path, backup_path(path))
            tmp = path.with_name(path.name + ".tmp")
            tree.write(tmp, encoding="utf-8", xml_declaration=True)
            tmp.replace(path)
        return CleanupResult(path=path, changed=changed, removed=removed, method="xml")
    except ET.ParseError as exc:
        parse_error = f"XML parse error: {exc}"
        if not text_fallback:
            return CleanupResult(path=path, changed=False, removed=[], error=parse_error, method="xml")
        try:
            raw = path.read_text(encoding="utf-8", errors="replace")
            new_raw, removed = cleanup_text(raw, terms, exact)
            changed = bool(removed) and new_raw != raw
            if changed and apply:
                if make_backup:
                    shutil.copy2(path, backup_path(path))
                tmp = path.with_name(path.name + ".tmp")
                tmp.write_text(new_raw, encoding="utf-8")
                tmp.replace(path)
            err = parse_error if not changed else f"{parse_error}; used text fallback"
            return CleanupResult(path=path, changed=changed, removed=removed, error=err, method="text-fallback")
        except OSError as read_exc:
            return CleanupResult(path=path, changed=False, removed=[], error=f"{parse_error}; text fallback failed: {read_exc}", method="text-fallback")
    except OSError as exc:
        return CleanupResult(path=path, changed=False, removed=[], error=str(exc), method="xml")


def write_csv_report(path: Path, results: Sequence[CleanupResult]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sidecar", "changed", "method", "removed_tags", "error"])
        for r in results:
            w.writerow([str(r.path), r.changed, r.method, "|".join(r.removed), r.error])


def main() -> int:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()

    if not root.exists():
        print(f"Not found: {root}", file=sys.stderr)
        return 2

    sidecars = list(iter_sidecars(root, args.recurse))
    if not sidecars:
        print(f"No .xmp sidecars found under: {root}", file=sys.stderr)
        return 2

    results: List[CleanupResult] = []
    for idx, path in enumerate(sidecars, start=1):
        results.append(
            cleanup_sidecar(
                path,
                args.terms,
                args.exact,
                apply=args.apply,
                make_backup=not args.no_backup,
                text_fallback=not args.no_text_fallback,
            )
        )
        if not args.quiet and idx % 5000 == 0:
            print(f"Scanned {idx:,} sidecars...", file=sys.stderr)

    changed_results = [r for r in results if r.changed]
    error_results = [r for r in results if r.error]
    text_fallback_results = [r for r in results if r.method == "text-fallback"]
    removed_count = sum(len(r.removed) for r in changed_results)

    mode = "APPLY" if args.apply else "DRY RUN"
    print(f"\nMode: {mode}")
    print(f"Root: {root}")
    print(f"Sidecars scanned: {len(sidecars)}")
    print(f"Sidecars with matching tags: {len(changed_results)}")
    print(f"Unique removed tag entries across changed sidecars: {removed_count}")
    print(f"Sidecars using text fallback: {len(text_fallback_results)}")
    print(f"Warnings/errors reported: {len(error_results)}")

    if not args.quiet:
        for r in changed_results[:100]:
            print(f"\n{r.path}")
            if r.method == "text-fallback":
                print("  method: text fallback")
            for tag in r.removed:
                print(f"  remove: {tag}")
        if len(changed_results) > 100:
            print(f"\n... {len(changed_results) - 100} more changed sidecars not shown")

        for r in error_results[:20]:
            print(f"\nWARNING {r.path}: {r.error}", file=sys.stderr)
        if len(error_results) > 20:
            print(f"\n... {len(error_results) - 20} more warnings/errors not shown", file=sys.stderr)

    if args.csv:
        csv_path = Path(args.csv).expanduser().resolve()
        write_csv_report(csv_path, results)
        print(f"\nWrote CSV report: {csv_path}")

    if not args.apply and changed_results:
        print("\nDry run only. Re-run with --apply to edit files.")
        print("Backups are created by default. Use --no-backup only if you already have a backup.")

    return 1 if (args.strict and error_results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
