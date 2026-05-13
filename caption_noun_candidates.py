#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
from collections import Counter, OrderedDict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Set, Tuple
from xml.etree import ElementTree as ET

NAMESPACES = {
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "dc": "http://purl.org/dc/elements/1.1/",
}

DEFAULT_LABELS_FILE = "labels_curated_hierarchical.txt"
DEFAULT_TAXONOMY_MAP = "labels_taxonomy_map.csv"

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "else", "when", "while",
    "of", "in", "on", "at", "to", "with", "without", "for", "from", "by", "as",
    "into", "onto", "over", "under", "near", "next", "beside", "behind", "front",
    "back", "between", "through", "across", "around", "inside", "outside",
    "is", "are", "was", "were", "be", "been", "being", "am", "has", "have",
    "had", "do", "does", "did", "can", "could", "would", "should", "will",
    "may", "might", "must", "this", "that", "these", "those", "there", "here",
    "it", "its", "they", "them", "their", "he", "she", "his", "her", "hers",
    "him", "you", "your", "we", "our", "i", "me", "my", "image", "photo",
    "picture", "photograph", "shot", "view", "scene", "showing", "shows", "depicts",
    "features", "contains", "looking", "standing", "sitting", "holding", "wearing",
    "walking", "playing", "one", "two", "three", "four", "five", "many", "several",
    "some", "few", "large", "small", "little", "big", "old", "young", "new", "white",
    "black", "blue", "red", "green", "yellow", "brown", "gray", "grey", "orange",
    "pink", "purple", "dark", "light", "left", "right", "top", "bottom", "middle", "center",
}

GENERIC_NOUNS = {
    "thing", "things", "object", "objects", "area", "place", "places", "part",
    "parts", "side", "sides", "room", "wall", "floor", "hand", "hands", "head",
    "face", "people", "person", "man", "woman", "boy", "girl", "child", "background",
    "foreground", "lot", "group", "bunch", "set", "line",
}


@dataclass
class Candidate:
    term: str
    total_count: int
    doc_count: int
    doc_pct: float
    action: str
    reason: str


def eprint(*args, **kwargs) -> None:
    print(*args, file=sys.stderr, **kwargs)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Read XMP captions, build a noun frequency list, and optionally add useful candidates to the curated label list and taxonomy map."
    )
    p.add_argument("root", help="Root folder containing .xmp sidecars.")
    p.add_argument("--recurse", action="store_true", help="Scan sidecars recursively.")
    p.add_argument("--labels-file", default=DEFAULT_LABELS_FILE)
    p.add_argument("--taxonomy-map", default=DEFAULT_TAXONOMY_MAP)
    p.add_argument("--min-count", type=int, default=3, help="Exclude terms with fewer total occurrences.")
    p.add_argument("--min-docs", type=int, default=2, help="Exclude terms appearing in fewer sidecars.")
    p.add_argument("--max-doc-pct", type=float, default=15.0, help="Exclude terms appearing in more than this percent of captioned sidecars.")
    p.add_argument("--min-len", type=int, default=3, help="Minimum candidate length.")
    p.add_argument("--limit-add", type=int, default=50, help="Maximum number of new labels to add when --apply is used.")
    p.add_argument("--taxonomy-prefix", default="Prospective/Nouns", help="Hierarchy prefix for newly discovered labels.")
    p.add_argument("--section", default="Prospective/Nouns", help="Section heading in labels_curated_hierarchical.txt for new labels.")
    p.add_argument("--include-phrases", action="store_true", help="With spaCy installed, also consider noun chunks such as 'pocket watch'.")
    p.add_argument("--spacy-model", default="en_core_web_sm", help="spaCy model to try. Falls back to heuristic extraction if unavailable.")
    p.add_argument("--no-spacy", action="store_true", help="Use the fallback heuristic extractor even if spaCy is installed.")
    p.add_argument("--top", type=int, default=100, help="Number of candidate rows to print.")
    p.add_argument("--csv", default=None, help="Write full candidate table to this CSV file.")
    p.add_argument("--json", default=None, help="Write summary and candidates to this JSON file.")
    p.add_argument("--apply", action="store_true", help="Actually add candidates. Without this, the program is a dry run.")
    p.add_argument("--yes", action="store_true", help="Do not prompt before applying changes. Only meaningful with --apply.")
    return p.parse_args()


def iter_sidecars(root: Path, recurse: bool) -> Iterable[Path]:
    yield from (root.rglob("*.xmp") if recurse else root.glob("*.xmp"))


def read_xmp_caption(path: Path) -> Optional[str]:
    try:
        tree = ET.parse(path)
    except ET.ParseError:
        return None
    root = tree.getroot()
    desc_node = root.find(".//dc:description/rdf:Alt/rdf:li", NAMESPACES)
    if desc_node is not None and desc_node.text and desc_node.text.strip():
        return desc_node.text.strip()
    desc_node = root.find(".//{*}description//{*}li")
    if desc_node is not None and desc_node.text and desc_node.text.strip():
        return desc_node.text.strip()
    return None


def normalize_label(s: str) -> str:
    s = s.strip().lower().replace("_", " ")
    s = re.sub(r"[^a-z0-9&' /-]+", "", s)
    s = re.sub(r"\s+", " ", s).strip(" -/")
    return s


def title_path_part(s: str) -> str:
    s = normalize_label(s)
    return s[0].upper() + s[1:] if s else ""


def simple_singular(word: str) -> str:
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 4 and word.endswith("es") and not word.endswith(("ses", "xes", "ches", "shes")):
        return word[:-2]
    if len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]
    return word


def is_bad_candidate(term: str, min_len: int) -> bool:
    term = normalize_label(term)
    if len(term) < min_len or term in STOPWORDS or term in GENERIC_NOUNS or term.isdigit():
        return True
    if re.fullmatch(r"[0-9a-f]{6,}", term):
        return True
    parts = term.split()
    if len(parts) > 1 and any(part in STOPWORDS for part in parts):
        return True
    return False


def extract_fallback(captions: Sequence[str], min_len: int) -> Tuple[Counter[str], Counter[str], str]:
    total: Counter[str] = Counter()
    docs: Counter[str] = Counter()
    for caption in captions:
        tokens = re.findall(r"[A-Za-z][A-Za-z'-]*", caption.lower())
        seen: Set[str] = set()
        for token in tokens:
            token = simple_singular(normalize_label(token))
            if is_bad_candidate(token, min_len):
                continue
            total[token] += 1
            seen.add(token)
        docs.update(seen)
    return total, docs, "fallback-heuristic"


def try_load_spacy(model_name: str):
    try:
        import spacy  # type: ignore
    except Exception:
        return None, "spaCy is not installed"
    try:
        return spacy.load(model_name), None
    except Exception as exc:
        return None, f"Could not load spaCy model {model_name!r}: {exc}"


def extract_spacy(captions: Sequence[str], model_name: str, min_len: int, include_phrases: bool) -> Tuple[Counter[str], Counter[str], str]:
    nlp, err = try_load_spacy(model_name)
    if nlp is None:
        eprint(f"Warning: {err}")
        eprint("Falling back to heuristic extraction. For better noun extraction: uv add spacy && uv run python -m spacy download en_core_web_sm")
        return extract_fallback(captions, min_len)
    total: Counter[str] = Counter()
    docs_counter: Counter[str] = Counter()
    for doc in nlp.pipe(captions, batch_size=64):
        seen: Set[str] = set()
        for token in doc:
            if token.pos_ not in {"NOUN", "PROPN"}:
                continue
            term = simple_singular(normalize_label(token.lemma_ if token.lemma_ else token.text))
            if token.is_stop or is_bad_candidate(term, min_len):
                continue
            total[term] += 1
            seen.add(term)
        if include_phrases:
            for chunk in doc.noun_chunks:
                words: List[str] = []
                for token in chunk:
                    if token.is_stop or token.pos_ not in {"NOUN", "PROPN", "ADJ"}:
                        continue
                    word = simple_singular(normalize_label(token.lemma_ if token.lemma_ else token.text))
                    if word and word not in STOPWORDS:
                        words.append(word)
                phrase = normalize_label(" ".join(words))
                if phrase and len(phrase.split()) >= 2 and not is_bad_candidate(phrase, min_len):
                    total[phrase] += 1
                    seen.add(phrase)
        docs_counter.update(seen)
    return total, docs_counter, f"spacy:{model_name}"


def read_labels(path: Path) -> Tuple[List[str], Set[str]]:
    labels: List[str] = []
    if not path.exists():
        return labels, set()
    for line in path.read_text(encoding="utf-8").splitlines():
        s = normalize_label(line)
        if s and not s.startswith("#"):
            labels.append(s)
    return labels, set(labels)


def read_label_lines(path: Path) -> List[str]:
    return path.read_text(encoding="utf-8").splitlines() if path.exists() else []


def label_exists_in_lines(lines: Sequence[str], label: str) -> bool:
    target = normalize_label(label)
    return any(normalize_label(line) == target for line in lines if normalize_label(line) and not normalize_label(line).startswith("#"))


def add_label_to_section(lines: List[str], label: str, section: str) -> List[str]:
    label = normalize_label(label)
    heading = f"# --- {section} ---"
    if label_exists_in_lines(lines, label):
        return lines
    if heading in lines:
        idx = lines.index(heading) + 1
        while idx < len(lines) and not lines[idx].startswith("# --- "):
            idx += 1
        insert_at = idx
        if insert_at > 0 and lines[insert_at - 1] == "":
            insert_at -= 1
        return lines[:insert_at] + [label] + lines[insert_at:]
    if lines and lines[-1].strip():
        lines = lines + [""]
    return lines + [heading, label]


def read_taxonomy(path: Path) -> "OrderedDict[str, List[str]]":
    rows: "OrderedDict[str, List[str]]" = OrderedDict()
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != ["label", "tags"]:
            raise ValueError(f"Expected CSV header label,tags in {path}")
        for row in reader:
            label = normalize_label(row.get("label", ""))
            tags_raw = row.get("tags", "") or ""
            tags: List[str] = []
            seen: Set[str] = set()
            for tag in tags_raw.split("|"):
                tag = "/".join(part.strip() for part in tag.strip().split("/") if part.strip())
                if tag and tag.casefold() not in seen:
                    tags.append(tag)
                    seen.add(tag.casefold())
            if label:
                rows[label] = tags
    return rows


def write_taxonomy(path: Path, rows: "OrderedDict[str, List[str]]") -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label", "tags"])
        for label, tags in rows.items():
            w.writerow([label, "|".join(tags)])


def backup(path: Path) -> None:
    if path.exists():
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        shutil.copy2(path, path.with_name(path.name + f".{stamp}.bak"))


def write_labels(path: Path, lines: List[str]) -> None:
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def build_candidates(total: Counter[str], docs: Counter[str], total_captioned: int, existing_labels: Set[str], min_count: int, min_docs: int, max_doc_pct: float, min_len: int) -> List[Candidate]:
    candidates: List[Candidate] = []
    for term in sorted(total.keys(), key=lambda t: (-total[t], -docs[t], t)):
        term = normalize_label(term)
        doc_count = docs[term]
        doc_pct = (doc_count / total_captioned * 100.0) if total_captioned else 0.0
        action = "add"
        reasons: List[str] = []
        if is_bad_candidate(term, min_len):
            action = "skip"; reasons.append("bad/generic")
        if total[term] < min_count:
            action = "skip"; reasons.append(f"rare total<{min_count}")
        if doc_count < min_docs:
            action = "skip"; reasons.append(f"rare docs<{min_docs}")
        if doc_pct > max_doc_pct:
            action = "skip"; reasons.append(f"common doc_pct>{max_doc_pct:g}")
        if term in existing_labels:
            action = "skip"; reasons.append("already in label list")
        candidates.append(Candidate(term, total[term], doc_count, doc_pct, action, ", ".join(reasons) if reasons else "new candidate"))
    return candidates


def taxonomy_path_for_candidate(term: str, prefix: str) -> str:
    prefix = "/".join(part.strip() for part in prefix.split("/") if part.strip())
    return f"{prefix}/{title_path_part(term)}"


def write_candidate_csv(path: Path, candidates: Sequence[Candidate]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["term", "total_count", "doc_count", "doc_pct", "action", "reason"])
        for c in candidates:
            w.writerow([c.term, c.total_count, c.doc_count, f"{c.doc_pct:.2f}", c.action, c.reason])


def apply_candidates(candidates: Sequence[Candidate], labels_path: Path, taxonomy_path: Path, section: str, taxonomy_prefix: str, limit_add: int, yes: bool) -> int:
    to_add = [c for c in candidates if c.action == "add"][:limit_add]
    if not to_add:
        print("No new labels to add.")
        return 0
    print(f"\nAbout to add {len(to_add)} labels:")
    for c in to_add:
        print(f"  {c.term} -> {taxonomy_path_for_candidate(c.term, taxonomy_prefix)}")
    if not yes:
        answer = input("\nType YES to apply these changes: ").strip()
        if answer != "YES":
            print("Cancelled. No files changed.")
            return 0
    label_lines = read_label_lines(labels_path)
    taxonomy = read_taxonomy(taxonomy_path)
    for c in to_add:
        label_lines = add_label_to_section(label_lines, c.term, section)
        if c.term not in taxonomy:
            taxonomy[c.term] = [taxonomy_path_for_candidate(c.term, taxonomy_prefix)]
    backup(labels_path)
    backup(taxonomy_path)
    write_labels(labels_path, label_lines)
    write_taxonomy(taxonomy_path, taxonomy)
    print(f"Added {len(to_add)} labels.")
    return len(to_add)


def main() -> int:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    labels_path = Path(args.labels_file).expanduser().resolve()
    taxonomy_path = Path(args.taxonomy_map).expanduser().resolve()

    sidecars = list(iter_sidecars(root, args.recurse))
    captions = [c for p in sidecars for c in [read_xmp_caption(p)] if c]
    _labels, existing = read_labels(labels_path)

    if args.no_spacy:
        total, docs, extractor = extract_fallback(captions, args.min_len)
    else:
        total, docs, extractor = extract_spacy(captions, args.spacy_model, args.min_len, args.include_phrases)

    candidates = build_candidates(total, docs, len(captions), existing, args.min_count, args.min_docs, args.max_doc_pct, args.min_len)
    addable = [c for c in candidates if c.action == "add"]
    rare = [c for c in candidates if "rare" in c.reason]
    common = [c for c in candidates if "common" in c.reason]
    existing_skips = [c for c in candidates if "already in label list" in c.reason]

    print("\nSummary")
    print("-------")
    print(f"root: {root}")
    print(f"sidecars_scanned: {len(sidecars)}")
    print(f"captioned_sidecars: {len(captions)}")
    print(f"extractor: {extractor}")
    print(f"existing_labels: {len(existing)}")
    print(f"unique_noun_candidates_seen: {len(candidates)}")
    print(f"new_candidates_after_filters: {len(addable)}")
    print(f"skipped_rare: {len(rare)}")
    print(f"skipped_common: {len(common)}")
    print(f"skipped_existing_labels: {len(existing_skips)}")
    print(f"mode: {'APPLY' if args.apply else 'DRY RUN'}")

    print(f"\nTop {args.top} addable noun candidates")
    print("-" * 36)
    for c in addable[: args.top]:
        print(f"{c.total_count:>6} total  {c.doc_count:>5} docs  {c.doc_pct:>6.2f}%  {c.term}")

    print(f"\nTop {min(args.top, 30)} skipped common terms")
    print("-" * 35)
    for c in common[: min(args.top, 30)]:
        print(f"{c.total_count:>6} total  {c.doc_count:>5} docs  {c.doc_pct:>6.2f}%  {c.term}  ({c.reason})")

    if args.csv:
        write_candidate_csv(Path(args.csv), candidates)
        print(f"\nWrote CSV: {args.csv}")
    if args.json:
        payload = {
            "summary": {
                "root": str(root),
                "sidecars_scanned": len(sidecars),
                "captioned_sidecars": len(captions),
                "extractor": extractor,
                "existing_labels": len(existing),
                "unique_noun_candidates_seen": len(candidates),
                "new_candidates_after_filters": len(addable),
                "mode": "apply" if args.apply else "dry-run",
            },
            "candidates": [c.__dict__ for c in candidates],
        }
        Path(args.json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote JSON: {args.json}")

    if args.apply:
        apply_candidates(candidates, labels_path, taxonomy_path, args.section, args.taxonomy_prefix, args.limit_add, args.yes)
    else:
        print("\nDry run only. No files changed.")
        print("To apply, rerun with --apply. Add --yes to skip the confirmation prompt.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
