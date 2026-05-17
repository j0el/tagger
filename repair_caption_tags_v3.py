#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Set, Tuple
from xml.etree import ElementTree as ET

NAMESPACES = {
    "x": "adobe:ns:meta/",
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "dc": "http://purl.org/dc/elements/1.1/",
}
for prefix, uri in NAMESPACES.items():
    ET.register_namespace(prefix, uri)

AI_PREFIX = "ai:"

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
    "contains", "featuring", "featured",
}

GENERIC = {
    "thing", "things", "object", "objects", "stuff", "item", "items", "part", "parts",
    "area", "place", "places", "background", "foreground", "something", "someone",
    "person", "people", "man", "woman", "boy", "girl", "child", "children", "group",
}

LOW_QUALITY_DEFAULT_TAG = ""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Repair sidecars with captions but no tags by adding one "
            "ai:prospective/<best-term> tag based on the caption."
        )
    )
    p.add_argument("root", help="Folder containing .xmp sidecars.")
    p.add_argument("--recurse", action="store_true", help="Scan recursively.")
    p.add_argument("--labels-file", default="labels_curated_hierarchical.txt", help="Curated labels file.")
    p.add_argument("--apply", action="store_true", help="Actually write changes. Default is dry run.")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--prospective-prefix", default="prospective", help="Prefix after ai:, default ai:prospective/<term>")
    p.add_argument("--min-quality", type=float, default=0.55, help="Minimum caption quality score, 0..1.")
    p.add_argument("--allow-low-quality", action="store_true", help="Allow low-quality captions anyway.")
    p.add_argument("--bad-caption-tag", default=LOW_QUALITY_DEFAULT_TAG, help="Optional fixed tag for low-quality captions, e.g. ai:caption/low-quality")
    p.add_argument("--prefer-bigrams", action="store_true", default=True, help="Prefer 2-word phrases over single words.")
    return p.parse_args()


def iter_sidecars(root: Path, recurse: bool) -> Iterable[Path]:
    yield from (root.rglob("*.xmp") if recurse else root.glob("*.xmp"))


def read_labels(path: Path) -> Set[str]:
    existing: Set[str] = set()
    if not path.exists():
        return existing
    for line in path.read_text(encoding="utf-8").splitlines():
        s = normalize_label(line)
        if s and not s.startswith("#"):
            existing.add(s)
    return existing


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

    short_repeats = [w for w, c in counts.items() if len(w) <= 4 and c >= 4]
    if short_repeats:
        reasons.append("short fragment loop: " + ", ".join(short_repeats[:5]))
    if raw.count(" ' ") >= 3:
        reasons.append("apostrophe fragments")

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


def simple_singular(word: str) -> str:
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
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


def clean_caption(caption: str) -> str:
    s = caption.strip().lower()
    s = re.sub(r"^describe this image.*?:\s*", "", s, flags=re.I)
    s = re.sub(r"https?://\S+", "", s)
    s = s.replace("_", " ")
    s = re.sub(r"[^a-z0-9&' /-]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip(" -/")
    return s


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


def candidate_terms_from_caption(caption: str) -> List[str]:
    s = clean_caption(caption)
    words = collapse_repetition(words_from_text(s))
    # normalize individual words
    cleaned: List[str] = []
    for w in words:
        w = simple_singular(normalize_label(w))
        if not w or w in STOPWORDS:
            cleaned.append("")  # separator for phrase logic
            continue
        cleaned.append(w)

    candidates: List[str] = []

    # contiguous ngrams from non-empty cleaned terms
    max_n = 3
    for n in (2, 3, 1):  # prefer bigrams, then trigrams, then unigrams
        for i in range(len(cleaned) - n + 1):
            chunk = cleaned[i:i+n]
            if any(not x for x in chunk):
                continue
            phrase = " ".join(chunk).strip()
            if is_bad_candidate(phrase):
                continue
            candidates.append(phrase)

    # de-duplicate while preserving order
    seen: Set[str] = set()
    out: List[str] = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def is_bad_candidate(term: str) -> bool:
    term = normalize_label(term)
    if not term or len(term) < 3:
        return True
    parts = term.split()
    if any(p in STOPWORDS for p in parts):
        return True
    if all(p in GENERIC for p in parts):
        return True
    if len(parts) == 1 and parts[0] in GENERIC:
        return True
    # obvious junk
    if re.fullmatch(r"[0-9x]+", term):
        return True
    return False


def pick_best_prospective_tag(caption: str, existing_labels: Set[str]) -> Optional[str]:
    candidates = candidate_terms_from_caption(caption)

    def score(term: str) -> Tuple[int, int, int, str]:
        words = term.split()
        # prefer 2-word phrases, then 3-word, then 1-word
        phrase_pref = 3 if len(words) == 2 else 2 if len(words) == 3 else 1
        specific = sum(len(w) for w in words)
        return (phrase_pref, specific, -len(words), term)

    filtered = [c for c in candidates if c not in existing_labels]
    if not filtered:
        return None
    filtered.sort(key=score, reverse=True)
    return filtered[0]


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
    labels_path = Path(args.labels_file).expanduser().resolve()
    existing_labels = read_labels(labels_path)

    scanned = 0
    invalid = 0
    captioned_no_tags = 0
    repaired = 0
    skipped_no_caption = 0
    skipped_has_tags = 0
    skipped_low_quality = 0
    skipped_no_candidate = 0

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
            fixed_tag = args.bad_caption_tag.strip()
            if fixed_tag:
                tag = fixed_tag if fixed_tag.startswith(AI_PREFIX) else AI_PREFIX + fixed_tag
            else:
                skipped_low_quality += 1
                if args.verbose:
                    print(f"SKIP LOW QUALITY q={quality:.2f}: {xmp}")
                    print(f"  reasons: {', '.join(reasons)}")
                    print(f"  caption: {caption}")
                continue
        else:
            term = pick_best_prospective_tag(caption, existing_labels)
            if not term:
                skipped_no_candidate += 1
                if args.verbose:
                    print(f"SKIP NO CANDIDATE: {xmp}")
                    print(f"  quality: {quality:.2f}" + (f" ({', '.join(reasons)})" if reasons else ""))
                    print(f"  caption: {caption}")
                continue
            tag = AI_PREFIX + args.prospective_prefix.strip(" /") + "/" + term

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
    print(f"skipped_no_candidate: {skipped_no_candidate}")
    print(f"repaired: {repaired}")
    print(f"mode: {'APPLY' if args.apply else 'DRY RUN'}")

    if not args.apply:
        print("\nDry run only. Rerun with --apply to write tags.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
