#!/usr/bin/env python3
"""Generate the two taxonomy files from Immich captions.

Two-step workflow:

  1. extract — mine noun candidates from every asset description in Immich and
     write an editable candidates file, ordered by frequency (most common first):

       uv run python generate_taxonomy.py extract

  2. Edit tag_candidates.txt by hand: DELETE the lines you don't want as labels.
     Order and trailing "# 412x ..." comments don't matter.

  3. build — assign each surviving label one or more hierarchy paths using the
     local llama-server, then write labels_curated_hierarchical.txt (grouped into
     commented sections by top-level category) and labels_taxonomy_map.csv:

       uv run python generate_taxonomy.py build

Existing output files are backed up to <name>.bak-<timestamp> before overwrite.
After building, re-tag the library with --reprocess-all (see README, "Updating
and re-tagging") — the label/taxonomy change invalidates every cache entry.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import time
import urllib.request
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

from immich_api import ImmichClient, load_dotenv

DEFAULT_CANDIDATES_FILE = "tag_candidates.txt"
DEFAULT_LABELS_FILE = "labels_curated_hierarchical.txt"
DEFAULT_TAXONOMY_MAP = "labels_taxonomy_map.csv"
DEFAULT_VLM_URL = "http://localhost:8082"
DEFAULT_VLM_MODEL = "qwen2.5vl:7b"

# Fallback top-level categories; overridden by the ones found in an existing
# taxonomy map (so rebuilds stay consistent with the current hierarchy).
DEFAULT_TOP_LEVELS = [
    "People", "Events", "Activities", "Sports", "Nature", "Animals", "Plants",
    "Water", "Food", "Places", "Buildings", "Transportation", "Objects",
    "Clothing", "Technology", "Art",
]

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
    "pink", "purple", "dark", "light", "left", "right", "top", "bottom", "middle",
    "center", "while", "against", "atop", "amid", "among", "along",
    # Common VLM caption verbs/adjectives (captions come from Qwen2.5-VL, which
    # has a small, repetitive vocabulary for these).
    "stand", "stands", "sit", "sits", "seated", "smile", "smiling", "enjoy",
    "enjoying", "surrounded", "surrounding", "together", "nestled", "gather",
    "gathered", "gathering", "glide", "glides", "gliding", "labeled", "filled",
    "covered", "adorned", "captured", "posing", "pose", "poses", "visible",
    "appears", "appear", "seen", "clear", "colorful", "lush", "vibrant", "cozy",
    "serene", "scenic", "beautiful", "bright", "sunny", "cloudy", "blurred",
    "up", "down", "other", "another", "each", "both", "all", "various",
    "out", "what", "where", "amidst", "featuring", "featured", "displayed",
    "dressed", "distant", "warmly", "brightly", "softly", "gently", "gracefully",
    "casually", "neatly", "partially", "slightly",
}

GENERIC_NOUNS = {
    "thing", "things", "object", "objects", "area", "place", "places", "part",
    "parts", "side", "sides", "wall", "floor", "hand", "hands", "head",
    "face", "background", "foreground", "lot", "bunch", "set", "line",
    "day", "time", "way", "moment", "setting", "surface", "edge", "corner",
}


def eprint(*args, **kwargs) -> None:
    print(*args, file=sys.stderr, **kwargs)


# ------------------------------------------------------------------ #
# Shared text helpers                                                  #
# ------------------------------------------------------------------ #

def normalize_label(s: str) -> str:
    s = s.strip().lower().replace("_", " ")
    s = re.sub(r"[^a-z0-9&' /-]+", "", s)
    s = re.sub(r"\s+", " ", s).strip(" -/")
    return s


def simple_singular(word: str) -> str:
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    # Strip "es" only after sibilant stems (boxes, churches); "trees"/"houses"
    # are plain "+s" plurals and lose just the "s".
    if len(word) > 4 and word.endswith("es") and word[:-2].endswith(("s", "x", "z", "ch", "sh")):
        return word[:-2]
    if len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]
    return word


def is_bad_candidate(term: str, min_len: int = 3) -> bool:
    if len(term) < min_len or term in STOPWORDS or term in GENERIC_NOUNS or term.isdigit():
        return True
    if re.fullmatch(r"[0-9a-f]{6,}", term):
        return True
    if any(part in STOPWORDS for part in term.split()):
        return True
    return False


def title_path(path: str) -> str:
    """Normalize a hierarchy path to Title Case components: 'water/marine' -> 'Water/Marine'."""
    parts = []
    for part in path.split("/"):
        part = re.sub(r"\s+", " ", part.strip())
        if part:
            parts.append(" ".join(w if w in ("&",) else w[:1].upper() + w[1:] for w in part.split(" ")))
    return "/".join(parts)


def backup_if_exists(path: Path) -> None:
    if path.exists():
        bak = path.with_name(f"{path.name}.bak-{datetime.now().strftime('%Y%m%d-%H%M%S')}")
        shutil.copy2(path, bak)
        eprint(f"Backed up existing {path} -> {bak}")


# ------------------------------------------------------------------ #
# extract: captions -> frequency-ordered candidates file               #
# ------------------------------------------------------------------ #

def extract_terms(caption: str, min_len: int, bigrams: bool) -> Set[str]:
    """Return the set of candidate terms (unigrams + adjacent bigrams) in one caption."""
    raw = re.findall(r"[A-Za-z][A-Za-z'-]*", caption.lower())
    kept: List[Optional[str]] = []
    for tok in raw:
        term = simple_singular(normalize_label(tok))
        kept.append(term if term and not is_bad_candidate(term, min_len) else None)

    terms: Set[str] = {t for t in kept if t}
    if bigrams:
        for a, b in zip(kept, kept[1:]):
            if a and b:
                terms.add(f"{a} {b}")
    return terms


def cmd_extract(args: argparse.Namespace) -> int:
    load_dotenv()
    import os
    url, key = os.environ.get("IMMICH_URL"), os.environ.get("IMMICH_API_KEY")
    if not url or not key:
        eprint("ERROR: IMMICH_URL / IMMICH_API_KEY not set (create .env).")
        return 1
    client = ImmichClient(url, key)

    total: Counter[str] = Counter()
    docs: Counter[str] = Counter()
    people_words: Set[str] = set()
    assets = captioned = 0
    for asset in client.find_new_assets(page_size=args.page_size):
        assets += 1
        if assets % 2000 == 0:
            eprint(f"  scanned {assets} assets ({captioned} with captions)...")
        for name in asset.people_names:
            people_words.update(simple_singular(w) for w in normalize_label(name).split())
        if not asset.description:
            continue
        captioned += 1
        terms = extract_terms(asset.description, args.min_len, not args.no_bigrams)
        docs.update(terms)
        # Per-caption presence is what matters for taxonomy labels; counting a
        # term once per caption also keeps unigram/bigram counts comparable.
        total.update(terms)

    if captioned == 0:
        eprint("No captioned assets found; nothing to extract.")
        return 1

    # People names (from Immich face recognition) end up in captions verbatim;
    # they belong to face tags, not the taxonomy.
    if people_words:
        eprint(f"Excluding {len(people_words)} words from named people.")

    rows: List[Tuple[str, int, int, float]] = []
    for term, n in total.most_common():
        if any(w in people_words for w in term.split()):
            continue
        d = docs[term]
        pct = 100.0 * d / captioned
        if n < args.min_count or d < args.min_docs or pct > args.max_doc_pct:
            continue
        rows.append((term, n, d, pct))

    out = Path(args.out)
    backup_if_exists(out)
    with out.open("w", encoding="utf-8") as f:
        f.write(
            f"# Candidate tags mined from {captioned} captioned assets "
            f"(of {assets} total) on {datetime.now():%Y-%m-%d %H:%M}.\n"
            f"# Ordered by frequency. EDIT THIS FILE: delete the lines you do NOT want\n"
            f"# as labels, then run:  uv run python generate_taxonomy.py build\n"
            f"# Trailing '# Nx ...' comments are informational and ignored by build.\n\n"
        )
        for term, n, d, pct in rows:
            f.write(f"{term:<32}# {n}x, in {d} captions ({pct:.1f}%)\n")

    eprint(
        f"Wrote {len(rows)} candidates to {out} "
        f"(from {captioned}/{assets} captioned assets; filters: "
        f"min-count={args.min_count} min-docs={args.min_docs} max-doc-pct={args.max_doc_pct})"
    )
    return 0


# ------------------------------------------------------------------ #
# build: edited candidates -> hierarchy via llama-server -> two files  #
# ------------------------------------------------------------------ #

def read_candidates(path: Path) -> List[str]:
    labels: List[str] = []
    seen: Set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0]
        label = normalize_label(line)
        if label and label not in seen:
            seen.add(label)
            labels.append(label)
    return labels


def existing_top_levels(taxonomy_path: Path) -> List[str]:
    """Top-level categories from an existing taxonomy map, most-used first."""
    if not taxonomy_path.exists():
        return []
    counts: Counter[str] = Counter()
    for line in taxonomy_path.read_text(encoding="utf-8").splitlines()[1:]:
        if "," not in line:
            continue
        _, _, tags = line.partition(",")
        for p in re.split(r"[|;]", tags):
            top = title_path(p).split("/")[0]
            if top:
                counts[top] += 1
    return [t for t, _ in counts.most_common()]


def chat(vlm_url: str, model: str, prompt: str, max_tokens: int, timeout: int = 300) -> str:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        # temperature 0 makes this model fall into exact repetition loops on
        # list-shaped output; a little sampling noise breaks them.
        "temperature": 0.3,
    }
    req = urllib.request.Request(
        f"{vlm_url.rstrip('/')}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        result = json.loads(resp.read())
    return result.get("choices", [{}])[0].get("message", {}).get("content", "")


HIERARCHY_PROMPT = """\
You are organizing tags for a personal photo library into a hierarchy.

For each label below output exactly one line:
label => Top/Sub/Label

Rules:
- The path has 2 or 3 levels, most general first, ending with the label itself in Title Case.
- The FIRST component MUST be one of: {top_levels}
- One line per label, same order as given, no other text.

Examples:
sky => Nature/Sky
boat => Transportation/Boat
birthday party => Events/Birthday Party
oak tree => Plants/Trees/Oak Tree

Labels:
{labels}
"""


def assign_hierarchy(
    labels: Sequence[str],
    top_levels: Sequence[str],
    vlm_url: str,
    model: str,
    batch_size: int,
    verbose: bool,
) -> Dict[str, List[str]]:
    """label -> list of Title Case hierarchy paths, via llama-server."""
    allowed = {t.lower() for t in top_levels}
    mapping: Dict[str, List[str]] = {}
    pending = list(labels)
    for attempt in range(2):
        missing: List[str] = []
        for i in range(0, len(pending), batch_size):
            batch = pending[i : i + batch_size]
            prompt = HIERARCHY_PROMPT.format(
                top_levels=", ".join(top_levels), labels="\n".join(batch)
            )
            try:
                # ~24 tokens per output line; the cap bounds the damage if the
                # model loops anyway.
                reply = chat(vlm_url, model, prompt, max_tokens=24 * len(batch) + 128)
            except Exception as exc:
                eprint(f"WARNING: VLM request failed ({exc}); will retry batch")
                missing.extend(batch)
                continue
            got = parse_hierarchy_reply(reply, batch, allowed, verbose)
            mapping.update(got)
            missing.extend(l for l in batch if l not in mapping)
            done = len(mapping)
            eprint(f"  hierarchy: {done}/{len(labels)} labels assigned")
        if not missing:
            break
        pending = missing
        if attempt == 0 and missing:
            eprint(f"Retrying {len(missing)} unassigned labels...")
            time.sleep(2)

    for label in labels:
        if label not in mapping:
            eprint(f"WARNING: no hierarchy for {label!r}; using Prospective/{title_path(label)}")
            mapping[label] = [f"Prospective/{title_path(label)}"]
    return mapping


def parse_hierarchy_reply(
    reply: str, batch: Sequence[str], allowed_tops: Set[str], verbose: bool
) -> Dict[str, List[str]]:
    by_norm = {normalize_label(l): l for l in batch}
    out: Dict[str, List[str]] = {}
    for line in reply.splitlines():
        if "=>" not in line:
            continue
        left, _, right = line.partition("=>")
        label = by_norm.get(normalize_label(left))
        if label is None:
            continue
        paths: List[str] = []
        for p in re.split(r"[|;]", right):
            path = title_path(re.sub(r"[^A-Za-z0-9&' /-]+", "", p))
            parts = path.split("/")
            if not (2 <= len(parts) <= 4) or not all(parts):
                continue
            if parts[0].lower() not in allowed_tops:
                if verbose:
                    eprint(f"  note: off-list top level {parts[0]!r} for {label!r}")
                continue
            if path not in paths:
                paths.append(path)
        if paths:
            out[label] = paths[:2]
    return out


def cmd_build(args: argparse.Namespace) -> int:
    candidates_path = Path(args.candidates)
    if not candidates_path.exists():
        eprint(f"ERROR: {candidates_path} not found — run the extract step first.")
        return 1
    labels = read_candidates(candidates_path)
    if not labels:
        eprint(f"ERROR: no labels left in {candidates_path}.")
        return 1

    if args.top_levels:
        top_levels = [title_path(t) for t in args.top_levels.split(",") if t.strip()]
    else:
        top_levels = existing_top_levels(Path(args.out_taxonomy)) or DEFAULT_TOP_LEVELS
    eprint(f"Building hierarchy for {len(labels)} labels; top levels: {', '.join(top_levels)}")

    try:
        urllib.request.urlopen(f"{args.vlm_url.rstrip('/')}/health", timeout=5)
    except Exception:
        eprint(f"ERROR: llama-server not reachable at {args.vlm_url} (check `systemctl status llama-vlm`).")
        return 1

    mapping = assign_hierarchy(
        labels, top_levels, args.vlm_url, args.vlm_model, args.batch_size, args.verbose
    )

    # Group labels by the top level of their first path, in top_levels order,
    # preserving candidate (frequency) order within each section.
    section_order = top_levels + sorted(
        {mapping[l][0].split("/")[0] for l in labels} - set(top_levels)
    )
    sections: Dict[str, List[str]] = {s: [] for s in section_order}
    for label in labels:
        sections[mapping[label][0].split("/")[0]].append(label)

    labels_out = Path(args.out_labels)
    taxonomy_out = Path(args.out_taxonomy)
    if args.dry_run:
        for label in labels:
            print(f"{label} -> {' | '.join(mapping[label])}")
        eprint(f"Dry run: would write {labels_out} and {taxonomy_out}.")
        return 0

    backup_if_exists(labels_out)
    backup_if_exists(taxonomy_out)

    with labels_out.open("w", encoding="utf-8") as f:
        f.write(f"# Generated by generate_taxonomy.py on {datetime.now():%Y-%m-%d} "
                f"from {candidates_path.name}. Editable — see README.\n")
        for section in section_order:
            if not sections[section]:
                continue
            f.write(f"\n# --- {section} ---\n")
            for label in sections[section]:
                f.write(f"{label}\n")

    with taxonomy_out.open("w", encoding="utf-8") as f:
        f.write("label,tags\n")
        for section in section_order:
            for label in sections[section]:
                f.write(f"{label},{'|'.join(mapping[label])}\n")

    eprint(
        f"Wrote {len(labels)} labels to {labels_out} and {taxonomy_out}.\n"
        f"Next: test on one asset (--asset-id ... --dry-run --verbose), then re-tag "
        f"the library with --reprocess-all (README, 'Updating and re-tagging')."
    )
    return 0


# ------------------------------------------------------------------ #
# CLI                                                                  #
# ------------------------------------------------------------------ #

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    ex = sub.add_parser("extract", help="Mine caption nouns from Immich into an editable candidates file.")
    ex.add_argument("--out", default=DEFAULT_CANDIDATES_FILE)
    ex.add_argument("--min-count", type=int, default=3, help="Drop terms with fewer total occurrences.")
    ex.add_argument("--min-docs", type=int, default=2, help="Drop terms appearing in fewer captions.")
    ex.add_argument("--max-doc-pct", type=float, default=50.0,
                    help="Drop terms appearing in more than this %% of captions (boilerplate).")
    ex.add_argument("--min-len", type=int, default=3)
    ex.add_argument("--no-bigrams", action="store_true", help="Unigrams only (skip 'birthday party' style phrases).")
    ex.add_argument("--page-size", type=int, default=100)
    ex.set_defaults(func=cmd_extract)

    b = sub.add_parser("build", help="Build the two taxonomy files from the edited candidates file.")
    b.add_argument("--candidates", default=DEFAULT_CANDIDATES_FILE)
    b.add_argument("--out-labels", default=DEFAULT_LABELS_FILE)
    b.add_argument("--out-taxonomy", default=DEFAULT_TAXONOMY_MAP)
    b.add_argument("--vlm-url", default=DEFAULT_VLM_URL)
    b.add_argument("--vlm-model", default=DEFAULT_VLM_MODEL)
    b.add_argument("--batch-size", type=int, default=20)
    b.add_argument("--top-levels", default=None,
                   help="Comma-separated top-level categories (default: from existing taxonomy map).")
    b.add_argument("--dry-run", action="store_true", help="Print the mapping; write nothing.")
    b.add_argument("--verbose", action="store_true")
    b.set_defaults(func=cmd_build)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
