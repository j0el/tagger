#!/usr/bin/env python3
"""
Streamlit GUI for Joel's Immich caption/tag sidecar tools.

Place this file in the same project directory as:
  - immich_caption_and_tag.py
  - tag_stats.py
  - caption_noun_candidates.py
  - taxonomy_manager.py
  - labels_curated_hierarchical.txt
  - labels_taxonomy_map.csv

Run:
  uv add streamlit
  uv run streamlit run immich_tagger_streamlit_app.py
"""
from __future__ import annotations

import csv
import io
import json
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Iterator, List, Optional, Sequence, Tuple, Union
from xml.etree import ElementTree as ET

import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError

try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
    HEIF_ENABLED = True
except Exception:
    HEIF_ENABLED = False

APP_TITLE = "Immich Caption + Tag Tools"
DEFAULT_ROOT = "/Volumes/oldmacData/library/upload"
DEFAULT_LABELS = "labels_curated_hierarchical.txt"
DEFAULT_TAXONOMY = "labels_taxonomy_map.csv"
DEFAULT_DB = ".immich_auto_tagger_cache.sqlite3"

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".heic", ".heif"
}
VIDEO_EXTENSIONS = {
    ".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm", ".mts", ".m2ts", ".3gp"
}

NAMESPACES = {
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "dc": "http://purl.org/dc/elements/1.1/",
}
AI_PREFIX = "ai:"


# -----------------------------------------------------------------------------
# General helpers
# -----------------------------------------------------------------------------

def path_text(label: str, default: str, help_text: Optional[str] = None, key: Optional[str] = None) -> str:
    return st.text_input(label, value=default, help=help_text, key=key)


def expand_path(value: str, base_dir: Optional[Path] = None) -> Path:
    p = Path(value).expanduser()
    if not p.is_absolute() and base_dir is not None:
        p = base_dir / p
    return p


def shell_join(args: Sequence[str]) -> str:
    return " ".join(shlex.quote(str(a)) for a in args)


def split_command_prefix(prefix: str) -> List[str]:
    try:
        return shlex.split(prefix)
    except ValueError:
        return [prefix]


def script_path(project_dir: Path, requested_name: str) -> Path:
    """Return a script path. Also tolerate downloaded names like foo(1).py."""
    requested = Path(requested_name).expanduser()
    if requested.is_absolute():
        return requested

    primary = project_dir / requested_name
    if primary.exists():
        return primary

    stem = requested.stem
    suffix = requested.suffix
    alternates = [
        project_dir / f"{stem}(1){suffix}",
        project_dir / f"{stem} (1){suffix}",
    ]
    for alt in alternates:
        if alt.exists():
            return alt
    return primary


def add_bool(args: List[str], flag: str, enabled: bool) -> None:
    if enabled:
        args.append(flag)


def add_opt(args: List[str], flag: str, value: Union[str, int, float, None], *, skip_blank: bool = True) -> None:
    if value is None:
        return
    if skip_blank and isinstance(value, str) and not value.strip():
        return
    args.extend([flag, str(value)])


def command_base(prefix: str, project_dir: Path, script_name: str) -> List[str]:
    return split_command_prefix(prefix) + [str(script_path(project_dir, script_name))]


def run_command_live(args: Sequence[str], cwd: Path, title: str = "Run output") -> int:
    st.caption("Command")
    st.code(shell_join(args), language="bash")

    out_box = st.empty()
    status_box = st.empty()
    output_lines: List[str] = []
    started = time.time()

    try:
        proc = subprocess.Popen(
            list(args),
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )
    except FileNotFoundError as exc:
        st.error(f"Could not start command: {exc}")
        return 127
    except Exception as exc:
        st.error(f"Could not start command: {exc}")
        return 1

    assert proc.stdout is not None
    for line in proc.stdout:
        output_lines.append(line.rstrip("\n"))
        # Keep the browser responsive by not rendering an enormous log every line forever.
        visible = "\n".join(output_lines[-500:])
        out_box.code(visible, language="text")
        status_box.caption(f"{title}: running for {time.time() - started:,.0f}s")

    rc = proc.wait()
    final_text = "\n".join(output_lines[-1000:])
    out_box.code(final_text or "(no output)", language="text")
    if rc == 0:
        status_box.success(f"Finished successfully in {time.time() - started:,.0f}s")
    else:
        status_box.error(f"Exited with code {rc} after {time.time() - started:,.0f}s")
    return rc


# -----------------------------------------------------------------------------
# XMP indexing and boolean search
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class SidecarRecord:
    sidecar_path: str
    media_path: str
    media_exists: bool
    tags: Tuple[str, ...]
    caption: str
    valid_xml: bool


def iter_sidecars(root: Path, recurse: bool) -> Iterator[Path]:
    if recurse:
        yield from root.rglob("*.xmp")
    else:
        yield from root.glob("*.xmp")


def media_from_sidecar(sidecar: Path) -> Path:
    # Your sidecars are normally IMG_1234.jpg.xmp, so dropping .xmp restores the media path.
    return sidecar.with_suffix("")


def read_xmp_sidecar(sidecar: Path) -> Tuple[List[str], str, bool]:
    try:
        tree = ET.parse(sidecar)
        root = tree.getroot()
    except ET.ParseError:
        return [], "", False
    except Exception:
        return [], "", False

    tags: List[str] = []
    for li in root.findall(".//dc:subject/rdf:Bag/rdf:li", NAMESPACES):
        if li.text and li.text.strip():
            tags.append(li.text.strip())
    if not tags:
        for li in root.findall(".//{*}subject//{*}li"):
            if li.text and li.text.strip():
                tags.append(li.text.strip())

    caption = ""
    node = root.find(".//dc:description/rdf:Alt/rdf:li", NAMESPACES)
    if node is not None and node.text and node.text.strip():
        caption = node.text.strip()
    if not caption:
        node = root.find(".//{*}description//{*}li")
        if node is not None and node.text and node.text.strip():
            caption = node.text.strip()

    return tags, caption, True


@st.cache_data(show_spinner="Scanning XMP sidecars...", ttl=60 * 30)
def build_sidecar_index(root_str: str, recurse: bool, max_sidecars: int) -> List[SidecarRecord]:
    root = Path(root_str).expanduser()
    records: List[SidecarRecord] = []
    if not root.exists():
        return records

    for i, sidecar in enumerate(iter_sidecars(root, recurse), start=1):
        if max_sidecars > 0 and i > max_sidecars:
            break
        media = media_from_sidecar(sidecar)
        tags, caption, ok = read_xmp_sidecar(sidecar)
        records.append(
            SidecarRecord(
                sidecar_path=str(sidecar),
                media_path=str(media),
                media_exists=media.exists(),
                tags=tuple(tags),
                caption=caption,
                valid_xml=ok,
            )
        )
    return records


def norm_text(value: str) -> str:
    value = value.lower().strip()
    value = value.replace("_", " ")
    value = re.sub(r"\s+", " ", value)
    return value


def stripped_ai(tag: str) -> str:
    tag_n = norm_text(tag)
    return tag_n[len(AI_PREFIX):] if tag_n.startswith(AI_PREFIX) else tag_n


def tag_segments(tag: str) -> List[str]:
    raw = stripped_ai(tag)
    pieces = [raw]
    pieces.extend(part.strip() for part in re.split(r"[/|;:,>]", raw) if part.strip())
    for part in list(pieces):
        pieces.extend(w for w in re.findall(r"[a-z0-9&'#-]+", part) if w)
    # Dedupe while preserving order.
    seen = set()
    out = []
    for p in pieces:
        p = norm_text(p)
        if p and p not in seen:
            out.append(p)
            seen.add(p)
    return out


def term_matches(term: str, tags: Sequence[str], caption: str, match_mode: str, include_caption: bool) -> bool:
    t = norm_text(term)
    if not t:
        return False

    haystacks: List[str] = []
    segments: List[str] = []
    for tag in tags:
        raw = stripped_ai(tag)
        haystacks.append(raw)
        segments.extend(tag_segments(tag))

    if include_caption and caption:
        haystacks.append(norm_text(caption))

    if match_mode == "exact tag":
        return any(t == stripped_ai(tag) or t == norm_text(tag) for tag in tags)
    if match_mode == "path segment / word":
        return any(t == seg for seg in segments)
    # Friendly default: any substring within full tag/caption text.
    return any(t in h for h in haystacks)


Token = Tuple[str, str]
Ast = Union[Tuple[str, str], Tuple[str, object], Tuple[str, object, object]]

TOKEN_RE = re.compile(
    r"\s*(?:(AND|OR|NOT)\b|([()])|\"([^\"]+)\"|'([^']+)'|([^\s()]+))",
    re.IGNORECASE,
)


def tokenize(expr: str) -> List[Token]:
    tokens: List[Token] = []
    pos = 0
    while pos < len(expr):
        m = TOKEN_RE.match(expr, pos)
        if not m:
            raise ValueError(f"Could not parse near: {expr[pos:pos+40]!r}")
        pos = m.end()
        if m.group(1):
            tokens.append((m.group(1).upper(), m.group(1).upper()))
        elif m.group(2):
            tokens.append((m.group(2), m.group(2)))
        else:
            term = m.group(3) or m.group(4) or m.group(5) or ""
            tokens.append(("TERM", term))
    return tokens


class BooleanParser:
    def __init__(self, tokens: Sequence[Token]):
        self.tokens = list(tokens)
        self.i = 0

    def peek(self) -> Optional[Token]:
        if self.i >= len(self.tokens):
            return None
        return self.tokens[self.i]

    def accept(self, kind: str) -> Optional[Token]:
        tok = self.peek()
        if tok and tok[0] == kind:
            self.i += 1
            return tok
        return None

    def parse(self) -> Ast:
        if not self.tokens:
            raise ValueError("Enter a boolean search expression.")
        ast = self.parse_or()
        if self.peek() is not None:
            raise ValueError(f"Unexpected token: {self.peek()[1]}")
        return ast

    def parse_or(self) -> Ast:
        node = self.parse_and()
        while self.accept("OR"):
            node = ("OR", node, self.parse_and())
        return node

    def parse_and(self) -> Ast:
        node = self.parse_not()
        while True:
            if self.accept("AND"):
                node = ("AND", node, self.parse_not())
                continue
            nxt = self.peek()
            # Friendly implicit AND: dog cat means dog AND cat, and dog (cat OR bird) works.
            if nxt and nxt[0] in {"TERM", "NOT", "("}:
                node = ("AND", node, self.parse_not())
                continue
            break
        return node

    def parse_not(self) -> Ast:
        if self.accept("NOT"):
            return ("NOT", self.parse_not())
        return self.parse_atom()

    def parse_atom(self) -> Ast:
        tok = self.peek()
        if tok is None:
            raise ValueError("Unexpected end of expression.")
        if tok[0] == "TERM":
            self.i += 1
            return ("TERM", tok[1])
        if self.accept("("):
            node = self.parse_or()
            if not self.accept(")"):
                raise ValueError("Missing closing parenthesis.")
            return node
        raise ValueError(f"Unexpected token: {tok[1]}")


def parse_boolean(expr: str) -> Ast:
    return BooleanParser(tokenize(expr)).parse()


def eval_ast(ast: Ast, record: SidecarRecord, match_mode: str, include_caption: bool) -> bool:
    op = ast[0]
    if op == "TERM":
        return term_matches(str(ast[1]), record.tags, record.caption, match_mode, include_caption)
    if op == "NOT":
        return not eval_ast(ast[1], record, match_mode, include_caption)  # type: ignore[arg-type]
    if op == "AND":
        return eval_ast(ast[1], record, match_mode, include_caption) and eval_ast(ast[2], record, match_mode, include_caption)  # type: ignore[arg-type]
    if op == "OR":
        return eval_ast(ast[1], record, match_mode, include_caption) or eval_ast(ast[2], record, match_mode, include_caption)  # type: ignore[arg-type]
    raise ValueError(f"Unknown AST op: {op}")


@st.cache_data(show_spinner=False, max_entries=512)
def image_thumb_bytes(path_str: str, max_px: int = 384) -> Optional[bytes]:
    path = Path(path_str)
    if not path.exists() or path.suffix.lower() not in IMAGE_EXTENSIONS:
        return None
    try:
        with Image.open(path) as img:
            img = ImageOps.exif_transpose(img).convert("RGB")
            img.thumbnail((max_px, max_px), Image.Resampling.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=88)
            return buf.getvalue()
    except (UnidentifiedImageError, OSError, ValueError):
        return None


def records_to_csv(records: Sequence[SidecarRecord]) -> bytes:
    out = io.StringIO()
    w = csv.writer(out)
    w.writerow(["media_path", "sidecar_path", "media_exists", "valid_xml", "caption", "tags"])
    for r in records:
        w.writerow([r.media_path, r.sidecar_path, r.media_exists, r.valid_xml, r.caption, "|".join(r.tags)])
    return out.getvalue().encode("utf-8")


# -----------------------------------------------------------------------------
# Streamlit UI
# -----------------------------------------------------------------------------

st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)
st.caption("Local browser GUI for the caption/tagger, tag statistics, taxonomy maintenance, noun candidates, and boolean sidecar search.")

with st.sidebar:
    st.header("Project paths")
    project_dir_str = st.text_input("Project directory", value=str(Path.cwd()), help="Folder containing the existing Python scripts.")
    project_dir = Path(project_dir_str).expanduser().resolve()

    run_prefix = st.text_input("Python command prefix", value="uv run python", help="Examples: uv run python, python, /path/to/.venv/bin/python")
    library_root = st.text_input("Library / sidecar root", value=DEFAULT_ROOT)
    labels_file = st.text_input("Labels file", value=DEFAULT_LABELS)
    taxonomy_map = st.text_input("Taxonomy map", value=DEFAULT_TAXONOMY)
    db_path = st.text_input("Tagger cache DB", value=DEFAULT_DB)

    st.divider()
    st.header("Script names")
    tagger_script = st.text_input("Caption/tag script", value="immich_caption_and_tag.py")
    stats_script = st.text_input("Tag stats script", value="tag_stats.py")
    noun_script = st.text_input("Noun candidates script", value="caption_noun_candidates.py")
    taxonomy_script = st.text_input("Taxonomy manager script", value="taxonomy_manager.py")

    st.divider()
    st.header("Common")
    common_recurse = st.checkbox("Recurse", value=True)
    show_cmd_only = st.checkbox("Build commands only; do not run", value=False)

    if not HEIF_ENABLED:
        st.warning("pillow-heif is not loaded, so HEIC previews may not display in the search tab.")

root_path = expand_path(library_root)
labels_path = expand_path(labels_file, project_dir)
taxonomy_path = expand_path(taxonomy_map, project_dir)
db_full_path = expand_path(db_path, project_dir)

if not project_dir.exists():
    st.warning(f"Project directory does not exist yet: {project_dir}")


tab_tagger, tab_search, tab_stats, tab_nouns, tab_taxonomy = st.tabs([
    "Run caption/tagger",
    "Boolean tag search",
    "Tag statistics",
    "Caption noun candidates",
    "Taxonomy manager",
])

with tab_tagger:
    st.subheader("Run the main caption + tag program")
    st.write("This wraps `immich_caption_and_tag.py` with common options exposed as controls.")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        include_videos = st.checkbox("Include videos", value=False)
        skip_captioning = st.checkbox("Skip captioning", value=False)
        dry_run = st.checkbox("Dry run", value=True)
        verbose = st.checkbox("Verbose", value=True)
    with c2:
        force = st.checkbox("Force recompute", value=False)
        rebuild_hash_cache = st.checkbox("Rebuild hash cache", value=False)
        only_untagged = st.checkbox("Only untagged", value=False)
        only_poorly_tagged = st.checkbox("Only poorly tagged", value=False)
    with c3:
        no_prefix = st.checkbox("No ai: prefix", value=False)
        no_merge_existing = st.checkbox("Do not merge existing tags", value=False)
        device = st.selectbox("Device", ["auto", "mps", "cuda", "cpu"], index=0)
        limit = st.number_input("Limit files, 0 = no limit", min_value=0, value=25 if dry_run else 0, step=25)
    with c4:
        batch_size = st.number_input("Tag batch size", min_value=1, value=4, step=1)
        caption_batch_size = st.number_input("Caption batch size", min_value=1, value=4, step=1)
        hash_workers = st.number_input("Hash workers", min_value=1, value=4, step=1)
        preload_workers = st.number_input("Preload workers", min_value=1, value=4, step=1)

    with st.expander("Advanced model and threshold settings"):
        c1, c2, c3 = st.columns(3)
        with c1:
            zero_shot_model = st.text_input("Zero-shot model", value="google/siglip-so400m-patch14-384")
            caption_model = st.text_input("Caption model", value="Salesforce/blip-image-captioning-base")
            caption_task = st.selectbox("Caption task", ["auto", "image-to-text", "image-text-to-text"], index=0)
        with c2:
            top_k = st.number_input("Top K", min_value=1, value=6, step=1)
            threshold = st.number_input("Threshold", min_value=0.0, max_value=1.0, value=0.32, step=0.01, format="%.2f")
            relative_threshold = st.number_input("Relative threshold", min_value=0.0, max_value=1.0, value=0.70, step=0.01, format="%.2f")
        with c3:
            fallback_top_k = st.number_input("Fallback Top K", min_value=1, value=3, step=1)
            fallback_threshold = st.number_input("Fallback threshold", min_value=0.0, max_value=1.0, value=0.20, step=0.01, format="%.2f")
            fallback_relative_threshold = st.number_input("Fallback relative", min_value=0.0, max_value=1.0, value=0.50, step=0.01, format="%.2f")

        c4, c5, c6 = st.columns(3)
        with c4:
            max_side = st.number_input("Max image side", min_value=128, value=768, step=64)
            caption_max_new_tokens = st.number_input("Caption max new tokens", min_value=8, value=40, step=4)
        with c5:
            video_frames = st.number_input("Video frames", min_value=1, value=3, step=1)
            heic_fallback = st.selectbox("HEIC fallback", ["auto", "sips", "ffmpeg", "none"], index=0)
        with c6:
            after_date = st.text_input("After lastTaggedAt date", value="", help="Example: 2026-05-01")
            modified_after = st.text_input("File modified after", value="", help="Example: 2026-05-01")
            tagger_version = st.text_input("Tagger version", value="hierarchical-v7-video-captionrescue")

    args = command_base(run_prefix, project_dir, tagger_script)
    args.append(str(root_path))
    add_bool(args, "--recurse", common_recurse)
    add_bool(args, "--include-videos", include_videos)
    add_opt(args, "--video-frames", int(video_frames))
    add_opt(args, "--labels-file", str(labels_path))
    add_opt(args, "--taxonomy-map", str(taxonomy_path))
    add_opt(args, "--db-path", str(db_full_path))
    add_opt(args, "--device", device)
    add_opt(args, "--zero-shot-model", zero_shot_model)
    add_bool(args, "--skip-captioning", skip_captioning)
    add_opt(args, "--caption-model", caption_model)
    add_opt(args, "--caption-task", caption_task)
    add_opt(args, "--caption-max-new-tokens", int(caption_max_new_tokens))
    add_opt(args, "--caption-batch-size", int(caption_batch_size))
    add_opt(args, "--heic-fallback", heic_fallback)
    add_opt(args, "--batch-size", int(batch_size))
    add_opt(args, "--top-k", int(top_k))
    add_opt(args, "--threshold", float(threshold))
    add_opt(args, "--relative-threshold", float(relative_threshold))
    add_opt(args, "--fallback-threshold", float(fallback_threshold))
    add_opt(args, "--fallback-relative-threshold", float(fallback_relative_threshold))
    add_opt(args, "--fallback-top-k", int(fallback_top_k))
    add_opt(args, "--max-side", int(max_side))
    add_opt(args, "--hash-workers", int(hash_workers))
    add_opt(args, "--preload-workers", int(preload_workers))
    add_bool(args, "--dry-run", dry_run)
    add_bool(args, "--verbose", verbose)
    add_bool(args, "--no-prefix", no_prefix)
    add_bool(args, "--no-merge-existing", no_merge_existing)
    add_opt(args, "--limit", int(limit))
    add_bool(args, "--force", force)
    add_bool(args, "--rebuild-hash-cache", rebuild_hash_cache)
    add_bool(args, "--only-untagged", only_untagged)
    add_bool(args, "--only-poorly-tagged", only_poorly_tagged)
    add_opt(args, "--after-date", after_date)
    add_opt(args, "--modified-after", modified_after)
    add_opt(args, "--tagger-version", tagger_version)

    st.caption("Command preview")
    st.code(shell_join(args), language="bash")
    if st.button("Run caption/tagger", type="primary", disabled=show_cmd_only):
        run_command_live(args, project_dir, "caption/tagger")

with tab_search:
    st.subheader("Boolean search over XMP tags")
    st.write("Examples: `food AND NOT people`, `(table OR bed) AND dog`, `\"red phalarope\" AND NOT blurry`.")

    c1, c2, c3, c4 = st.columns([3, 1.2, 1.2, 1.2])
    with c1:
        query = st.text_input("Search expression", value="food AND NOT people")
    with c2:
        match_mode = st.selectbox("Match mode", ["contains", "path segment / word", "exact tag"], index=0)
    with c3:
        include_caption = st.checkbox("Also search captions", value=False)
    with c4:
        search_recurse = st.checkbox("Recurse search", value=common_recurse)

    c5, c6, c7 = st.columns(3)
    with c5:
        max_sidecars = st.number_input("Max sidecars to index, 0 = all", min_value=0, value=0, step=1000)
    with c6:
        max_display = st.number_input("Max displayed matches", min_value=1, value=50, step=10)
    with c7:
        thumb_px = st.number_input("Preview width px, 384 ≈ 4 in", min_value=128, value=384, step=32)

    reload_index = st.button("Clear cached index")
    if reload_index:
        build_sidecar_index.clear()
        image_thumb_bytes.clear()
        st.success("Cache cleared. Run the search again to rescan sidecars.")

    run_search = st.button("Search tags", type="primary")
    if run_search:
        if not root_path.exists():
            st.error(f"Root folder does not exist: {root_path}")
        else:
            try:
                ast = parse_boolean(query)
                records = build_sidecar_index(str(root_path), search_recurse, int(max_sidecars))
                matches = [r for r in records if r.valid_xml and eval_ast(ast, r, match_mode, include_caption)]
                st.success(f"Found {len(matches):,} matches from {len(records):,} sidecars.")
                st.download_button(
                    "Download matches CSV",
                    data=records_to_csv(matches),
                    file_name="immich_tag_search_matches.csv",
                    mime="text/csv",
                    disabled=not matches,
                )

                for r in matches[: int(max_display)]:
                    media_path = Path(r.media_path)
                    with st.container(border=True):
                        left, right = st.columns([1, 2])
                        with left:
                            thumb = image_thumb_bytes(r.media_path, int(thumb_px))
                            if thumb:
                                st.image(thumb, width=int(thumb_px))
                            elif media_path.suffix.lower() in VIDEO_EXTENSIONS:
                                st.info("Video match; thumbnail not generated in this starter app.")
                            elif not r.media_exists:
                                st.warning("Media file not found beside sidecar.")
                            else:
                                st.warning("Preview unavailable for this file type.")
                        with right:
                            st.markdown(f"**Media**: `{r.media_path}`")
                            st.markdown(f"**Sidecar**: `{r.sidecar_path}`")
                            if r.caption:
                                st.markdown(f"**Caption**: {r.caption}")
                            st.markdown("**Tags**")
                            st.code("\n".join(r.tags) if r.tags else "(no tags)", language="text")

                if len(matches) > int(max_display):
                    st.info(f"Showing {int(max_display):,} of {len(matches):,} matches. Increase Max displayed matches to see more.")
            except Exception as exc:
                st.error(str(exc))

with tab_stats:
    st.subheader("Run tag statistics")
    st.write("This wraps `tag_stats.py` and can produce CSV/JSON reports.")
    c1, c2, c3 = st.columns(3)
    with c1:
        stats_top = st.number_input("Top tags", min_value=1, value=50, step=10)
    with c2:
        csv_prefix = st.text_input("CSV prefix", value="tag_stats")
    with c3:
        json_path = st.text_input("JSON output", value="tag_stats_summary.json")

    args = command_base(run_prefix, project_dir, stats_script)
    args.append(str(root_path))
    add_bool(args, "--recurse", common_recurse)
    add_opt(args, "--taxonomy-map", str(taxonomy_path))
    add_opt(args, "--top", int(stats_top))
    add_opt(args, "--csv-prefix", csv_prefix)
    add_opt(args, "--json", json_path)

    st.caption("Command preview")
    st.code(shell_join(args), language="bash")
    if st.button("Run tag statistics", type="primary", disabled=show_cmd_only):
        run_command_live(args, project_dir, "tag stats")

with tab_nouns:
    st.subheader("Find candidate labels from captions")
    st.write("This wraps `caption_noun_candidates.py`; default mode is dry-run unless Apply is checked.")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        min_count = st.number_input("Min total count", min_value=1, value=3, step=1)
        min_docs = st.number_input("Min docs", min_value=1, value=2, step=1)
    with c2:
        max_doc_pct = st.number_input("Max doc percent", min_value=0.1, max_value=100.0, value=15.0, step=1.0)
        min_len = st.number_input("Min term length", min_value=1, value=3, step=1)
    with c3:
        noun_top = st.number_input("Top rows", min_value=1, value=100, step=10)
        limit_add = st.number_input("Limit add", min_value=1, value=50, step=5)
    with c4:
        include_phrases = st.checkbox("Include noun phrases", value=False)
        no_spacy = st.checkbox("No spaCy fallback only", value=False)
        apply_candidates = st.checkbox("Apply changes", value=False)
        yes_apply = st.checkbox("Skip apply prompt", value=False)

    c5, c6, c7, c8 = st.columns(4)
    with c5:
        taxonomy_prefix = st.text_input("Taxonomy prefix", value="Prospective/Nouns")
    with c6:
        section = st.text_input("Labels section", value="Prospective/Nouns")
    with c7:
        noun_csv = st.text_input("Candidate CSV", value="caption_noun_candidates.csv")
    with c8:
        noun_json = st.text_input("Candidate JSON", value="caption_noun_candidates.json")

    args = command_base(run_prefix, project_dir, noun_script)
    args.append(str(root_path))
    add_bool(args, "--recurse", common_recurse)
    add_opt(args, "--labels-file", str(labels_path))
    add_opt(args, "--taxonomy-map", str(taxonomy_path))
    add_opt(args, "--min-count", int(min_count))
    add_opt(args, "--min-docs", int(min_docs))
    add_opt(args, "--max-doc-pct", float(max_doc_pct))
    add_opt(args, "--min-len", int(min_len))
    add_opt(args, "--limit-add", int(limit_add))
    add_opt(args, "--taxonomy-prefix", taxonomy_prefix)
    add_opt(args, "--section", section)
    add_bool(args, "--include-phrases", include_phrases)
    add_bool(args, "--no-spacy", no_spacy)
    add_opt(args, "--top", int(noun_top))
    add_opt(args, "--csv", noun_csv)
    add_opt(args, "--json", noun_json)
    add_bool(args, "--apply", apply_candidates)
    add_bool(args, "--yes", yes_apply)

    if apply_candidates:
        st.warning("Apply mode will edit your labels and taxonomy files. The script itself writes timestamped backups.")
    st.caption("Command preview")
    st.code(shell_join(args), language="bash")
    if st.button("Run noun candidate scan", type="primary", disabled=show_cmd_only):
        run_command_live(args, project_dir, "noun candidates")

with tab_taxonomy:
    st.subheader("Maintain candidate labels and taxonomy map")
    st.write("This wraps `taxonomy_manager.py` for audit, show, add, remove, rename, and path edits.")

    operation = st.selectbox(
        "Operation",
        ["audit", "show", "add", "remove", "rename", "set-tags", "add-path", "remove-path"],
        index=0,
    )

    args = command_base(run_prefix, project_dir, taxonomy_script)
    add_opt(args, "--labels-file", str(labels_path))
    add_opt(args, "--taxonomy-map", str(taxonomy_path))
    args.append(operation)

    if operation == "show":
        label = st.text_input("Label", value="boat")
        args.append(label)
    elif operation == "add":
        label = st.text_input("New label", value="red phalarope")
        paths = st.text_input("Hierarchy path(s), separated with |", value="Nature/Birds/Red phalarope")
        section_name = st.text_input("Section", value="Nature")
        args.append(label)
        add_opt(args, "--tags", paths)
        add_opt(args, "--section", section_name)
    elif operation == "remove":
        label = st.text_input("Label to remove", value="red phalarope")
        args.append(label)
    elif operation == "rename":
        old_label = st.text_input("Old label", value="plane flying")
        new_label = st.text_input("New label", value="airplane flying")
        args.extend([old_label, new_label])
    elif operation == "set-tags":
        label = st.text_input("Label", value="boat")
        paths = st.text_input("Replacement path(s), separated with |", value="Water/Marine/Boat|Transportation/Boat")
        args.append(label)
        add_opt(args, "--tags", paths)
    elif operation == "add-path":
        label = st.text_input("Label", value="boat")
        path_to_add = st.text_input("Hierarchy path to add", value="Transportation/Boat")
        args.append(label)
        add_opt(args, "--tag", path_to_add)
    elif operation == "remove-path":
        label = st.text_input("Label", value="boat")
        path_to_remove = st.text_input("Hierarchy path to remove", value="Transportation/Boat")
        args.append(label)
        add_opt(args, "--tag", path_to_remove)

    st.caption("Command preview")
    st.code(shell_join(args), language="bash")
    if st.button("Run taxonomy command", type="primary", disabled=show_cmd_only):
        run_command_live(args, project_dir, "taxonomy manager")
