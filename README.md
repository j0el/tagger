# Immich Caption and Tag Tools

A small toolkit for maintaining XMP sidecars for a photo library, intended to
give Immich better AI captions and hierarchical tags than it produces on its own.

The main program scans image (and optionally video) files and, for each one:

1. creates a caption only when the sidecar has no existing `dc:description`,
2. creates an XMP sidecar if one does not already exist,
3. adds AI tags from the curated candidate-label list,
4. maps those labels into the hierarchy from `labels_taxonomy_map.csv`,
5. if the classifier finds no useful curated tag, mines curated labels from the caption,
6. only if that still finds nothing, creates a conservative fallback tag from the caption.

Human tags already present in a sidecar are preserved. Existing `ai:` tags are
replaced by the current AI run.

## What's in this repo

Everything below is current and maintained. Each Python tool also carries a short
docstring at the top of the file describing its role.

### Core tagging pipeline

| File | Purpose |
| --- | --- |
| `immich_caption_and_tag.py` | Main one-pass caption + tagging program. The canonical entry point. |
| `labels_curated_hierarchical.txt` | Candidate labels used by the classifier. |
| `labels_taxonomy_map.csv` | Mapping from candidate labels to hierarchical tag paths. |
| `tag_stats.py` | Summarizes captions and tags found in sidecars; can validate against the taxonomy and write CSV/JSON reports. |
| `taxonomy_manager.py` | Safely adds/removes/renames labels and edits hierarchy mappings. Writes timestamped backups before every change. |

### Supporting tools

| File | Purpose |
| --- | --- |
| `caption_noun_candidates.py` | Reads existing captions, builds a noun-frequency list, and can append useful new candidates to the label list and taxonomy map. Also used by the GUI. |
| `mirror_dc_subject_to_immich_xmp_tags.py` | Specialty interop tool. Mirrors `dc:subject` tags into `digiKam:TagsList` and `lr:HierarchicalSubject` so tags show up in Immich, digiKam, and Lightroom. Dry-run by default. |
| `remove_sidecar_tags_v2.py` | Removes unwanted Google-Takeout / immich-go tags from sidecars. Dry-run by default. |
| `immich_tagger_streamlit_app.py` | Local browser GUI that wraps the tools above. See `README_STREAMLIT_GUI.md`. |
| `run_daily_new_images.sh` | Example daily-automation script for a full-library run. |

### Setup and docs

| File | Purpose |
| --- | --- |
| `README.md` | This file. |
| `README_STREAMLIT_GUI.md` | Setup and usage for the Streamlit GUI. |
| `INSTALL_NEW_SYSTEM.md` | Installing the toolkit on a fresh machine. |
| `pyproject.toml`, `uv.lock` | Project dependencies (managed with `uv`). |

## Initial setup

From inside the project directory:

```bash
git init
uv init --bare
uv add torch torchvision transformers pillow tqdm accelerate sentencepiece protobuf pillow-heif
git add .
git commit -m "Initial caption and tag tools"
```

On macOS, the default `--device auto` should choose Metal/MPS when available.

For a full walkthrough on a new machine, see `INSTALL_NEW_SYSTEM.md`.

## Run the main program

A cautious first test on a small copied folder:

```bash
uv run python immich_caption_and_tag.py /path/to/test-folder \
  --recurse \
  --labels-file labels_curated_hierarchical.txt \
  --taxonomy-map labels_taxonomy_map.csv \
  --db-path .immich_auto_tagger_cache.sqlite3 \
  --verbose
```

Typical full-library run:

```bash
uv run python immich_caption_and_tag.py /Volumes/oldmacData/library/upload \
  --recurse \
  --labels-file labels_curated_hierarchical.txt \
  --taxonomy-map labels_taxonomy_map.csv \
  --db-path .immich_auto_tagger_cache.sqlite3
```

Useful options:

```bash
# Preview which files would be processed without writing sidecars
uv run python immich_caption_and_tag.py /path/to/library \
  --recurse --labels-file labels_curated_hierarchical.txt \
  --taxonomy-map labels_taxonomy_map.csv --dry-run --verbose

# Recompute tags even when the cache says the image was already classified
uv run python immich_caption_and_tag.py /path/to/library \
  --recurse --labels-file labels_curated_hierarchical.txt \
  --taxonomy-map labels_taxonomy_map.csv --force

# Skip caption creation and only run tagging
uv run python immich_caption_and_tag.py /path/to/library \
  --recurse --labels-file labels_curated_hierarchical.txt \
  --taxonomy-map labels_taxonomy_map.csv --skip-captioning
```

## Inspect the tags actually present in sidecars

```bash
uv run python tag_stats.py /Volumes/oldmacData/library/upload --recurse
```

With hierarchy validation against the current taxonomy:

```bash
uv run python tag_stats.py /Volumes/oldmacData/library/upload \
  --recurse --taxonomy-map labels_taxonomy_map.csv --top 50
```

Write CSV/JSON reports:

```bash
uv run python tag_stats.py /Volumes/oldmacData/library/upload \
  --recurse --taxonomy-map labels_taxonomy_map.csv \
  --csv-prefix tag_stats --json tag_stats_summary.json
```

## Maintain the candidate labels and hierarchy

```bash
# Show a label
uv run python taxonomy_manager.py show "boat"

# Add a new candidate label and hierarchy path
uv run python taxonomy_manager.py add "red phalarope" \
  --tags "Nature/Birds/Red phalarope" --section "Nature"

# Remove a candidate label from both files
uv run python taxonomy_manager.py remove "red phalarope"

# Replace the hierarchy path(s) for a label
uv run python taxonomy_manager.py set-tags "boat" \
  --tags "Water/Marine/Boat|Transportation/Boat"

# Add or remove one hierarchy path without replacing the others
uv run python taxonomy_manager.py add-path "boat" --tag "Transportation/Boat"
uv run python taxonomy_manager.py remove-path "boat" --tag "Transportation/Boat"

# Rename a label while preserving its hierarchy mapping
uv run python taxonomy_manager.py rename "plane flying" "airplane flying"

# Audit for duplicates and mismatches between the two source files
uv run python taxonomy_manager.py audit
```

Every mutating `taxonomy_manager.py` command writes timestamped backups before
editing the two source files.

## Discover new candidate labels from captions

`caption_noun_candidates.py` reads the captions already in your sidecars and
surfaces frequently occurring nouns that aren't yet curated labels. It can
optionally append the useful ones to the label list and taxonomy map (dry-run by
default). It is also exposed through the GUI's "Caption noun candidates" tab.

## Mirror tags for digiKam / Lightroom (optional)

If you also browse the library in digiKam or Lightroom, mirror the `dc:subject`
tags into the fields those apps read. Dry-run by default; pass `--apply` to write:

```bash
uv run python mirror_dc_subject_to_immich_xmp_tags.py /path/to/library \
  --recurse --apply
```

## Remove unwanted imported tags (optional)

Libraries imported via Google Takeout or immich-go often carry noise tags.
`remove_sidecar_tags_v2.py` strips them. Dry-run by default; pass `--apply` to write.

## The GUI

For a point-and-click wrapper around the caption/tagger, tag statistics, boolean
tag search, noun-candidate discovery, and taxonomy management, see
`README_STREAMLIT_GUI.md`:

```bash
uv add streamlit
uv run streamlit run immich_tagger_streamlit_app.py
```

## Suggested workflow

1. Add or refine candidate labels with `taxonomy_manager.py`.
2. Run `taxonomy_manager.py audit`.
3. Run the main caption/tagger on a small test folder.
4. Inspect results with `tag_stats.py`.
5. If the tags look good, run the main program against the full library.
6. Once stable, schedule the main program once daily; because it skips existing
   captions and caches tagging work, daily runs mostly process newly added images.

## Daily automation

After testing, the simplest automation is a daily run. `run_daily_new_images.sh`
is an example wrapper (adjust the paths inside it for your machine). A file-watcher
could be added later, but a daily job is easier to trust because the program is
restartable and already skips completed work.
