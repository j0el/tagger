# Immich Caption and Tag Tools

This project maintains XMP sidecars for a photo library.

The main program:

1. scans image files,
2. creates a caption only when the image sidecar has no existing `dc:description`,
3. creates an XMP sidecar if one does not already exist,
4. adds AI tags from the curated candidate-label list,
5. maps those labels into the hierarchy from `labels_taxonomy_map.csv`,
6. if the classifier finds no useful curated tag, tries to mine curated labels from the caption,
7. only if that still finds nothing, creates a conservative fallback tag from the caption.

Human tags already present in a sidecar are preserved. Existing `ai:` tags are replaced by the current AI run.

## Project files

- `immich_caption_and_tag.py` — main one-pass caption + tagging program
- `labels_curated_hierarchical.txt` — candidate labels used by the classifier
- `labels_taxonomy_map.csv` — mapping from candidate labels to hierarchical tag paths
- `tag_stats.py` — summarizes tags found in sidecars
- `taxonomy_manager.py` — safely adds/removes/renames labels and edits hierarchy mappings

## Initial setup

From inside the new project directory:

```bash
git init
uv init --bare
uv add torch torchvision transformers pillow tqdm accelerate sentencepiece protobuf pillow-heif
git add .
git commit -m "Initial caption and tag tools"
```

On macOS, the default `--device auto` should choose Metal/MPS when available.

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
  --recurse \
  --labels-file labels_curated_hierarchical.txt \
  --taxonomy-map labels_taxonomy_map.csv \
  --dry-run \
  --verbose

# Recompute tags even when the cache says the image was already classified
uv run python immich_caption_and_tag.py /path/to/library \
  --recurse \
  --labels-file labels_curated_hierarchical.txt \
  --taxonomy-map labels_taxonomy_map.csv \
  --force

# Skip caption creation and only run tagging
uv run python immich_caption_and_tag.py /path/to/library \
  --recurse \
  --labels-file labels_curated_hierarchical.txt \
  --taxonomy-map labels_taxonomy_map.csv \
  --skip-captioning
```

## Inspect the tags actually present in sidecars

```bash
uv run python tag_stats.py /Volumes/oldmacData/library/upload --recurse
```

With hierarchy validation against the current taxonomy:

```bash
uv run python tag_stats.py /Volumes/oldmacData/library/upload \
  --recurse \
  --taxonomy-map labels_taxonomy_map.csv \
  --top 50
```

Write CSV/JSON reports:

```bash
uv run python tag_stats.py /Volumes/oldmacData/library/upload \
  --recurse \
  --taxonomy-map labels_taxonomy_map.csv \
  --csv-prefix tag_stats \
  --json tag_stats_summary.json
```

## Maintain the candidate labels and hierarchy

Show a label:

```bash
uv run python taxonomy_manager.py show "boat"
```

Add a new candidate label and hierarchy path:

```bash
uv run python taxonomy_manager.py add "red phalarope" \
  --tags "Nature/Birds/Red phalarope" \
  --section "Nature"
```

Remove a candidate label from both files:

```bash
uv run python taxonomy_manager.py remove "red phalarope"
```

Replace the hierarchy path or paths for a label:

```bash
uv run python taxonomy_manager.py set-tags "boat" \
  --tags "Water/Marine/Boat|Transportation/Boat"
```

Add or remove one additional hierarchy path without replacing the others:

```bash
uv run python taxonomy_manager.py add-path "boat" --tag "Transportation/Boat"
uv run python taxonomy_manager.py remove-path "boat" --tag "Transportation/Boat"
```

Rename a label while preserving its hierarchy mapping:

```bash
uv run python taxonomy_manager.py rename "plane flying" "airplane flying"
```

Audit for duplicates and mismatches between the two source files:

```bash
uv run python taxonomy_manager.py audit
```

Every mutating `taxonomy_manager.py` command writes timestamped backups before editing the two source files.

## Suggested workflow

1. Add or refine candidate labels with `taxonomy_manager.py`.
2. Run `taxonomy_manager.py audit`.
3. Run the main caption/tagger on a small test folder.
4. Inspect results with `tag_stats.py`.
5. If the tags look good, run the main program against the full library.
6. Once the workflow is stable, schedule the main program once daily; because it skips existing captions and uses a cache for tagging, daily runs should mostly process newly added images.

## Daily automation later

After testing, the simplest automation is a daily run. A file-watcher can be added later, but a daily job is easier to trust and maintain because the program is restartable and already skips completed work.
