# Immich Caption/Tagger — New System Setup

This guide documents how to install the combined Immich sidecar caption/tagging workflow on a new macOS system.

The workflow is:

1. install system tools with Homebrew,
2. create or clone the project directory,
3. install Python dependencies with `uv`,
4. configure optional API/auth tokens,
5. run a small test,
6. install the daily `launchd` job at 11:00 PM.

The main program writes XMP sidecars next to images. It does **not** directly modify Immich through the Immich API. Immich must be configured separately to read sidecars or rescan metadata.

---

## 1. Recommended folder location

Use a folder outside `~/Documents` to avoid macOS privacy/TCC issues with LaunchAgents.

Recommended location:

```bash
/Users/jberman/Projects/immich-tagger
```

Create it:

```bash
mkdir -p /Users/jberman/Projects
cd /Users/jberman/Projects
```

If this is a Git repository:

```bash
git clone <YOUR_REPO_URL> immich-tagger
cd immich-tagger
```

If you are copying files manually, create the folder and copy the project files into it:

```bash
mkdir -p /Users/jberman/Projects/immich-tagger
cd /Users/jberman/Projects/immich-tagger
```

Required project files:

```text
immich_caption_and_tag.py
labels_curated_hierarchical.txt
labels_taxonomy_map.csv
tag_stats.py
taxonomy_manager.py
caption_noun_candidates.py
README.md
pyproject.toml
uv.lock
```

---

## 2. Install Homebrew tools

Install Homebrew first if needed:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

Then install the system tools:

```bash
brew install git uv ffmpeg
```

Optional but useful:

```bash
brew install exiftool
```

What these are for:

- `git` — version control
- `uv` — Python project/dependency runner
- `ffmpeg` — fallback HEIC/HEIF decoding and future video frame extraction
- `exiftool` — optional sidecar/XMP inspection and cleanup

macOS includes `sips`, which the captioner can use as another HEIC/HEIF fallback.

---

## 3. Install Python dependencies

From the project folder:

```bash
cd /Users/jberman/Projects/immich-tagger
uv sync
```

If there is no usable `pyproject.toml` yet, initialize one:

```bash
uv init --bare
uv add torch torchvision transformers pillow tqdm accelerate sentencepiece protobuf pillow-heif
```

Optional, for better noun extraction from captions:

```bash
uv add spacy
uv run python -m spacy download en_core_web_sm
```

The tagger can run without spaCy. `caption_noun_candidates.py` falls back to a simpler heuristic extractor when spaCy is unavailable.

---

## 4. Optional Hugging Face token

The program can run without a Hugging Face token, but you may see this warning:

```text
Warning: You are sending unauthenticated requests to the HF Hub.
```

This is usually harmless once the models are cached. For higher rate limits, create a Hugging Face token and set `HF_TOKEN`.

Add it to your shell profile:

```bash
echo 'export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"' >> ~/.zshrc
source ~/.zshrc
```

For the daily launchd job, also add the token to the run script created below:

```bash
export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

Keep tokens out of Git. Do not commit API keys or `.env` files.

---

## 5. Confirm paths

Typical project path:

```bash
PROJECT_DIR="/Users/jberman/Projects/immich-tagger"
```

Typical Immich upload/library path:

```bash
LIBRARY_DIR="/Volumes/oldmacData/library/upload"
```

Confirm the library exists:

```bash
ls -ld /Volumes/oldmacData/library/upload
```

Confirm `uv` is visible:

```bash
which uv
```

In the working setup, this was:

```text
/Users/jberman/.local/bin/uv
```

but the launch script uses `uv` through `PATH`, so it does not need a hard-coded path.

---

## 6. First sanity checks

Audit the curated label files:

```bash
uv run python taxonomy_manager.py audit
```

Expected healthy output:

```text
duplicate labels in labels file: 0
duplicate labels in taxonomy CSV: 0
labels with no taxonomy row: 0
taxonomy rows missing from labels file: 0
empty taxonomy mappings: 0
labels with duplicate hierarchy paths: 0
```

Run a small test folder before touching the full library:

```bash
uv run python immich_caption_and_tag.py test_images \
  --recurse \
  --labels-file labels_curated_hierarchical.txt \
  --taxonomy-map labels_taxonomy_map.csv \
  --db-path .immich_auto_tagger_cache.sqlite3 \
  --verbose
```

Inspect the result:

```bash
uv run python tag_stats.py test_images \
  --recurse \
  --taxonomy-map labels_taxonomy_map.csv \
  --top 50
```

Healthy test results should show:

```text
valid_xml: 200
invalid_xml: 0
captioned_sidecars: 200
uncaptioned_sidecars: 0
sidecars_with_tags: close to the number of images
```

Some `caption/...` unmapped tags are expected. Those are deliberate last-resort fallback tags.

---

## 7. Run the full library manually

Normal daily/new-image run, without `--force`:

```bash
uv run python immich_caption_and_tag.py /Volumes/oldmacData/library/upload \
  --recurse \
  --labels-file labels_curated_hierarchical.txt \
  --taxonomy-map labels_taxonomy_map.csv \
  --db-path .immich_auto_tagger_cache.sqlite3 \
  --verbose
```

Use `--force` only after intentionally changing the label/taxonomy files and wanting the entire library re-evaluated:

```bash
uv run python immich_caption_and_tag.py /Volumes/oldmacData/library/upload \
  --recurse \
  --labels-file labels_curated_hierarchical.txt \
  --taxonomy-map labels_taxonomy_map.csv \
  --db-path .immich_auto_tagger_cache.sqlite3 \
  --force \
  --verbose
```

Check the full-library result:

```bash
uv run python tag_stats.py /Volumes/oldmacData/library/upload \
  --recurse \
  --taxonomy-map labels_taxonomy_map.csv \
  --top 100
```

---

## 8. Caption noun discovery workflow

Dry-run candidate discovery from captions:

```bash
uv run python caption_noun_candidates.py /Volumes/oldmacData/library/upload \
  --recurse \
  --labels-file labels_curated_hierarchical.txt \
  --taxonomy-map labels_taxonomy_map.csv \
  --min-count 5 \
  --min-docs 3 \
  --max-doc-pct 10 \
  --include-phrases \
  --top 200 \
  --csv noun_candidates_full.csv
```

Apply reviewed candidates:

```bash
uv run python caption_noun_candidates.py /Volumes/oldmacData/library/upload \
  --recurse \
  --labels-file labels_curated_hierarchical.txt \
  --taxonomy-map labels_taxonomy_map.csv \
  --min-count 5 \
  --min-docs 3 \
  --max-doc-pct 10 \
  --include-phrases \
  --limit-add 50 \
  --apply
```

Then audit and commit:

```bash
uv run python taxonomy_manager.py audit
git add labels_curated_hierarchical.txt labels_taxonomy_map.csv
git commit -m "Add noun candidates from caption analysis"
```

After changing labels/taxonomy, run one full `--force` tagging pass.

---

## 9. Install the daily launchd job

The daily job should run **without `--force`**. It will caption and tag newly added images and skip already-completed work.

Create the logs directory:

```bash
cd /Users/jberman/Projects/immich-tagger
mkdir -p logs
```

Create the daily run script:

```bash
cat > /Users/jberman/Projects/immich-tagger/run_daily_new_images.sh <<'EOF'
#!/bin/zsh
set -euo pipefail

cd /Users/jberman/Projects/immich-tagger

export PATH="/Users/jberman/.local/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin"

# Optional Hugging Face token. Uncomment and set if desired.
# export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

echo "===== $(date) Starting daily Immich caption/tag run ====="
echo "Working directory: $(pwd)"
echo "uv path: $(command -v uv || true)"

uv run python immich_caption_and_tag.py /Volumes/oldmacData/library/upload \
  --recurse \
  --labels-file labels_curated_hierarchical.txt \
  --taxonomy-map labels_taxonomy_map.csv \
  --db-path .immich_auto_tagger_cache.sqlite3

echo "===== $(date) Finished daily Immich caption/tag run ====="
EOF

chmod 755 /Users/jberman/Projects/immich-tagger/run_daily_new_images.sh
```

Test the script manually:

```bash
/bin/zsh /Users/jberman/Projects/immich-tagger/run_daily_new_images.sh
```

Create the LaunchAgent:

```bash
cat > ~/Library/LaunchAgents/net.joelberman.immich-caption-tagger.plist <<'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
 "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
  <dict>
    <key>Label</key>
    <string>net.joelberman.immich-caption-tagger</string>

    <key>ProgramArguments</key>
    <array>
      <string>/bin/zsh</string>
      <string>/Users/jberman/Projects/immich-tagger/run_daily_new_images.sh</string>
    </array>

    <key>WorkingDirectory</key>
    <string>/Users/jberman/Projects/immich-tagger</string>

    <key>StartCalendarInterval</key>
    <dict>
      <key>Hour</key>
      <integer>23</integer>
      <key>Minute</key>
      <integer>0</integer>
    </dict>

    <key>StandardOutPath</key>
    <string>/Users/jberman/Projects/immich-tagger/logs/daily_tagger.out.log</string>

    <key>StandardErrorPath</key>
    <string>/Users/jberman/Projects/immich-tagger/logs/daily_tagger.err.log</string>

    <key>RunAtLoad</key>
    <false/>
  </dict>
</plist>
EOF
```

Load it:

```bash
launchctl bootout gui/$(id -u) ~/Library/LaunchAgents/net.joelberman.immich-caption-tagger.plist 2>/dev/null || true
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/net.joelberman.immich-caption-tagger.plist
launchctl enable gui/$(id -u)/net.joelberman.immich-caption-tagger
```

Kick it once now:

```bash
cd /Users/jberman/Projects/immich-tagger

: > logs/daily_tagger.out.log
: > logs/daily_tagger.err.log

launchctl kickstart -k gui/$(id -u)/net.joelberman.immich-caption-tagger
```

Check logs:

```bash
tail -n 100 logs/daily_tagger.out.log
tail -n 100 logs/daily_tagger.err.log
```

A healthy run starts like this:

```text
===== Sat May ... Starting daily Immich caption/tag run =====
Working directory: /Users/jberman/Projects/immich-tagger
uv path: /Users/jberman/.local/bin/uv
```

The job will run daily at 11:00 PM.

---

## 10. launchd troubleshooting

Check job status:

```bash
launchctl print gui/$(id -u)/net.joelberman.immich-caption-tagger
```

Clear logs:

```bash
: > /Users/jberman/Projects/immich-tagger/logs/daily_tagger.out.log
: > /Users/jberman/Projects/immich-tagger/logs/daily_tagger.err.log
```

Unload/reload:

```bash
launchctl bootout gui/$(id -u) ~/Library/LaunchAgents/net.joelberman.immich-caption-tagger.plist 2>/dev/null || true
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/net.joelberman.immich-caption-tagger.plist
launchctl enable gui/$(id -u)/net.joelberman.immich-caption-tagger
```

If launchd says it cannot open a script inside `~/Documents`, move the project to `~/Projects`. This setup intentionally uses:

```text
/Users/jberman/Projects/immich-tagger
```

to avoid that issue.

---

## 11. Normal operating rules

For newly added images:

```bash
uv run python immich_caption_and_tag.py /Volumes/oldmacData/library/upload \
  --recurse \
  --labels-file labels_curated_hierarchical.txt \
  --taxonomy-map labels_taxonomy_map.csv \
  --db-path .immich_auto_tagger_cache.sqlite3
```

Do **not** use `--force` for routine daily runs.

Use `--force` when:

- the curated label file changed,
- the taxonomy map changed,
- the model thresholds changed,
- you intentionally want to replace existing `ai:` tags.

After any major change:

```bash
uv run python tag_stats.py /Volumes/oldmacData/library/upload \
  --recurse \
  --taxonomy-map labels_taxonomy_map.csv \
  --top 100
```

Commit stable changes:

```bash
git status
git add .
git commit -m "Update caption/tag workflow"
```
