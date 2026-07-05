# Immich Auto-Tagger (v2)

Automatic AI captions and hierarchical tags for a self-hosted [Immich](https://immich.app)
photo library, written entirely through the Immich HTTP API — no sidecar files, no direct
filesystem access to the library.

For each asset the pipeline:

1. downloads the Immich preview thumbnail,
2. runs SigLIP zero-shot classification against a curated list of ~330 candidate labels,
3. maps the winning labels into a hierarchical taxonomy and assigns them as `ai:`-prefixed
   Immich tags (creating parent tags as needed, so Immich shows a browsable tree),
4. generates a one-sentence caption with a local Qwen2.5-VL vision-language model — with the
   names of people Immich face recognition has already identified injected into the prompt,
   so captions say "Joel and Sam at the beach" instead of "two people at the beach",
5. writes the tags and the caption (asset description) back via the API.

Human-created tags are never touched: incremental runs only *add* tags, and reprocess runs
remove/replace only tags whose value starts with `ai:`. Existing descriptions are kept unless
`--reprocess-captions` is passed.

Results are checkpointed in a local SQLite cache keyed by a **model signature** (a hash of the
models, labels, taxonomy, and thresholds), so interrupted runs resume where they left off and
repeat runs are cheap.

## Master directory

Everything current lives at the top level. Each Python file carries a docstring describing its role.

### Core pipeline

| File | Purpose |
| --- | --- |
| `immich_caption_and_tag_v2.py` | Main program: asset selection, SigLIP tagging, VLM captioning, API writes, SQLite cache. The canonical entry point. |
| `immich_api.py` | Immich HTTP API client: asset search, thumbnails, tag CRUD (with hierarchy creation), description updates, retry logic. Also provides `.env` loading. |
| `vlm_backend.py` | VLM caption client. Speaks either the Ollama API or the OpenAI-compatible API (used with llama-server). Holds the default caption prompt. |
| `labels_curated_hierarchical.txt` | The 330 curated candidate labels SigLIP scores each image against. Grouped by commented sections (People, Events, Nature, ...). |
| `labels_taxonomy_map.csv` | Maps each candidate label to one or more hierarchical tag paths (e.g. `boat → Water/Marine/Boat|Transportation/Boat`). |

### Run scripts and monitoring

| File | Purpose |
| --- | --- |
| `run_daily_new_images.sh` | Incremental run: processes assets created since the last run. This is the nightly cron job. |
| `run_reprocess_all.sh` | Full-library job: incremental pass first, then a `--reprocess-all --reprocess-captions` backfill. flock-guarded so overlapping invocations no-op. Used for the initial backfill and for re-tagging after taxonomy/model changes. |
| `health_snapshot.py` | One-line hourly system snapshot (temps, memory, PSI pressure, GPU busy, images tagged total/last-hour), cron-run at :05, appends to `logs/health.log`. Stdlib-only. |

### Utilities

| File | Purpose |
| --- | --- |
| `demo_comparison.py` | Compares existing Immich captions/tags vs what the pipeline would produce for N random assets. Useful before committing to a full reprocess. |
| `timing_test.py` | VLM benchmark: captions the same N images and reports seconds/image and tokens/second. Useful when tuning server flags or trying a different model. |

### Project files

| File | Purpose |
| --- | --- |
| `pyproject.toml`, `uv.lock` | Python dependencies, managed with `uv`. |
| `.env` | `IMMICH_URL` and `IMMICH_API_KEY` (not committed). |
| `.immich_tagger_v2_cache.sqlite3` | SQLite cache: per-asset results + the incremental last-run bookmark (not committed). |
| `logs/` | Run logs (`daily_new_images.log`, `reprocess_all.log`, `health.log`) (not committed). |

The earlier **v1 sidecar-based toolkit is gone**: `immich_caption_and_tag.py`, `tag_stats.py`,
`taxonomy_manager.py`, `caption_noun_candidates.py`, `mirror_dc_subject_to_immich_xmp_tags.py`,
`remove_sidecar_tags_v2.py`, the Streamlit GUI, and their READMEs no longer exist. Any untracked
leftovers in the working directory (`tagger_caption_rescue*.py`, `tag_stats_*.csv/json`,
`demo_report*.txt`, `.immich_auto_tagger_cache.sqlite3`, `logs/daily_tagger.*.log`) are v1-era
remnants, not part of the pipeline.

## Installation

Target system: Ubuntu on an AMD Ryzen AI 9 HX 370 with a Radeon 890M iGPU (Vulkan via Mesa).
Nothing is AMD-specific except the VLM server setup — SigLIP runs on CPU, and any
OpenAI-compatible or Ollama VLM endpoint works for captions.

### 1. Python environment

Requires [uv](https://docs.astral.sh/uv/) and Python ≥ 3.14 (uv installs it automatically):

```bash
git clone <this-repo> immich-tagger
cd immich-tagger
uv sync
```

The SigLIP model (~3.5 GB) downloads from Hugging Face automatically on first run.

### 2. Immich credentials

Create `.env` in the project root (an Immich API key is created under
Account Settings → API Keys; it needs asset read/update and tag permissions):

```bash
IMMICH_URL=http://localhost:2283
IMMICH_API_KEY=<your-key>
```

The run scripts `source .env` and export both variables; the Python program reads them from the
environment (and falls back to reading `.env` itself).

### 3. VLM caption server (llama-server + Vulkan)

Captions use **Qwen2.5-VL 7B** served by **llama.cpp's `llama-server`** with the Vulkan backend,
so both the language model and the vision encoder run on the iGPU. (Plain Ollama also works via
`--vlm-api ollama`, but Ollama refuses to offload the qwen2.5-vl vision encoder on shared-memory
iGPUs, leaving CLIP on CPU at ~14 s/image — llama-server with Vulkan is ~2× faster end to end.)

Install Ollama once just to download the model weights (a single GGUF containing both the LLM and
the vision tensors), then serve that blob with the `llama-server` binary Ollama ships:

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen2.5vl:7b
# Find the model blob path (the "vnd.ollama.image.model" digest):
python3 -m json.tool /usr/share/ollama/.ollama/models/manifests/registry.ollama.ai/library/qwen2.5vl/7b
```

Systemd unit (`/etc/systemd/system/llama-vlm.service`) as currently deployed — substitute the
blob digest from the manifest above:

```ini
[Unit]
Description=llama.cpp server for immich-tagger VLM captions (Vulkan, GPU vision encoder)
After=network-online.target

[Service]
User=ollama
Group=ollama
Environment="LD_LIBRARY_PATH=/usr/local/lib/ollama"
Environment="GGML_BACKEND_PATH=/usr/local/lib/ollama/vulkan/libggml-vulkan.so"
ExecStart=/usr/local/lib/ollama/llama-server \
  --model /usr/share/ollama/.ollama/models/blobs/sha256-a99b7f834d754b88f122d865f32758ba9f0994a83f8363df2c1e71c17605a025 \
  --mmproj /usr/share/ollama/.ollama/models/blobs/sha256-a99b7f834d754b88f122d865f32758ba9f0994a83f8363df2c1e71c17605a025 \
  --port 8082 --host 127.0.0.1 --no-webui --offline \
  -c 8192 -np 2 -ngl 99 \
  --image-min-tokens 1024 --flash-attn auto -b 1024 -ub 1024
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

Server flags that matter:

- `-np 2` — two parallel slots; must match the pipeline's `--vlm-workers` (default 2).
- `-c 8192` — context split across the slots; each image costs ~1.2k tokens.
- `-ngl 99` — offload all layers to the GPU (Vulkan).
- `--image-min-tokens 1024` — floor on vision tokens so small images still caption well.
- `--mmproj` points at the same GGUF as `--model` because the Ollama qwen2.5vl blob bundles the
  vision projector in the one file.

```bash
sudo systemctl enable --now llama-vlm
curl http://localhost:8082/health   # → {"status":"ok"}
```

## Running

### Incremental (the normal case)

```bash
./run_daily_new_images.sh
```

Processes assets created since the stored last-run date, then advances that bookmark. First ever
run (no bookmark) processes the whole library. Pass extra flags through, e.g.
`./run_daily_new_images.sh --dry-run --verbose`.

### Full reprocess

```bash
./run_reprocess_all.sh
```

Incremental pass first, then `--reprocess-all --reprocess-captions` over every asset. Assets
already processed under the current model signature are skipped, so this is resumable and cheap
to re-run; after a taxonomy or model change it re-does everything (see
[Updating and re-tagging](#updating-and-re-tagging)). Logs to `logs/reprocess_all.log`.

### Manual invocations

```bash
# Preview what would change, writing nothing
uv run python immich_caption_and_tag_v2.py \
  --labels-file labels_curated_hierarchical.txt \
  --taxonomy-map labels_taxonomy_map.csv \
  --db-path .immich_tagger_v2_cache.sqlite3 \
  --vlm-url http://localhost:8082 --vlm-api openai \
  --dry-run --verbose --limit 10

# Test the pipeline on one specific asset (repeatable flag)
uv run python immich_caption_and_tag_v2.py \
  --labels-file labels_curated_hierarchical.txt \
  --taxonomy-map labels_taxonomy_map.csv \
  --vlm-url http://localhost:8082 --vlm-api openai \
  --asset-id <IMMICH-ASSET-UUID> --force --verbose
```

To find an asset ID, search `POST /api/search/metadata` by `originalFileName` — the
UUID-looking segment in the on-disk storage path is a checksum shard, *not* the asset ID.

### Scheduling (current crontab)

```cron
# Nightly incremental run, 06:00 UTC (= 11 PM Pacific). flock makes it no-op if a
# reprocess/backfill run still holds the lock.
0 6 * * * flock -n /tmp/immich_reprocess_all.lock /bin/bash /home/jberman/Projects/immich-tagger/run_daily_new_images.sh >> /home/jberman/Projects/immich-tagger/logs/daily_new_images.log 2>&1

# Resume the full backfill after a reboot. Once the backfill is complete this is a
# cheap skip-pass (everything cached); kept as a safety net.
@reboot sleep 60 && /bin/bash /home/jberman/Projects/immich-tagger/run_reprocess_all.sh >> /home/jberman/Projects/immich-tagger/logs/reprocess_all.log 2>&1

# Hourly one-line health snapshot
5 * * * * /usr/bin/python3 /home/jberman/Projects/immich-tagger/health_snapshot.py >> /home/jberman/Projects/immich-tagger/logs/health.log 2>&1
```

The crontab lives only on the host (`crontab -l`), not in this repo — re-add these lines when
rebuilding the machine. Reboot survival also requires `cron` and `docker` enabled at boot and
`restart: always` on the Immich containers (the default).

## Parameter reference

All flags of `immich_caption_and_tag_v2.py`, with defaults:

### Data sources

| Flag | Default | Meaning |
| --- | --- | --- |
| `--labels-file` | (required) | Newline-separated candidate labels; `#` lines are comments. |
| `--taxonomy-map` | none | CSV mapping label → hierarchical tag path(s). |
| `--db-path` | `.immich_tagger_v2_cache.sqlite3` | SQLite cache location. |

### Asset selection

| Flag | Default | Meaning |
| --- | --- | --- |
| *(none)* | — | Incremental: assets created after the stored last-run date. |
| `--reprocess-all` | off | Every asset in the library; also removes stale `ai:` tags before re-assigning. Respects the model-signature cache. |
| `--since ISO-DATE` | — | Assets created after this date (overrides the stored bookmark; doesn't update it). |
| `--asset-id UUID` | — | Only this asset (repeatable). For testing; doesn't update the bookmark. |
| `--reprocess-captions` | off | Regenerate captions even when a description already exists. |
| `--force` | off | Ignore the cache and re-classify anyway. |
| `--limit N` | 0 (no limit) | Stop after N assets. |
| `--stop-at HH:MM` | none | Stop gracefully at this local time (for overnight windows). |

### Models

| Flag | Default | Meaning |
| --- | --- | --- |
| `--device` | `auto` | `auto`/`cuda`/`cpu` for SigLIP (auto → CPU on this box; torch build has no ROCm). |
| `--zero-shot-model` | `google/siglip-so400m-patch14-384` | HF model for zero-shot tagging. |
| `--vlm-model` | `qwen2.5vl:7b` | Model name sent to the VLM server (llama-server ignores it — it serves one model — but Ollama uses it). |
| `--vlm-url` | `http://localhost:11434` | VLM server base URL. **The run scripts pass `http://localhost:8082`** (llama-server). |
| `--vlm-api` | `ollama` | `ollama` (`/api/chat`) or `openai` (`/v1/chat/completions`). **The run scripts pass `openai`.** |
| `--vlm-timeout` | 120 | Seconds to wait per caption. |
| `--vlm-workers` | 2 | Concurrent caption requests; match the server's parallel slots (`-np`). 1 = serial. |
| `--caption-prompt` | built-in | Override the caption prompt template; `{people_clause}` is replaced with the names sentence. |
| `--skip-captioning` | off | Tags only, no VLM. |

### Classification thresholds

| Flag | Default | Meaning |
| --- | --- | --- |
| `--batch-size` | 8 | Images per SigLIP batch. |
| `--top-k` | 6 | Max labels kept per image (primary pass). |
| `--threshold` | 0.32 | Minimum sigmoid score (primary pass). |
| `--relative-threshold` | 0.70 | A label must also score ≥ 70% of the image's best label. |
| `--fallback-threshold` | 0.20 | Looser floor used only when the primary pass keeps nothing. |
| `--fallback-relative-threshold` | 0.50 | Relative floor for the fallback pass. |
| `--fallback-top-k` | 3 | Max labels from the fallback pass. |
| `--max-side` | 768 | Long-side pixel cap for images fed to SigLIP. |
| `--vlm-max-side` | 1024 | Long-side cap for images sent to the VLM (qwen2.5-vl cost grows linearly with pixels; full 1440 px previews caption 3–4× slower). |

### Output

| Flag | Default | Meaning |
| --- | --- | --- |
| `--dry-run` | off | Print planned tags/captions; write nothing to Immich. |
| `--verbose` | off | Per-asset detail including top-5 raw scores. |
| `--no-caption-tags` | off | Disable the caption-noun fallback tagging. |

## Models, parameters, and prompts

### Tagging: SigLIP zero-shot

- Model: **`google/siglip-so400m-patch14-384`**, run on CPU via `transformers`.
- Each of the 330 candidate labels is embedded once at startup with the template
  **"This is a photo of {label}."**; per image, scores are
  `sigmoid(img_emb · txt_emb × logit_scale + logit_bias)` — identical output to the HF
  zero-shot pipeline, but ~4× faster because text embeddings aren't recomputed per batch.
- A label is kept if its score ≥ `0.32` **and** ≥ 70% of that image's best score, up to 6
  labels. If nothing survives, a fallback pass over the *same* scores applies looser floors
  (0.20 / 50%, up to 3 labels) so most images get at least something.

### Captioning: Qwen2.5-VL 7B via llama-server

- Served on `localhost:8082` by the `llama-vlm` systemd service (Vulkan, fully on the
  Radeon 890M iGPU — including the vision encoder, which is why llama-server is used instead
  of Ollama).
- Request parameters: OpenAI chat-completions format, `temperature 0.3`, `max_tokens 120`,
  image resized to ≤ 1024 px long side and sent as base64 JPEG (quality 90).
- The default prompt (in `vlm_backend.py`):

  ```text
  Write one natural sentence describing this photo.
  {people_clause}
  If you see an animal or bird, identify the specific species (e.g. the exact bird or animal name, not just 'bird' or 'animal').
  Start the sentence directly — do not begin with 'The image shows', 'A photo of', 'This is', or 'In this image'.
  ```

  When Immich face recognition has named people in the photo, `{people_clause}` becomes:

  ```text
  The people in this photo are: <name1>, <name2>. Include their name(s) naturally in your
  sentence — do NOT output just a name alone.
  ```

  Otherwise the line is dropped. Override the whole template with `--caption-prompt`.

### Pipeline concurrency

Three stages overlap so the iGPU stays busy: the main thread fetches thumbnails (4-way pooled)
and classifies batch N+1 on CPU while `--vlm-workers` caption threads keep llama-server's slots
fed with batch N, and a single writer thread does the Immich API writes and cache commits.
Bounded queues provide backpressure. Observed throughput during the 2026-07 backfill:
~165 s per 8-image batch (captioning is the bottleneck; SigLIP hides under it).

## The tag hierarchy: how it's created and used

Two files define it:

1. **`labels_curated_hierarchical.txt`** — the flat list of ~330 labels SigLIP can "see"
   (e.g. `person`, `birthday party`, `boat`). Organized in commented sections; only
   non-comment lines are used. This list is what the classifier scores — a concept absent
   here can never be tagged by the primary path.
2. **`labels_taxonomy_map.csv`** — `label,tags` rows mapping each label to one or more
   hierarchical paths, multiple paths separated by `|` (or `;`):

   ```csv
   label,tags
   boat,Water/Marine/Boat|Transportation/Boat
   birthday party,Events/Birthday
   ```

   A label with no mapping falls through as itself (normalized) — so unmapped labels still
   produce a flat tag rather than being dropped.

At write time every mapped path is normalized (lowercase, cleaned punctuation) and prefixed
with **`ai:`**, e.g. `ai:water/marine/boat`. `ImmichClient.ensure_tag()` then walks the path
and creates each missing ancestor as a real Immich tag with the correct `parentId`
(`ai:water` → `ai:water/marine` → `ai:water/marine/boat`), so the Immich UI shows a browsable
tree under the single `ai:` root. Tags are cached in memory per run, so existing tags cost
nothing.

The `ai:` prefix is the safety boundary: it's how the pipeline distinguishes its own tags from
human ones. Reprocess runs delete only `ai:*` tags before re-assigning; anything else is
untouchable.

**Caption-noun fallback:** if zero-shot produced no tags at all for an image but a caption
exists, the caption's words (crudely stemmed: `hiking → hike`, `dogs → dog`) are matched
against the taxonomy labels, and up to 5 matching labels' hierarchy tags are assigned. Disable
with `--no-caption-tags`.

## How captioning works

1. Only assets with **no existing description** are captioned (or all selected assets when
   `--reprocess-captions` is set) — human-written descriptions are never overwritten in
   normal runs.
2. The preview thumbnail is resized to ≤ 1024 px and sent to llama-server with the prompt
   above; named people from Immich face recognition are injected via `{people_clause}`.
3. A successful caption is written to the asset's description through
   `PUT /api/assets/{id}`. On VLM failure the existing description is kept and the asset is
   still tagged — captions are best-effort.
4. If the VLM server is unreachable at startup, the run continues tags-only with a warning.

## Updating and re-tagging

The cache stores a **model signature** per asset — a hash over the zero-shot model, VLM model,
the full label list, the full taxonomy map, and all thresholds. Change any of those and every
asset's cache entry no longer matches, so the next `--reprocess-all` run redoes everything;
unchanged configuration means `--reprocess-all` skips everything already done (that's also the
resume mechanism).

Typical workflows:

```bash
# 1. Edit the taxonomy (add a label, change a path) — then test on a handful of assets:
uv run python immich_caption_and_tag_v2.py \
  --labels-file labels_curated_hierarchical.txt --taxonomy-map labels_taxonomy_map.csv \
  --vlm-url http://localhost:8082 --vlm-api openai \
  --asset-id <uuid> --force --dry-run --verbose

# 2. Happy? Re-tag the whole library (tags only — keeps all existing captions, much faster):
nohup uv run python immich_caption_and_tag_v2.py \
  --labels-file labels_curated_hierarchical.txt --taxonomy-map labels_taxonomy_map.csv \
  --db-path .immich_tagger_v2_cache.sqlite3 \
  --reprocess-all --skip-captioning \
  >> logs/reprocess_all.log 2>&1 &

# 3. Or full re-tag + re-caption (days, not hours, for a ~19k-asset library):
nohup ./run_reprocess_all.sh &
```

Notes:

- `--reprocess-all` first strips the asset's existing `ai:` tags, then assigns the new set, so
  removed/renamed taxonomy paths disappear cleanly. Incremental runs never remove tags.
- Note that `--skip-captioning` is part of the model signature, so workflow 2 and workflow 3
  maintain separate cache generations — expected, just don't be surprised by the skip counts.
- Old parent tags that no longer have any children are *not* garbage-collected in Immich;
  prune those manually in the Immich UI if a rename leaves empty branches behind.
- New labels only take effect if SigLIP can plausibly score them — check with
  `demo_comparison.py` or `--asset-id ... --verbose` (shows the top-5 raw scores) before
  trusting a new label.
- Adding a label to `labels_curated_hierarchical.txt` without a row in
  `labels_taxonomy_map.csv` yields a flat `ai:<label>` tag; add the CSV row to place it in
  the hierarchy.

## Monitoring

- `tail -f logs/reprocess_all.log` / `logs/daily_new_images.log` — run output, per-run
  summary line: `Done. assets=... written=... skipped_cached=... captions=... errors=...`.
- `logs/health.log` — hourly one-liners with temps, memory, PSI, `gpu_busy`, and
  `imgs_total` / `imgs_last_hour` straight from the cache DB. If `imgs_last_hour=0` while a
  run should be active, something is stuck.
- `sqlite3 .immich_tagger_v2_cache.sqlite3 "SELECT COUNT(*) FROM asset_cache"` — total
  processed assets.
- `journalctl -u llama-vlm -f` — VLM server logs; `curl localhost:8082/health` — liveness.
