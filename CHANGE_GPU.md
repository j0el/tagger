# Switching to a machine with an NVIDIA or Intel GPU

This pipeline has exactly two GPU-dependent pieces: the `llama-vlm` systemd service
(VLM captions) and the SigLIP zero-shot model in `immich_caption_and_tag_v2.py`.
Everything else (run scripts, Immich API code, cache, cron) is GPU-agnostic.

## 1. VLM captions — `llama-vlm` systemd service

The service (`/etc/systemd/system/llama-vlm.service`) selects its GPU backend with a
single line:

```
Environment="GGML_BACKEND_PATH=/usr/local/lib/ollama/vulkan/libggml-vulkan.so"
```

Ollama already ships the alternative backends in `/usr/local/lib/ollama/`:
`cuda_v12`, `cuda_v13`, `rocm_v7_2`, `vulkan`.

### NVIDIA

Point `GGML_BACKEND_PATH` at the CUDA backend instead:

```
Environment="GGML_BACKEND_PATH=/usr/local/lib/ollama/cuda_v13/libggml-cuda.so"
```

(Use `cuda_v12` if the installed driver is older.) Then:

```sh
sudo systemctl daemon-reload && sudo systemctl restart llama-vlm
curl localhost:8082/health
```

Everything else — model blob paths, `-ngl 99`, port 8082, the run scripts — stays
identical. Vulkan would also work on NVIDIA's driver, but the CUDA backend is
meaningfully faster for the vision encoder.

### Intel (Arc / iGPU)

Keep Vulkan as-is — it runs on Intel's Mesa (ANV) driver. The only requirement is
that Vulkan drivers are installed (`mesa-vulkan-drivers`). The theoretically faster
SYCL backend is not bundled by Ollama; it would require building llama.cpp with
oneAPI yourself. Not worth it unless captioning becomes the bottleneck again.

## 2. SigLIP zero-shot tagging — PyTorch in `immich_caption_and_tag_v2.py`

SigLIP currently runs on CPU because torch has no ROCm build in this environment
(`--device auto` → cpu on the Radeon machine).

### NVIDIA

**Zero changes.** The locked torch is already the CUDA build (`2.11.0+cu130`), and
`choose_device("auto")` (`immich_caption_and_tag_v2.py:217`) already picks `cuda`
when available. SigLIP silently moves from CPU to GPU on first run. This is where
the biggest speedup lands, since SigLIP is forced onto CPU today.

Verify with:

```sh
uv run python -c "import torch; print(torch.cuda.is_available())"
```

If the driver is too old for CUDA 13, repin torch to the cu12x wheels.

### Intel

Torch's XPU backend needs the XPU wheels (`--index-url
https://download.pytorch.org/whl/xpu` in pyproject/uv config), plus three small
code edits:

1. Add `"xpu"` to the `--device` choices (line 88).
2. Check `torch.xpu.is_available()` in `choose_device()` (line 217).
3. Mirror the `torch.cuda.empty_cache()` call for xpu (line ~783).

Alternatively, skip all of that and leave SigLIP on CPU as today — it works, just
slower.

## 3. Housekeeping

- The unit's `Description` says "Vulkan" and CLAUDE.md/README state "SigLIP on CPU
  is expected, not a bug" — both become stale on NVIDIA; update them.
- The GPU is the small part of a machine move. Also migrate:
  - `.env` (`IMMICH_URL` / `IMMICH_API_KEY` — never committed)
  - `.immich_tagger_v2_cache.sqlite3` (or eat a multi-day full reprocess of ~19k assets)
  - the Ollama model blobs the service points at
    (`/usr/share/ollama/.ollama/models/blobs/...`)
  - the host crontab entries (`crontab -l`): daily incremental, `@reboot` backfill,
    hourly `health_snapshot.py`
