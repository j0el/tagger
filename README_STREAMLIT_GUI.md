# Streamlit GUI for Immich Caption + Tag Tools

This is a local browser GUI for the Immich caption/tag sidecar tools.

## Files

Put `immich_tagger_streamlit_app.py` in the same directory as:

- `immich_caption_and_tag.py`
- `tag_stats.py`
- `caption_noun_candidates.py`
- `taxonomy_manager.py`
- `labels_curated_hierarchical.txt`
- `labels_taxonomy_map.csv`

## Install

From the project folder:

```bash
uv add streamlit
```

Your existing project dependencies should already include Pillow and pillow-heif. If HEIC previews do not display, also run:

```bash
uv add pillow-heif
```

## Run

```bash
uv run streamlit run immich_tagger_streamlit_app.py
```

The app opens in your browser.

## What it does

The app has five tabs:

1. **Run caption/tagger** — wraps `immich_caption_and_tag.py` with checkboxes and parameter entries.
2. **Boolean tag search** — scans `.xmp` sidecars and searches `dc:subject` tags.
3. **Tag statistics** — wraps `tag_stats.py` and can write CSV/JSON reports.
4. **Caption noun candidates** — wraps `caption_noun_candidates.py`.
5. **Taxonomy manager** — wraps `taxonomy_manager.py`.

## Boolean search examples

```text
food AND NOT people
(table OR bed) AND dog
"red phalarope" AND NOT blurry
ai:water/marine/boat OR kayak
```

Operator precedence is:

1. `NOT`
2. `AND`
3. `OR`

Parentheses are supported. Adjacent terms are treated as `AND`, so `dog beach` means `dog AND beach`.

For multi-word tags, use quotes.

## Preview size

The search tab uses a default preview width of 384 pixels, which is approximately 4 inches at 96 DPI in a browser.

## Notes

- The GUI runs the existing scripts as subprocesses. It does not import or refactor them.
- The first search over a large library may take a little while because it scans XMP files. The result is cached for 30 minutes.
- Use the **Clear cached index** button after changing tags if you want the search tab to rescan immediately.
- Mutating taxonomy actions still rely on `taxonomy_manager.py`, which writes timestamped backups before editing files.
