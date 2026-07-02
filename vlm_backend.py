"""Local VLM caption generation via Ollama."""
from __future__ import annotations

import base64
import json
import urllib.error
import urllib.request
from typing import Optional


DEFAULT_CAPTION_PROMPT = """\
Write one natural sentence describing this photo.
{people_clause}
If you see an animal or bird, identify the specific species (e.g. the exact bird or animal name, not just 'bird' or 'animal').
Start the sentence directly — do not begin with 'The image shows', 'A photo of', 'This is', or 'In this image'.\
"""


class OllamaVLM:
    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434",
        timeout: int = 120,
        temperature: float = 0.3,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.temperature = temperature

    def is_available(self) -> bool:
        try:
            req = urllib.request.Request(f"{self.base_url}/api/tags", method="GET")
            with urllib.request.urlopen(req, timeout=5):
                return True
        except Exception:
            return False

    def caption(
        self,
        image_bytes: bytes,
        people_names: list[str],
        prompt_template: str = DEFAULT_CAPTION_PROMPT,
    ) -> Optional[str]:
        """Generate a caption for the image.

        Injects known people names into the prompt so the VLM can use them.
        Returns None on failure — caller should keep the existing description.
        """
        b64 = base64.b64encode(image_bytes).decode()

        if people_names:
            names_str = ", ".join(people_names)
            people_clause = (
                f"The people in this photo are: {names_str}. "
                f"Include their name(s) naturally in your sentence — do NOT output just a name alone."
            )
        else:
            people_clause = ""

        prompt = prompt_template.format(people_clause=people_clause).strip()
        # Collapse blank line left by empty people_clause
        prompt = "\n".join(line for line in prompt.splitlines() if line.strip())

        body = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                    "images": [b64],
                }
            ],
            "stream": False,
            # Keep the model resident: the default 5m keep_alive made Ollama
            # unload + reload the ~5GB model whenever classification between
            # caption batches took longer than 5 minutes.
            "keep_alive": -1,
            "options": {"temperature": self.temperature},
        }

        data = json.dumps(body).encode()
        req = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                result = json.loads(resp.read())
            text = result.get("message", {}).get("content", "").strip()
            return text or None
        except urllib.error.HTTPError as exc:
            body_text = exc.read().decode(errors="replace") if exc.fp else ""
            raise RuntimeError(f"Ollama HTTP {exc.code}: {body_text}") from exc
        except Exception:
            return None
