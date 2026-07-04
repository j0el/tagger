#!/usr/bin/env python3
"""One-line health snapshot: thermal, memory, CPU/GPU pressure, tagger progress.

Intended to run hourly from cron, appending to logs/health.log. Reads only
/proc and /sys plus the tagger cache DB — no third-party dependencies.
"""
from __future__ import annotations

import glob
import sqlite3
import time
from datetime import datetime
from pathlib import Path

DB_PATH = Path(__file__).parent / ".immich_tagger_v2_cache.sqlite3"


def hwmon_temp(chip: str) -> float | None:
    """First temp*_input (°C) of the hwmon chip with the given name."""
    for hw in glob.glob("/sys/class/hwmon/hwmon*"):
        try:
            if Path(hw, "name").read_text().strip() != chip:
                continue
            inputs = sorted(glob.glob(f"{hw}/temp*_input"))
            if inputs:
                return int(Path(inputs[0]).read_text()) / 1000.0
        except OSError:
            continue
    return None


def psi(resource: str) -> dict[str, str]:
    """avg60 values from /proc/pressure/<resource>, keyed 'some'/'full'."""
    out: dict[str, str] = {}
    try:
        for line in Path(f"/proc/pressure/{resource}").read_text().splitlines():
            kind, rest = line.split(" ", 1)
            fields = dict(kv.split("=") for kv in rest.split())
            out[kind] = fields["avg60"]
    except OSError:
        pass
    return out


def meminfo() -> dict[str, int]:
    """Selected /proc/meminfo fields, in kB."""
    out: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, val = line.split(":", 1)
        if key in ("MemTotal", "MemAvailable", "SwapTotal", "SwapFree"):
            out[key] = int(val.strip().split()[0])
    return out


def gpu_busy() -> int | None:
    for p in glob.glob("/sys/class/drm/card*/device/gpu_busy_percent"):
        try:
            return int(Path(p).read_text())
        except OSError:
            continue
    return None


def tagger_counts() -> tuple[int | None, int | None]:
    """(total cached assets, assets tagged in the last hour)."""
    if not DB_PATH.exists():
        return None, None
    try:
        conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
        total = conn.execute("SELECT COUNT(*) FROM asset_cache").fetchone()[0]
        hour = conn.execute(
            "SELECT COUNT(*) FROM asset_cache WHERE tagged_at > ?",
            (time.time() - 3600,),
        ).fetchone()[0]
        conn.close()
        return total, hour
    except sqlite3.Error:
        return None, None


def main() -> None:
    parts: list[str] = [datetime.now().strftime("%Y-%m-%dT%H:%M:%S")]

    for label, chip in (("cpu_temp", "k10temp"), ("gpu_temp", "amdgpu"), ("nvme_temp", "nvme")):
        t = hwmon_temp(chip)
        parts.append(f"{label}={t:.0f}C" if t is not None else f"{label}=n/a")

    mem = meminfo()
    used_gb = (mem["MemTotal"] - mem["MemAvailable"]) / 1048576
    total_gb = mem["MemTotal"] / 1048576
    swap_gb = (mem["SwapTotal"] - mem["SwapFree"]) / 1048576
    parts.append(f"mem={used_gb:.1f}/{total_gb:.0f}G")
    parts.append(f"swap={swap_gb:.1f}G")

    load1 = Path("/proc/loadavg").read_text().split()[0]
    parts.append(f"load1={load1}")

    for res in ("cpu", "memory", "io"):
        vals = psi(res)
        parts.append(f"psi_{res}={vals.get('some', 'n/a')}/{vals.get('full', '-')}")

    busy = gpu_busy()
    parts.append(f"gpu_busy={busy}%" if busy is not None else "gpu_busy=n/a")

    total, hour = tagger_counts()
    parts.append(f"imgs_total={total if total is not None else 'n/a'}")
    parts.append(f"imgs_last_hour={hour if hour is not None else 'n/a'}")

    print(" ".join(parts))


if __name__ == "__main__":
    main()
