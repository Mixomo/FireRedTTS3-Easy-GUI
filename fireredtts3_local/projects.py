from __future__ import annotations

import json
import math
import re
import shutil
from datetime import datetime
from pathlib import Path

import soundfile as sf


def safe_project_name(name: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._ -]+", "_", (name or "").strip()).strip(" .")
    return value or "project"


def list_projects(projects_root: Path) -> list[str]:
    projects_root.mkdir(parents=True, exist_ok=True)
    names = [p.name for p in projects_root.iterdir() if p.is_dir()]
    return ["None", *sorted(names, key=str.casefold)]


def create_project(projects_root: Path, name: str) -> tuple[str, Path]:
    safe = safe_project_name(name)
    root = projects_root / safe
    root.mkdir(parents=True, exist_ok=True)
    meta = root / "project.json"
    if not meta.exists():
        meta.write_text(
            json.dumps({"name": safe, "created": datetime.now().isoformat()}, indent=2),
            encoding="utf-8",
        )
    return safe, root


def delete_project(projects_root: Path, name: str) -> None:
    if not name or name == "None":
        raise ValueError("Select a project first.")
    target = projects_root / name
    if target.is_dir():
        shutil.rmtree(target)


def save_project(projects_root: Path, name: str, data: dict) -> Path:
    if not name or name == "None":
        raise ValueError("Create or select a project first.")
    root = projects_root / name
    root.mkdir(parents=True, exist_ok=True)
    payload = dict(data)
    payload["name"] = name
    payload["saved"] = datetime.now().isoformat()
    path = root / "project.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_project(projects_root: Path, name: str) -> dict:
    if not name or name == "None":
        return {}
    path = projects_root / name / "project.json"
    if not path.is_file():
        return {"name": name}
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        return data if isinstance(data, dict) else {"name": name}
    except Exception:
        return {"name": name}


def analyze_manifest(manifest_path: str | Path) -> dict:
    path = Path(manifest_path)
    if not path.is_file():
        raise ValueError("Prepared manifest does not exist.")
    durations: list[float] = []
    samples = 0
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            audio = Path(row.get("audio", ""))
            if not audio.is_file():
                continue
            try:
                info = sf.info(str(audio))
                duration = float(info.duration)
                if duration > 0:
                    durations.append(duration)
                    samples += 1
            except Exception:
                pass
    if not samples:
        raise ValueError("Manifest contains no readable audio samples.")

    values = sorted(durations)
    total = sum(values)

    def percentile(q: float) -> float:
        if len(values) == 1:
            return values[0]
        pos = (len(values) - 1) * q
        lo = int(math.floor(pos))
        hi = int(math.ceil(pos))
        if lo == hi:
            return values[lo]
        frac = pos - lo
        return values[lo] * (1.0 - frac) + values[hi] * frac

    avg = total / samples
    variance = sum((x - avg) ** 2 for x in values) / max(1, samples)
    std = math.sqrt(variance)

    return {
        "sample_count": samples,
        "total_seconds": total,
        "avg_seconds": avg,
        "median_seconds": percentile(0.50),
        "p90_seconds": percentile(0.90),
        "p95_seconds": percentile(0.95),
        "max_seconds": max(values),
        "min_seconds": min(values),
        "std_seconds": std,
        "duration_cv": std / max(avg, 1e-6),
    }


VRAM_PRESETS = {
    # Conservative defaults for the experimental all-linear LoRA objective.
    # Gradient accumulation is primarily a stability/effective-batch control,
    # not a VRAM requirement.
    "12 GB": {"budget": 12, "rank": 4, "grad": 24, "lr": 8e-6},
    "16 GB": {"budget": 16, "rank": 8, "grad": 16, "lr": 9e-6},
    "24 GB": {"budget": 24, "rank": 8, "grad": 12, "lr": 1.0e-5},
    "32 GB+": {"budget": 32, "rank": 16, "grad": 8, "lr": 1.2e-5},
}

def autotune(
    manifest_path: str | Path,
    preset_name: str,
    physical_vram_gb: float | None = None,
) -> dict:
    if preset_name not in VRAM_PRESETS:
        raise ValueError(f"Unknown VRAM preset: {preset_name}")

    profile = dict(VRAM_PRESETS[preset_name])
    analysis = analyze_manifest(manifest_path)

    if physical_vram_gb and physical_vram_gb + 0.25 < profile["budget"]:
        raise ValueError(
            f"{preset_name} preset expects about {profile['budget']} GB VRAM; "
            f"detected {physical_vram_gb:.1f} GB."
        )

    n = int(analysis["sample_count"])
    minutes = float(analysis["total_seconds"]) / 60.0
    median = float(analysis["median_seconds"])
    p90 = float(analysis["p90_seconds"])
    spread = float(analysis["duration_cv"])

    # Effective batch: short homogeneous clips can safely average more samples;
    # long/heterogeneous clips use a little less accumulation.
    grad = int(profile["grad"])
    if median <= 5.0 and p90 <= 9.0:
        grad = min(32, grad + 4)
    elif p90 >= 14.0 or spread >= 0.75:
        grad = max(4, grad - 4)
    profile["grad"] = grad

    # Rank is data-limited before it is VRAM-limited. Avoid rank 16/32 for
    # modest single-voice datasets where the extra capacity tends to overfit.
    if minutes < 15 or n < 120:
        profile["rank"] = min(profile["rank"], 4)
    elif minutes < 90 or n < 800:
        profile["rank"] = min(profile["rank"], 8)
    elif minutes < 240 or n < 1800:
        profile["rank"] = min(profile["rank"], 16)

    # LR scales down for smaller data, higher rank and heterogeneous duration.
    lr = float(profile["lr"])
    if minutes < 15:
        lr *= 0.70
    elif minutes < 45:
        lr *= 0.85
    elif minutes > 180:
        lr *= 1.10

    if profile["rank"] >= 16:
        lr *= 0.85
    if spread >= 0.75:
        lr *= 0.85

    # Clamp experimental all-linear LoRA to a conservative range.
    profile["lr"] = max(5e-6, min(1.5e-5, lr))

    # Steps are derived from effective dataset passes, not fixed buckets.
    # Typical voice-clone datasets need more passes when small, fewer when large.
    if minutes < 10:
        target_passes = 55
    elif minutes < 30:
        target_passes = 42
    elif minutes < 60:
        target_passes = 34
    elif minutes < 120:
        target_passes = 26
    elif minutes < 240:
        target_passes = 20
    else:
        target_passes = 14

    optimizer_steps_per_pass = max(1, math.ceil(n / max(1, profile["grad"])))
    suggested_steps = optimizer_steps_per_pass * target_passes
    suggested_steps = int(round(suggested_steps / 50.0) * 50)
    profile["suggested_steps"] = max(500, min(3000, suggested_steps))

    # Keep roughly 8-12 checkpoints, aligned to clean 25/50 step boundaries.
    raw_save = max(25, profile["suggested_steps"] / 10.0)
    quantum = 25 if raw_save < 150 else 50
    profile["save_steps"] = max(
        quantum,
        int(round(raw_save / quantum) * quantum),
    )

    # Epoch mode remains available, but Steps is the recommendation for this
    # experimental trainer because optimizer updates are the stable unit.
    profile["suggested_epochs"] = max(
        5,
        int(math.ceil(profile["suggested_steps"] / optimizer_steps_per_pass)),
    )
    profile["save_epochs"] = max(1, profile["suggested_epochs"] // 8)
    profile["mode"] = "Steps"
    profile["analysis"] = analysis
    return profile

