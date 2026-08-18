from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

from .paths import ROOT, VOICES_DIR, VOICE_INDEX, ensure_dirs

NONE_VOICE = "None"
AUDIO_EXTS = {".wav", ".flac", ".mp3", ".m4a", ".ogg", ".aac", ".opus"}
TEXT_KEYS = ("Text", "text", "Transcript", "transcript", "ReferenceText", "reference_text")
SAMPLES_DIR = ROOT / "samples"  # Higgs-compatible secondary library root.


def _safe_name(name: str) -> str:
    name = re.sub(r"[^A-Za-z0-9._ -]+", "_", (name or "").strip()).strip(" .")
    return name or "voice"


def _roots() -> tuple[Path, ...]:
    ensure_dirs()
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    # FireRed uses voices/ as its canonical write target, but reads Higgs' samples/
    # too so a copied/shared library works without conversion.
    return VOICES_DIR, SAMPLES_DIR


def _read_legacy_index() -> dict:
    if not VOICE_INDEX.is_file():
        return {}
    try:
        data = json.loads(VOICE_INDEX.read_text(encoding="utf-8-sig"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _audio_files(root: Path):
    if not root.exists():
        return
    for p in root.iterdir():
        if p.is_file() and p.suffix.lower() in AUDIO_EXTS:
            yield p


def _same_stem_audio(root: Path, stem: str) -> Path | None:
    for p in _audio_files(root) or ():
        if p.stem.casefold() == stem.casefold():
            return p
    return None


def _read_sidecar(audio: Path, meta: dict | None = None) -> str:
    txt = audio.with_suffix(".txt")
    if txt.is_file():
        try:
            value = txt.read_text(encoding="utf-8-sig").strip()
            if value:
                return value
        except OSError:
            pass
    data = meta
    if data is None:
        js = audio.with_suffix(".json")
        if js.is_file():
            try:
                data = json.loads(js.read_text(encoding="utf-8-sig"))
            except Exception:
                data = None
    if isinstance(data, dict):
        for key in TEXT_KEYS:
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return ""


def _resolve_from_root(root: Path, name: str) -> tuple[Path | None, str]:
    js = root / f"{name}.json"
    meta = None
    if js.is_file():
        try:
            meta = json.loads(js.read_text(encoding="utf-8-sig"))
        except Exception:
            meta = None
    if isinstance(meta, dict):
        raw = meta.get("audio")
        if isinstance(raw, str) and raw.strip():
            candidate = Path(raw.strip()).expanduser()
            if not candidate.is_absolute():
                candidate = root / candidate
            if candidate.is_file():
                return candidate, _read_sidecar(candidate, meta)
    audio = _same_stem_audio(root, name)
    if audio is not None:
        return audio, _read_sidecar(audio, meta)
    return None, ""


def list_voices() -> list[str]:
    names: set[str] = set()
    for root in _roots():
        for p in _audio_files(root) or ():
            names.add(p.stem)
        for meta in root.glob("*.json"):
            audio, _ = _resolve_from_root(root, meta.stem)
            if audio is not None:
                names.add(meta.stem)

    # Preserve compatibility with the original FireRed Easy GUI voices.json index.
    for display_name, item in _read_legacy_index().items():
        if not isinstance(item, dict):
            continue
        raw = item.get("audio")
        if isinstance(raw, str) and (VOICES_DIR / raw).is_file():
            names.add(str(display_name))
    return [NONE_VOICE, *sorted(names, key=str.casefold)]


def save_voice(name: str, audio_path: str, transcript: str) -> str:
    if not audio_path:
        raise ValueError("Reference audio is required.")
    src = Path(audio_path)
    if not src.is_file():
        raise ValueError(f"Reference audio file does not exist: {src}")
    ensure_dirs()
    safe = _safe_name(name or src.stem)
    suffix = src.suffix.lower() if src.suffix.lower() in AUDIO_EXTS else ".wav"
    dst = VOICES_DIR / f"{safe}{suffix}"
    if src.resolve() != dst.resolve():
        shutil.copy2(src, dst)
    text = (transcript or "").strip()

    # Higgs reads same-stem audio + .txt/.json {Type,Text}; MOSS also accepts
    # audio/transcript. Write the superset so a single directory is portable.
    dst.with_suffix(".txt").write_text(text, encoding="utf-8")
    dst.with_suffix(".json").write_text(
        json.dumps(
            {
                "Type": "Sample",
                "Text": text,
                "audio": dst.name,
                "transcript": text,
                "format": "MOSS-Higgs-FireRed-compatible",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return safe


def resolve_voice(name: str | None):
    if not name or name == NONE_VOICE:
        return None, ""
    for root in _roots():
        audio, text = _resolve_from_root(root, name)
        if audio is not None:
            return str(audio), text

    legacy = _read_legacy_index().get(name)
    if isinstance(legacy, dict):
        raw = legacy.get("audio", "")
        path = VOICES_DIR / str(raw)
        if path.is_file():
            return str(path), str(legacy.get("transcript", "") or "")
    return None, ""


def delete_voice(name: str) -> None:
    if not name or name == NONE_VOICE:
        return
    for root in _roots():
        audio, _ = _resolve_from_root(root, name)
        candidates = {root / f"{name}.json", root / f"{name}.txt"}
        if audio is not None and audio.parent.resolve() == root.resolve():
            candidates.update({audio, audio.with_suffix(".txt"), audio.with_suffix(".json")})
        for p in candidates:
            try:
                p.unlink(missing_ok=True)
            except OSError:
                pass

    # Clean only the matching legacy index entry; leave all others intact.
    data = _read_legacy_index()
    if name in data:
        data.pop(name, None)
        VOICE_INDEX.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
