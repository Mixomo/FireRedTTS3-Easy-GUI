from __future__ import annotations

import threading
import urllib.request
from pathlib import Path

_FASTTEXT_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz"
_LANG_TO_LOCALE = {
    "ar":"ar-SA","cs":"cs-CZ","de":"de-DE","el":"el-GR","en":"en-US","es":"es-MX",
    "fi":"fi-FI","fr":"fr-FR","hi":"hi-IN","id":"id-ID","it":"it-IT","ja":"ja-JP",
    "ko":"ko-KR","nl":"nl-NL","pl":"pl-PL","pt":"pt-BR","ro":"ro-RO","ru":"ru-RU",
    "th":"th-TH","tr":"tr-TR","uk":"uk-UA","vi":"vi-VN","zh":"zh-CN",
}
_LOCK = threading.Lock()
_MODEL = None
_ATTEMPTED = False


def model_path(models_root: Path) -> Path:
    return Path(models_root) / "text_frontend" / "lid.176.ftz"


def ensure_fasttext_model(models_root: Path, log=print) -> Path | None:
    global _ATTEMPTED
    path = model_path(models_root)
    if path.exists() and path.stat().st_size > 100_000:
        return path
    with _LOCK:
        if path.exists() and path.stat().st_size > 100_000:
            return path
        if _ATTEMPTED:
            return None
        _ATTEMPTED = True
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            log("[language] Downloading Meta FastText lid.176 model...")
            tmp = path.with_suffix(".tmp")
            urllib.request.urlretrieve(_FASTTEXT_URL, tmp)
            tmp.replace(path)
            log("[language] FastText language detector ready.")
            return path
        except Exception as exc:
            log(f"[WARN] FastText language model unavailable: {exc}")
            return None


class FastTextDetector:
    def __init__(self, models_root: Path, log=print):
        self.models_root = Path(models_root)
        self.log = log
        self._model = None

    def _load(self):
        if self._model is not None:
            return self._model
        path = ensure_fasttext_model(self.models_root, self.log)
        if not path:
            return None
        try:
            import fasttext
            self._model = fasttext.load_model(str(path))
        except Exception as exc:
            self.log(f"[WARN] FastText detector could not load: {exc}")
            self._model = None
        return self._model

    def detect_locale(self, text: str):
        model = self._load()
        if model is None:
            return None
        snippet = (text or "").replace("\n", " ").strip()
        if not snippet:
            return None
        try:
            preds = model.f.predict(snippet, 1, 0.0, "strict")
            if not preds:
                return None
            _prob, label = preds[0]
            return _LANG_TO_LOCALE.get(label.replace("__label__", ""))
        except Exception:
            return None
