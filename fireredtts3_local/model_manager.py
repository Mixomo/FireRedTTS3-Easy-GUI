from __future__ import annotations

import gc
import importlib
import os
import sys
import threading
from pathlib import Path

import torch
from huggingface_hub import snapshot_download

from .paths import MODEL_DIR, ensure_dirs

MODEL_REPO = "FireRedTeam/FireRedTTS3"


class ModelManager:
    def __init__(self):
        self.lock = threading.RLock()
        self.base = None
        self.instruct = None
        self._shared_redae = None
        self._base_key = None
        self._instruct_key = None
        self._attention_backend = None

    def ensure_weights(self, log=print) -> Path:
        ensure_dirs()
        required = [MODEL_DIR / "fireredtts3_base", MODEL_DIR / "fireredtts3_instruct"]
        if all(p.exists() for p in required):
            return MODEL_DIR
        log(f"[download] Downloading {MODEL_REPO} to {MODEL_DIR} ...")
        snapshot_download(repo_id=MODEL_REPO, local_dir=str(MODEL_DIR))
        log("[download] Model download completed.")
        return MODEL_DIR

    @staticmethod
    def _ensure_upstream_namespace():
        upstream = importlib.import_module("fireredtts3_upstream")
        if sys.modules.get("fireredtts3") is not upstream:
            sys.modules["fireredtts3"] = upstream
        return upstream

    def _patch_shared_redae(self):
        if self._shared_redae is not None:
            return
        self._ensure_upstream_namespace()
        from fireredtts3.redae.redae import RedAE
        if not hasattr(RedAE, "_easy_gui_original_from_pretrained"):
            RedAE._easy_gui_original_from_pretrained = RedAE.from_pretrained
        real = RedAE._easy_gui_original_from_pretrained
        manager = self

        def shared(*args, **kwargs):
            if manager._shared_redae is None:
                manager._shared_redae = real(*args, **kwargs)
            return manager._shared_redae

        RedAE.from_pretrained = shared

    @staticmethod
    def _compile_runtime(model, enabled: bool, mode: str, log=print):
        if not enabled:
            return model
        if not hasattr(torch, "compile"):
            log("[WARN] torch.compile is not available in this PyTorch build.")
            return model
        chosen = mode or "default"
        try:
            # Compile the repeated custom transformer components rather than the
            # whole dynamic generation loop/cache machinery.
            model.tts_core.dit = torch.compile(model.tts_core.dit, mode=chosen, dynamic=True)
            model.tts_core.patch_encoder = torch.compile(model.tts_core.patch_encoder, mode=chosen, dynamic=True)
            log(f"[acceleration] torch.compile enabled for DiT + PatchEncoder ({chosen}).")
        except Exception as exc:
            log(f"[WARN] torch.compile setup failed; continuing eager: {exc}")
        return model

    @staticmethod
    def _attach_adapter(model, adapter_path: str | None, log=print):
        if not adapter_path:
            return model
        path = Path(adapter_path)
        if not path.exists():
            raise FileNotFoundError(f"LoRA adapter not found: {path}")
        from peft import PeftModel
        model.tts_core = PeftModel.from_pretrained(model.tts_core, str(path), is_trainable=False)
        model.tts_core.eval()
        log(f"[adapter] Loaded FireRedTTS3 Base LoRA: {path}")
        return model

    def _prepare_attention(self, backend: str, log=print):
        backend = backend or "flash_attention_2"
        if backend not in {"flash_attention_2", "sdpa"}:
            backend = "flash_attention_2"
        if self._attention_backend is not None and self._attention_backend != backend:
            log(f"[acceleration] Attention backend changed {self._attention_backend} -> {backend}; reloading model modules.")
            self.base = None
            self.instruct = None
            self._shared_redae = None
            self._base_key = None
            self._instruct_key = None
            self._attention_backend = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        self._attention_backend = backend
        os.environ["FIRERED_ATTN_IMPLEMENTATION"] = backend
        return backend

    def load_base(
        self,
        log=print,
        attention_backend: str = "flash_attention_2",
        torch_compile: bool = False,
        compile_mode: str = "default",
        adapter_path: str | None = None,
    ):
        with self.lock:
            attention_backend = self._prepare_attention(attention_backend, log)
            key = (attention_backend, bool(torch_compile), str(compile_mode), str(adapter_path or ""))
            if self.base is not None and self._base_key == key:
                return self.base
            if self.base is not None and self._base_key != key:
                self.base = None
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            self.ensure_weights(log)
            self._ensure_upstream_namespace()
            self._patch_shared_redae()
            from fireredtts3.core import FireRedTTS3
            log(f"[model] Loading FireRedTTS3 Base ({attention_backend}) ...")
            self.base = FireRedTTS3(str(MODEL_DIR), use_fasttext=True, use_llm_tn=False, use_wetext=True)
            self.base = self._attach_adapter(self.base, adapter_path, log)
            self.base = self._compile_runtime(self.base, bool(torch_compile), compile_mode, log)
            self._base_key = key
            log("[model] FireRedTTS3 Base ready.")
            return self.base

    def load_instruct(
        self,
        log=print,
        attention_backend: str = "flash_attention_2",
        torch_compile: bool = False,
        compile_mode: str = "default",
    ):
        with self.lock:
            attention_backend = self._prepare_attention(attention_backend, log)
            key = (attention_backend, bool(torch_compile), str(compile_mode))
            if self.instruct is not None and self._instruct_key == key:
                return self.instruct
            if self.instruct is not None and self._instruct_key != key:
                self.instruct = None
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            self.ensure_weights(log)
            self._ensure_upstream_namespace()
            self._patch_shared_redae()
            from fireredtts3.core import FireRedTTS3Instruct
            log(f"[model] Loading FireRedTTS3 Instruct ({attention_backend}) ...")
            self.instruct = FireRedTTS3Instruct(str(MODEL_DIR), use_fasttext=True, use_llm_tn=False, use_wetext=True)
            self.instruct = self._compile_runtime(self.instruct, bool(torch_compile), compile_mode, log)
            self._instruct_key = key
            log("[model] FireRedTTS3 Instruct ready.")
            return self.instruct

    def unload(self, log=print):
        with self.lock:
            objects = [self.base, self.instruct, self._shared_redae]
            self.base = None
            self.instruct = None
            self._shared_redae = None
            self._base_key = None
            self._instruct_key = None
            self._attention_backend = None

            # Drop common nested CUDA references before GC, including PEFT wrappers.
            for obj in objects:
                if obj is None:
                    continue
                try:
                    if hasattr(obj, "tts_core"):
                        obj.tts_core = None
                except Exception:
                    pass
                try:
                    if hasattr(obj, "redae"):
                        obj.redae = None
                except Exception:
                    pass
                try:
                    if hasattr(obj, "spk_extractor"):
                        obj.spk_extractor = None
                except Exception:
                    pass

            objects.clear()
            gc.collect()
            gc.collect()
            if torch.cuda.is_available():
                try:
                    torch.cuda.synchronize()
                except Exception:
                    pass
                torch.cuda.empty_cache()
                try:
                    torch.cuda.ipc_collect()
                except Exception:
                    pass
                torch.cuda.empty_cache()
            gc.collect()
            log("[model] All FireRedTTS3 model references released; CUDA cache cleared.")
