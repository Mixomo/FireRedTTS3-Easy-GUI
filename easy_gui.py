from __future__ import annotations

import os
import random
import secrets
import re
import sys
import threading
import time
import html
import traceback
try:
    import winsound
except ImportError:
    winsound = None
import webbrowser
import subprocess
import subprocess
import json
import math
from collections import deque
from datetime import datetime
from pathlib import Path

if sys.platform.startswith("win"):
    try:
        import asyncio
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except Exception:
        pass

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "models" / ".cache"
os.environ.setdefault("HF_HOME", str(ROOT / "models"))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(CACHE / "huggingface"))
os.environ.setdefault("HF_XET_CACHE", str(CACHE / "xet"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(CACHE / "transformers"))
os.environ.setdefault("TORCH_HOME", str(CACHE / "torch"))
os.environ.setdefault("GRADIO_TEMP_DIR", str(CACHE / "tmp"))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE))

import gradio as gr
import torch

from fireredtts3_local.audio_utils import load_audio, save_tensor_audio
from fireredtts3_local.asr import ASRManager, WHISPER_LANGS, WHISPER_MODELS
from fireredtts3_local.model_manager import ModelManager
from fireredtts3_local.training import prepare_dataset, scan_dataset, stop_training, train_lora
from fireredtts3_local.paths import MODEL_DIR, OUTPUTS_DIR, CPP_ROOT, CPP_BUNDLES_DIR, ensure_dirs
from fireredtts3_local.voice_library import NONE_VOICE, delete_voice, list_voices, resolve_voice, save_voice
from fireredtts3_local.projects import (
    VRAM_PRESETS, analyze_manifest, autotune, create_project as project_create,
    delete_project as project_delete, list_projects, load_project, save_project,
)

ensure_dirs()
MANAGER = ModelManager()
ASR = ASRManager(ROOT / "models" / "asr")
TRAINING_ROOT = ROOT / "training"
TRAINING_PROJECTS = TRAINING_ROOT / "projects"
TRAINING_OUTPUTS = TRAINING_ROOT / "outputs"
TRAINING_PROJECTS.mkdir(parents=True, exist_ok=True)
TRAINING_OUTPUTS.mkdir(parents=True, exist_ok=True)
CMD_MIRROR_LINES = deque(maxlen=1200)
CMD_MIRROR_LOCK = threading.Lock()
CMD_MIRROR_CURRENT = ""
CMD_MIRROR_OVERWRITE = False

APP_TITLE = "FireRedTTS3 Easy GUI"
LANGUAGES = [
    "Auto-detect", "Arabic", "Cantonese", "Chinese", "Czech", "Dutch", "English", "Finnish",
    "French", "German", "Greek", "Hindi", "Indonesian", "Italian", "Japanese", "Korean",
    "Polish", "Portuguese", "Romanian", "Russian", "Spanish", "Thai", "Turkish", "Ukrainian", "Vietnamese",
    "ZH_Anhui", "ZH_Fujian", "ZH_Gansu", "ZH_Guizhou", "ZH_Hebei", "ZH_Henan", "ZH_Hubei", "ZH_Hunan",
    "ZH_Jiangxi", "ZH_Liaoning", "ZH_Minnan", "ZH_Ningxia", "ZH_Shaanxi", "ZH_Shandong", "ZH_Shanghai",
    "ZH_Shanxi", "ZH_Sichuan", "ZH_Tianjin", "ZH_Wenzhou", "ZH_Wu", "ZH_Yunnan",
]

CSS = """
.title-section { border-bottom: 1px solid var(--border-color-primary); margin-bottom: 6px; padding-bottom: 4px; align-items:center !important; }
.tabs { margin-top: 2px; }
.form-section { padding: 14px; border: 1px solid var(--border-color-primary); border-radius: 10px; background: var(--block-background-fill); }
.button-primary { background: #2563eb !important; color: white !important; }
.button-stop, .red-btn { background: #dc3545 !important; color: white !important; }
.green-btn { background: #28a745 !important; color: white !important; }
.global-toolbar { padding: 10px 12px; border: 1px solid var(--border-color-primary); border-radius: 10px; background: var(--block-background-fill); margin-bottom: 12px; }
.global-toolbar button { min-height: 38px !important; }
.audio-safe-space { overflow: visible !important; padding-bottom: 20px !important; border: 0 !important; box-shadow: none !important; }
.audio-safe-space .wave, .audio-safe-space [data-testid="waveform"] { margin-bottom: 26px !important; }
.output-clean, .output-clean > div, .output-clean .wrap { border: 0 !important; box-shadow: none !important; }
.output-path textarea { border: 0 !important; background: var(--input-background-fill) !important; min-height: 40px !important; }
.project-strip { padding: 12px; border-radius: 10px; border: 1px solid var(--border-color-primary); margin-bottom: 12px; }
.console-accordion, .console-accordion > div { border-radius: 8px !important; }
.cmd-mirror { display:block; width:100%; height:333px; border:0; border-radius:8px; overflow:hidden; }
.dialogue-toolbar { margin-bottom: 8px; }
.dialogue-turn-card { padding: 10px 12px !important; border: 1px solid var(--border-color-primary) !important; border-radius: 10px !important; margin-bottom: 8px !important; }
.dialogue-turn-card .gr-row { align-items: end !important; }
.dialogue-actions button { min-width: 40px !important; padding-left: 8px !important; padding-right: 8px !important; }
.progress-card { padding:8px 10px; border-radius:8px; border:1px solid var(--border-color-primary); margin-top:8px; }
.train-bar { height:10px; width:100%; border-radius:6px; background:#202638; overflow:hidden; margin-top:6px; } .train-fill { height:100%; background:#3b82f6; }
.small-note { opacity:.78; font-size:.9em; }
.tab-subtitle { opacity:.82; margin:0 0 4px 0 !important; padding:0 !important; }
.compact-status { margin:0 !important; padding:0 !important; min-height:0 !important; }
.title-section .prose, .title-section h1 { margin:0 !important; padding:0 !important; }
.title-section button { min-height:36px !important; white-space:nowrap; }

.fire-progress-card {
    padding: 12px 14px;
    border: 1px solid var(--border-color-primary);
    border-radius: 10px;
    background: var(--block-background-fill);
    margin: 10px 0 12px 0;
}
.fire-progress-head {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    gap: 14px;
    margin-bottom: 8px;
}
.fire-progress-title {
    font-weight: 650;
}
.fire-progress-meta {
    opacity: .78;
    font-size: .9em;
    text-align: right;
}
.fire-train-track {
    position: relative;
    width: 100%;
    height: 12px;
    overflow: hidden;
    border-radius: 999px;
    background: color-mix(in srgb, var(--border-color-primary) 65%, transparent);
}
.fire-train-fill {
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, #2563eb, #60a5fa);
    transition: width .35s ease;
}
.fire-progress-foot {
    display: flex;
    justify-content: space-between;
    gap: 12px;
    margin-top: 7px;
    font-size: .88em;
    opacity: .82;
}

"""


class _CmdMirror:
    def __init__(self, stream):
        self.stream = stream
        self.encoding = getattr(stream, "encoding", "utf-8")

    def write(self, data):
        if data:
            with CMD_MIRROR_LOCK:
                _mirror_write(str(data))
        return self.stream.write(data)

    def flush(self):
        return self.stream.flush()

    def isatty(self):
        return getattr(self.stream, "isatty", lambda: False)()


def _clean_cmd_line(line: str) -> str:
    return re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", line).rstrip()


def _mirror_commit(line: str) -> None:
    line = _clean_cmd_line(line)
    if line.strip():
        CMD_MIRROR_LINES.append(line)


def _mirror_write(data: str) -> None:
    global CMD_MIRROR_CURRENT, CMD_MIRROR_OVERWRITE
    for ch in data:
        if ch == "\r":
            CMD_MIRROR_CURRENT = ""
            CMD_MIRROR_OVERWRITE = True
            continue
        if ch == "\n":
            _mirror_commit(CMD_MIRROR_CURRENT)
            CMD_MIRROR_CURRENT = ""
            CMD_MIRROR_OVERWRITE = False
            continue
        if CMD_MIRROR_OVERWRITE:
            CMD_MIRROR_CURRENT = ""
            CMD_MIRROR_OVERWRITE = False
        CMD_MIRROR_CURRENT += ch


def _install_cmd_mirror():
    if getattr(sys.stdout, "_moss_cmd_mirror", False):
        return

    class LockedMirror(_CmdMirror):
        _moss_cmd_mirror = True

    sys.stdout = LockedMirror(sys.stdout)
    sys.stderr = LockedMirror(sys.stderr)


_HTML_ESC = str.maketrans({"&": "&amp;", "<": "&lt;", ">": "&gt;"})
_ATTR_ESC = str.maketrans({"&": "&amp;", '"': "&quot;", "<": "&lt;", ">": "&gt;"})


def _line_color(line: str) -> str:
    low = line.lower()
    if "error" in low or "traceback" in low or "exception" in low or "failed" in low:
        return "#f87171"
    if "warn" in low or "deprecated" in low:
        return "#fbbf24"
    if "download" in low or "snapshot" in low or "%|" in line:
        return "#60a5fa"
    if "train" in low or "epoch" in low or "lora" in low:
        return "#a78bfa"
    if "saved" in low or "ready" in low or "done" in low or "complete" in low:
        return "#4ade80"
    return "#cccccc"


def console_html():
    with CMD_MIRROR_LOCK:
        lines = list(CMD_MIRROR_LINES)
        current = _clean_cmd_line(CMD_MIRROR_CURRENT)
    if current.strip():
        lines.append(current)
    display = lines[-160:] if lines else ["Idle."]
    rows = []
    for line in display:
        safe = line.translate(_HTML_ESC)
        rows.append(f'<div style="color:{_line_color(line)};white-space:pre;line-height:1.55">{safe}</div>')
    content = "\n".join(rows)
    srcdoc = f"""<!doctype html><html><head><style>
html,body{{margin:0;background:#111;color:#ccc;font-family:Consolas,ui-monospace,monospace;font-size:12px;}}
#wrap{{height:333px;border-radius:8px;border:1px solid #333;overflow:hidden;box-sizing:border-box;}}
#body{{height:333px;overflow:auto;padding:8px 20px 8px 12px;box-sizing:border-box;scrollbar-width:thin;scrollbar-color:#555 transparent;}}
#body::-webkit-scrollbar{{width:5px;height:5px}} #body::-webkit-scrollbar-thumb{{background:#555;border-radius:3px}}
</style></head><body><div id="wrap"><div id="body">{content}<div id="anchor"></div></div></div>
<script>const b=document.getElementById('body'); b.onscroll=()=>{{window._paused=!(b.scrollTop+b.clientHeight>=b.scrollHeight-40);}}; if(!window._paused)b.scrollTop=b.scrollHeight; setTimeout(()=>{{if(!window._paused)b.scrollTop=b.scrollHeight;}},50);</script>
</body></html>"""
    return f'<iframe class="cmd-mirror" scrolling="no" srcdoc="{srcdoc.translate(_ATTR_ESC)}"></iframe>'


def log(msg: str):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)



_install_cmd_mirror()

def _lang(value):
    return None if not value or value == "Auto-detect" else value


def _voice_dropdown(selected=None):
    choices = list_voices()
    value = selected if selected in choices else NONE_VOICE
    return gr.update(choices=choices, value=value)


def select_voice(name):
    path, transcript = resolve_voice(name)
    return path, transcript


def add_voice(name, audio, transcript):
    try:
        saved = save_voice(name, audio, transcript)
        log(f"[voice] Saved voice: {saved}")
        return _voice_dropdown(saved), "Voice saved.", console_html()
    except Exception as exc:
        log(f"[error] {exc}")
        return _voice_dropdown(), f"Error: {exc}", console_html()


def remove_voice(name):
    delete_voice(name)
    log(f"[voice] Deleted voice: {name}")
    return _voice_dropdown(), None, "", console_html()


def unload_models():
    MANAGER.unload(log)
    return "Models unloaded.", console_html()



def _resolved_seed(seed, random_seed):
    if bool(random_seed):
        value = secrets.randbelow(2_147_483_647)
        log(f"[seed] Random seed: {value}")
        return value
    value = int(seed)
    log(f"[seed] Seed: {value}")
    return value


def _attn_backend(label):
    return "sdpa" if str(label).startswith("PyTorch SDPA") else "flash_attention_2"


def _adapter_choices():
    choices = ["None"]
    if TRAINING_OUTPUTS.exists():
        for p in sorted(TRAINING_OUTPUTS.iterdir()):
            if p.is_dir() and (p / "adapter_config.json").exists():
                choices.append(str(p))
    return choices


def refresh_adapters(selected="None"):
    choices = _adapter_choices()
    value = selected if selected in choices else "None"
    return gr.update(choices=choices, value=value)



def play_completion_chime():
    """Play assets/chime.wav asynchronously when a user-facing job completes."""
    if winsound is None:
        return
    chime = ROOT / "assets" / "chime.wav"
    if not chime.is_file():
        return
    try:
        winsound.PlaySound(
            str(chime),
            winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_NODEFAULT,
        )
    except Exception as exc:
        log(f"[chime] Could not play completion sound: {exc}")


def clone_voice(
    voice_name, prompt_audio, prompt_text, text, language,
    cfg, steps, seed, random_seed, do_tn,
    chunk_mode, chunk_silence, attention, use_compile, compile_mode, adapter,
    progress=gr.Progress(track_tqdm=False),
):
    try:
        progress(0.03, desc="Validating reference and text")
        if voice_name and voice_name != NONE_VOICE:
            progress(0.07, desc="Resolving Voice Library reference")
            lib_audio, lib_text = resolve_voice(voice_name)
            prompt_audio = lib_audio or prompt_audio
            prompt_text = lib_text or prompt_text
        if not prompt_audio:
            raise ValueError("Reference audio is required.")
        if not (prompt_text or "").strip():
            raise ValueError("Reference transcript is required.")

        progress(0.11, desc="Preparing text and chunk plan")
        chunks = split_long_text(text, chunk_mode or "None")
        if not chunks:
            raise ValueError("No usable text chunks were found.")
        actual_seed = _resolved_seed(seed, random_seed)
        adapter_path = None if not adapter or adapter == "None" else adapter

        progress(0.16, desc="Loading FireRedTTS3 Base runtime")
        tts = MANAGER.load_base(
            log,
            attention_backend=_attn_backend(attention),
            torch_compile=bool(use_compile),
            compile_mode=str(compile_mode),
            adapter_path=adapter_path,
        )
        progress(0.30, desc="Loading and encoding reference audio")
        wav, sr = load_audio(prompt_audio)

        generated = []
        out_sr = None
        total_chunks = max(1, len(chunks))
        for index, chunk in enumerate(chunks, 1):
            progress(
                0.34 + 0.52 * ((index - 1) / total_chunks),
                desc=f"Synthesizing chunk {index}/{total_chunks} · diffusion + decoding",
            )
            log(f"[long-form] chunk {index}/{total_chunks} · {len(chunk)} chars")
            audio, out_sr = tts.generate(
                text=chunk,
                language=_lang(language),
                prompt_text=prompt_text.strip(),
                prompt_audio=wav,
                prompt_audio_sr=sr,
                inference_cfg=float(cfg),
                n_timesteps=int(steps),
                seed=actual_seed,
                do_tn=bool(do_tn),
                do_split=False,
                cross_fade_ms=0,
            )
            generated.append(audio)
            progress(0.34 + 0.52 * (index / total_chunks), desc=f"Chunk {index}/{total_chunks} complete")

        progress(0.89, desc="Merging generated audio")
        final_audio = _concat_tensor_chunks(generated, out_sr, float(chunk_silence)) if len(generated) > 1 else generated[0]
        progress(0.95, desc="Saving WAV output")
        path = save_tensor_audio(final_audio, out_sr, "voice-clone")
        log(f"[done] {path}")
        play_completion_chime()
        progress(1.0, desc="Done")
        return path, path, console_html()
    except Exception as exc:
        log(traceback.format_exc())
        raise gr.Error(str(exc))

def voice_design(
    instruction, text, language, cfg, steps, seed, random_seed, do_tn,
    chunk_mode, chunk_silence, attention, use_compile, compile_mode,
    progress=gr.Progress(track_tqdm=False),
):
    try:
        progress(0.03, desc="Validating voice description and text")
        if not (instruction or "").strip() or not (text or "").strip():
            raise ValueError("Voice description and target text are required.")
        progress(0.10, desc="Preparing text and chunk plan")
        chunks = split_long_text(text, chunk_mode or "None")
        actual_seed = _resolved_seed(seed, random_seed)

        progress(0.16, desc="Loading FireRedTTS3 Instruct runtime")
        model = MANAGER.load_instruct(
            log,
            attention_backend=_attn_backend(attention),
            torch_compile=bool(use_compile),
            compile_mode=str(compile_mode),
        )

        audios = []
        sr = None
        plan = ""
        total_chunks = max(1, len(chunks))
        for index, chunk in enumerate(chunks, 1):
            progress(
                0.30 + 0.56 * ((index - 1) / total_chunks),
                desc=f"Voice Design · chunk {index}/{total_chunks} · synthesizing",
            )
            log(f"[long-form] Voice Design chunk {index}/{total_chunks} · {len(chunk)} chars")
            audio, sr, local_plan = model.generate_voice_design(
                instruction=instruction.strip(),
                text=chunk,
                language=_lang(language),
                inference_cfg=float(cfg),
                n_timesteps=int(steps),
                seed=actual_seed,
                do_tn=bool(do_tn),
                do_split=False,
                cross_fade_ms=0,
            )
            audios.append(audio)
            plan = plan or (local_plan or "")
            progress(0.30 + 0.56 * (index / total_chunks), desc=f"Voice Design · chunk {index}/{total_chunks} complete")

        progress(0.89, desc="Merging generated audio")
        final_audio = _concat_tensor_chunks(audios, sr, float(chunk_silence)) if len(audios) > 1 else audios[0]
        progress(0.95, desc="Saving WAV output")
        path = save_tensor_audio(final_audio, sr, "voice-design")
        progress(1.0, desc="Done")
        return path, plan, path, console_html()
    except Exception as exc:
        log(traceback.format_exc())
        raise gr.Error(str(exc))

def semantic_edit(
    audio_in, instruction, cfg, steps, seed, random_seed,
    attention, use_compile, compile_mode,
    progress=gr.Progress(track_tqdm=False),
):
    try:
        progress(0.04, desc="Validating input speech and edit instruction")
        if not audio_in or not (instruction or "").strip():
            raise ValueError("Input audio and edit instruction are required.")
        actual_seed = _resolved_seed(seed, random_seed)

        progress(0.14, desc="Loading FireRedTTS3 Instruct runtime")
        model = MANAGER.load_instruct(
            log,
            attention_backend=_attn_backend(attention),
            torch_compile=bool(use_compile),
            compile_mode=str(compile_mode),
        )
        progress(0.30, desc="Loading and encoding source speech")
        wav, sr = load_audio(audio_in)
        log("[edit] Running semantic edit...")
        progress(0.46, desc="Interpreting semantic edit instruction")
        progress(0.54, desc="Generating edited acoustic representation")
        audio, out_sr, edited_text = model.generate_semantic_edit(
            instruction=instruction.strip(),
            audio_in=wav,
            audio_in_sr=sr,
            inference_cfg=float(cfg),
            n_timesteps=int(steps),
            seed=actual_seed,
        )
        progress(0.90, desc="Decoding edited speech")
        progress(0.96, desc="Saving WAV output")
        path = save_tensor_audio(audio, out_sr, "semantic-edit")
        log(f"[done] {path}")
        play_completion_chime()
        progress(1.0, desc="Done")
        return path, edited_text or "", path, console_html()
    except Exception as exc:
        log(traceback.format_exc())
        raise gr.Error(str(exc))

def acoustic_edit(
    audio_in, attribute, value, cfg, steps, seed, random_seed,
    attention, use_compile, compile_mode,
    progress=gr.Progress(track_tqdm=False),
):
    try:
        progress(0.04, desc="Validating input speech and acoustic controls")
        if not audio_in:
            raise ValueError("Input audio is required.")
        if attribute == "Speed":
            instruction = f"adjust the speed to {float(value):.1f}x"
        elif attribute == "Volume":
            instruction = f"adjust the volume to {float(value):.1f}"
        else:
            n = int(round(float(value)))
            if n == 0:
                raise ValueError("Pitch shift cannot be 0 for the trained FireRedTTS3 acoustic-edit template.")
            instruction = f"shift the pitch by {n} step{'' if abs(n) == 1 else 's'}"
        actual_seed = _resolved_seed(seed, random_seed)

        progress(0.14, desc="Loading FireRedTTS3 Instruct runtime")
        model = MANAGER.load_instruct(
            log,
            attention_backend=_attn_backend(attention),
            torch_compile=bool(use_compile),
            compile_mode=str(compile_mode),
        )
        progress(0.30, desc="Loading and encoding source speech")
        wav, sr = load_audio(audio_in)
        log(f"[edit] {instruction}")
        progress(0.48, desc=f"Applying acoustic edit · {attribute}")
        audio, out_sr = model.generate_acoustic_edit(
            instruction=instruction,
            audio_in=wav,
            audio_in_sr=sr,
            inference_cfg=float(cfg),
            n_timesteps=int(steps),
            seed=actual_seed,
        )
        progress(0.90, desc="Decoding edited speech")
        progress(0.96, desc="Saving WAV output")
        path = save_tensor_audio(audio, out_sr, "acoustic-edit")
        log(f"[done] {path}")
        play_completion_chime()
        progress(1.0, desc="Done")
        return path, instruction, path, console_html()
    except Exception as exc:
        log(traceback.format_exc())
        raise gr.Error(str(exc))

def acoustic_slider(attribute):
    if attribute == "Pitch":
        return gr.update(minimum=-6, maximum=6, step=1, value=1, label="Pitch steps (-6 to +6, excluding 0)")
    if attribute == "Volume":
        return gr.update(minimum=0.3, maximum=2.0, step=0.1, value=1.0, label="Volume")
    return gr.update(minimum=0.5, maximum=2.0, step=0.1, value=1.0, label="Speed")


def transcribe_reference(audio_path, model_label, language, batch_size):
    try:
        if not audio_path:
            raise ValueError("Select or record audio first.")
        log(f"[asr] Transcribing with {model_label}...")
        text, status = ASR.transcribe(audio_path, model_label, language, int(batch_size))
        log(f"[asr] {status}")
        return text, status, console_html()
    except Exception as exc:
        log(traceback.format_exc())
        raise gr.Error(str(exc))


def clear_outputs():
    removed = 0
    for p in OUTPUTS_DIR.glob("*"):
        if p.is_file():
            try:
                p.unlink()
                removed += 1
            except OSError:
                pass
    log(f"[cleanup] Cleared {removed} output file(s).")
    return f"Cleared {removed} output file(s).", console_html()


def refresh_voice_dropdown(selected=None):
    return _voice_dropdown(selected)


def unload_all():
    try:
        MANAGER.unload(log)
    finally:
        try:
            ASR.unload()
        except Exception as exc:
            log(f"[models][WARN] ASR unload: {exc}")
        try:
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                try:
                    torch.cuda.ipc_collect()
                except Exception:
                    pass
                torch.cuda.empty_cache()
            gc.collect()
        except Exception as exc:
            log(f"[models][WARN] final CUDA cleanup: {exc}")
    log("[models] All models unloaded and VRAM cleanup requested.")
    return "All models unloaded / VRAM released.", console_html()


CHUNK_CHOICES = ["None", "Paragraph/Sentence Auto", "Periods", "Paragraphs", "Lines", "Speaker turns"]


def paragraph_sentence_split(text: str, max_chars: int = 120) -> list[str]:
    """Higgs/MOSS automatic long-text splitter."""
    text = (text or "").strip()
    if not text:
        return []
    chunks: list[str] = []
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]
    for paragraph in paragraphs or [text]:
        paragraph = re.sub(r"\s+", " ", paragraph)
        if len(paragraph) <= max_chars:
            chunks.append(paragraph)
            continue
        current = ""
        for sentence in re.split(r"(?<=[.!?…])\s+", paragraph):
            sentence = sentence.strip()
            if not sentence:
                continue
            candidate = f"{current} {sentence}".strip() if current else sentence
            if len(candidate) > max_chars and current:
                chunks.append(current)
                current = sentence
            else:
                current = candidate
        if current:
            chunks.append(current)
    return chunks


def split_by_periods(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", (text or "").strip())
    if not text:
        return []
    return [chunk.strip() for chunk in re.split(r"(?<=\.)\s+", text) if chunk.strip()]


def split_long_text(text: str, mode: str) -> list[str]:
    """Same long-form splitting rules used by the MOSS Easy GUI."""
    text = (text or "").strip()
    if not text:
        return []
    if mode == "None":
        return [text]
    if mode == "Paragraph/Sentence Auto":
        return paragraph_sentence_split(text)
    if mode == "Periods":
        return split_by_periods(text)
    if mode == "Paragraphs":
        return [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]
    if mode == "Lines":
        return [p.strip() for p in text.splitlines() if p.strip()]
    if mode == "Speaker turns":
        chunks=[]; current=[]
        for line in text.splitlines():
            if re.match(r"^\s*\[?SPEAKER\d+\]?", line, flags=re.IGNORECASE) and current:
                chunks.append("\n".join(current).strip()); current=[line.strip()]
            elif line.strip(): current.append(line.strip())
        if current: chunks.append("\n".join(current).strip())
        return chunks or [text]
    return [text]


def _concat_tensor_chunks(chunks, sr: int, gap_seconds: float):
    kept=[c.detach().cpu() for c in chunks if c is not None and c.numel()]
    if not kept: raise RuntimeError("No audio chunks were generated.")
    silence_frames=max(0,int(round(float(sr)*max(0.0,float(gap_seconds)))))
    silence=torch.zeros((kept[0].shape[0],silence_frames),dtype=kept[0].dtype)
    out=[]
    for i,c in enumerate(kept):
        if i and silence_frames: out.append(silence)
        out.append(c)
    return torch.cat(out,dim=1)


DIALOGUE_MAX_TURNS = 12


def _dialogue_unpack(values):
    rows = []
    for i in range(DIALOGUE_MAX_TURNS):
        voice = values[i*2] if i*2 < len(values) else NONE_VOICE
        text = values[i*2+1] if i*2+1 < len(values) else ""
        rows.append([voice or NONE_VOICE, text or ""])
    return rows


def _dialogue_updates(count, rows, message="Dialogue rows updated."):
    count = max(1, min(int(count), DIALOGUE_MAX_TURNS))
    rows = (rows + [[NONE_VOICE, ""] for _ in range(DIALOGUE_MAX_TURNS)])[:DIALOGUE_MAX_TURNS]
    flat = []
    for voice, text in rows:
        flat.extend([gr.update(value=voice), gr.update(value=text)])
    vis = [gr.update(visible=i<count) for i in range(DIALOGUE_MAX_TURNS)]
    return [count, *flat, *vis, message]


def dialogue_reset_rows():
    return _dialogue_updates(2, [[NONE_VOICE, ""] for _ in range(DIALOGUE_MAX_TURNS)], "Dialogue reset to two empty turns.")


def dialogue_clear_rows(count, *values):
    rows = _dialogue_unpack(values)
    for i in range(min(int(count), DIALOGUE_MAX_TURNS)):
        rows[i] = [NONE_VOICE, ""]
    return _dialogue_updates(count, rows, "Visible dialogue turns cleared.")


def dialogue_compact_rows(count, *values):
    rows = _dialogue_unpack(values)
    active = [r for r in rows[:int(count)] if str(r[1]).strip()]
    if not active:
        active = [[NONE_VOICE, ""]]
    return _dialogue_updates(len(active), active, "Removed empty turns.")


def dialogue_add_after(index, count, *values):
    rows = _dialogue_unpack(values); count = int(count)
    if count >= DIALOGUE_MAX_TURNS:
        return _dialogue_updates(count, rows, f"Maximum of {DIALOGUE_MAX_TURNS} turns reached.")
    rows.insert(min(int(index)+1, count), [NONE_VOICE, ""])
    return _dialogue_updates(count+1, rows[:DIALOGUE_MAX_TURNS], f"Added a turn after Turn {int(index)+1}.")


def dialogue_clone_row(index, count, *values):
    rows = _dialogue_unpack(values); count = int(count); idx = min(int(index), max(0, count-1))
    if count >= DIALOGUE_MAX_TURNS:
        return _dialogue_updates(count, rows, f"Maximum of {DIALOGUE_MAX_TURNS} turns reached.")
    rows.insert(idx+1, list(rows[idx]))
    return _dialogue_updates(count+1, rows[:DIALOGUE_MAX_TURNS], f"Cloned Turn {idx+1}.")


def dialogue_delete_row(index, count, *values):
    rows = _dialogue_unpack(values); count = max(1, int(count)); idx = min(int(index), count-1)
    if count == 1:
        rows[0] = [NONE_VOICE, ""]
        return _dialogue_updates(1, rows, "Last turn cleared.")
    rows.pop(idx); rows.append([NONE_VOICE, ""])
    return _dialogue_updates(count-1, rows, f"Deleted Turn {idx+1}.")


def dialogue_clear_row(index, count, *values):
    rows = _dialogue_unpack(values); rows[int(index)] = [NONE_VOICE, ""]
    return _dialogue_updates(count, rows, f"Cleared Turn {int(index)+1}.")


def dialogue_move_row(index, direction, count, *values):
    rows = _dialogue_unpack(values); count = int(count); idx = int(index); dest = idx + int(direction)
    if 0 <= idx < count and 0 <= dest < count:
        rows[idx], rows[dest] = rows[dest], rows[idx]
        return _dialogue_updates(count, rows, f"Moved Turn {idx+1} {'up' if direction < 0 else 'down'}.")
    return _dialogue_updates(count, rows, "Turn is already at that boundary.")


def generate_dialogue(
    language, pause_seconds, cfg, steps, seed, random_seed,
    do_tn, attention, use_compile, compile_mode, adapter, row_count, *values,
    progress=gr.Progress(track_tqdm=False),
):
    try:
        progress(0.03, desc="Validating dialogue rows")
        rows = _dialogue_unpack(values)[:max(1, min(int(row_count), DIALOGUE_MAX_TURNS))]
        active = [(voice, text.strip()) for voice, text in rows if str(text).strip()]
        if not active:
            raise ValueError("Dialogue has no turns with text.")
        if any(not voice or voice == NONE_VOICE for voice, _ in active):
            raise ValueError("Every dialogue turn with text requires a saved voice.")

        progress(0.10, desc="Resolving voices and generation settings")
        actual_seed = _resolved_seed(seed, random_seed)
        adapter_path = None if not adapter or adapter == "None" else adapter
        progress(0.15, desc="Loading FireRedTTS3 Base runtime")
        model = MANAGER.load_base(
            log,
            attention_backend=_attn_backend(attention),
            torch_compile=bool(use_compile),
            compile_mode=str(compile_mode),
            adapter_path=adapter_path,
        )

        audios = []
        sr_out = None
        total_turns = len(active)
        for index, (voice, text) in enumerate(active, 1):
            progress(0.25 + 0.60 * ((index - 1) / total_turns), desc=f"Turn {index}/{total_turns} · loading voice reference")
            ref, transcript = resolve_voice(voice)
            if not ref:
                raise ValueError(f"Saved voice '{voice}' has no valid audio.")
            if not (transcript or "").strip():
                raise ValueError(f"Saved voice '{voice}' needs a transcript.")
            wav, sr = load_audio(ref)
            progress(0.28 + 0.60 * ((index - 1) / total_turns), desc=f"Turn {index}/{total_turns} · synthesizing speech")
            audio, sr_out = model.generate(
                text=text,
                language=_lang(language),
                prompt_text=transcript.strip(),
                prompt_audio=wav,
                prompt_audio_sr=sr,
                inference_cfg=float(cfg),
                n_timesteps=int(steps),
                seed=actual_seed + index - 1,
                do_tn=bool(do_tn),
                do_split=False,
                cross_fade_ms=0,
            )
            audios.append(audio.detach().cpu())
            progress(0.25 + 0.60 * (index / total_turns), desc=f"Turn {index}/{total_turns} complete")

        progress(0.89, desc="Merging dialogue turns")
        pause = torch.zeros((1, max(0, int(float(pause_seconds) * sr_out))), dtype=audios[0].dtype)
        parts = []
        for index, audio in enumerate(audios):
            if index and pause.numel():
                parts.append(pause)
            parts.append(audio)
        final_audio = torch.cat(parts, dim=1)
        progress(0.96, desc="Saving dialogue WAV")
        path = save_tensor_audio(final_audio, sr_out, "dialogue")
        log(f"[done] Dialogue: {path}")
        play_completion_chime()
        progress(1.0, desc="Done")
        return path, path, f"Dialogue complete. Generated {len(active)} turns.", console_html()
    except Exception as exc:
        log(traceback.format_exc())
        raise gr.Error(str(exc))

def dataset_scan(source):
    try:
        files, with_text = scan_dataset(source)
        return f"Found {len(files)} audio files; {with_text} have sidecar .txt transcripts.", console_html()
    except Exception as exc:
        raise gr.Error(str(exc))


def dataset_prepare(project, source, language, transcribe_missing, asr_model, asr_lang, asr_batch):
    try:
        p, manifest, count = prepare_dataset(
            source, project, TRAINING_PROJECTS, language,
            asr_manager=ASR,
            asr_model=asr_model,
            asr_language=asr_lang,
            asr_batch_size=int(asr_batch),
            transcribe_missing=bool(transcribe_missing),
            log=log,
        )
        return str(manifest), f"Prepared {count} samples in {p}.", console_html()
    except Exception as exc:
        log(traceback.format_exc())
        raise gr.Error(str(exc))






TRAIN_UI_LOCK = threading.RLock()
TRAIN_UI_STATE = {
    "running": False,
    "pct": 0.0,
    "current": 0,
    "total": 0,
    "mode": "Steps",
    "text": "Idle",
    "eta_seconds": None,
    "elapsed_seconds": 0.0,
}


def _format_duration(seconds):
    if seconds is None or not math.isfinite(float(seconds)):
        return "—"
    seconds = max(0, int(round(float(seconds))))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours:d}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes:d}m {secs:02d}s"
    return f"{secs:d}s"


def _reset_training_progress(mode="Steps", text="Ready"):
    with TRAIN_UI_LOCK:
        TRAIN_UI_STATE.update(
            running=False,
            pct=0.0,
            current=0,
            total=0,
            mode=str(mode),
            text=str(text),
            eta_seconds=None,
            elapsed_seconds=0.0,
        )


def _set_training_progress(current, total, mode, text, started_at, running=True):
    current = max(0, int(current))
    total = max(1, int(total))
    elapsed = max(0.0, time.monotonic() - float(started_at))
    pct = min(100.0, max(0.0, current / total * 100.0))
    eta = None
    if current > 0 and elapsed > 0 and current < total:
        rate = current / elapsed
        if rate > 0:
            eta = (total - current) / rate
    elif current >= total:
        eta = 0.0
    with TRAIN_UI_LOCK:
        TRAIN_UI_STATE.update(
            running=bool(running),
            pct=pct,
            current=current,
            total=total,
            mode=str(mode),
            text=str(text),
            eta_seconds=eta,
            elapsed_seconds=elapsed,
        )


def training_progress_html():
    with TRAIN_UI_LOCK:
        state = dict(TRAIN_UI_STATE)
    pct = max(0.0, min(100.0, float(state.get("pct", 0.0))))
    current = int(state.get("current", 0))
    total = int(state.get("total", 0))
    mode = str(state.get("mode", "Steps"))
    text = html.escape(str(state.get("text", "Idle")))
    running = bool(state.get("running", False))
    eta = _format_duration(state.get("eta_seconds"))
    elapsed = _format_duration(state.get("elapsed_seconds", 0.0))

    if running:
        title = "Training"
        counter = f"{current:,} / {total:,} {mode.lower()}" if total else mode
    elif pct >= 100.0:
        title = "Training complete"
        counter = f"{current:,} / {total:,} {mode.lower()}" if total else mode
    else:
        title = "Training"
        counter = "Idle"

    return (
        '<div class="fire-progress-card">'
        '<div class="fire-progress-head">'
        f'<div class="fire-progress-title">{title}</div>'
        f'<div class="fire-progress-meta">{counter}</div>'
        '</div>'
        '<div class="fire-train-track">'
        f'<div class="fire-train-fill" style="width:{pct:.2f}%"></div>'
        '</div>'
        '<div class="fire-progress-foot">'
        f'<span>{pct:.1f}% · {text}</span>'
        f'<span>Elapsed {elapsed} · ETA {eta}</span>'
        '</div>'
        '</div>'
    )


def run_lora_training(
    project_name, manifest, output_name, rank, alpha, dropout, lr,
    length_mode, training_steps, training_epochs, grad_accum, seed,
    save_steps, save_epochs, resume_checkpoint,
    eval_enable, eval_text, eval_ref, eval_ref_text, eval_language,
    progress=gr.Progress(),
):
    try:
        if not manifest:
            raise ValueError("Prepare/select a manifest first.")
        if bool(eval_enable):
            if not eval_ref or not Path(str(eval_ref)).is_file():
                raise ValueError("Eval Audio is enabled: select a valid Eval Reference Audio first.")
            if not (eval_ref_text or "").strip():
                raise ValueError("Eval Audio is enabled: provide or transcribe the Eval Reference Transcript first.")
            if not (eval_text or "").strip():
                raise ValueError("Eval Audio is enabled: enter Eval Text first.")

        out_name = "".join(
            c if c.isalnum() or c in "-_" else "_"
            for c in (output_name or project_name or "adapter")
        )
        out = TRAINING_OUTPUTS / out_name
        save_every = int(save_steps) if str(length_mode) == "Steps" else int(save_epochs)
        mode_label = "Steps" if str(length_mode) == "Steps" else "Epochs"
        started_at = time.monotonic()
        with TRAIN_UI_LOCK:
            TRAIN_UI_STATE.update(
                running=True,
                pct=0.0,
                current=0,
                total=max(1, int(training_steps) if mode_label == "Steps" else int(training_epochs)),
                mode=mode_label,
                text="Initializing model and dataset...",
                eta_seconds=None,
                elapsed_seconds=0.0,
            )

        def cb(current, total, message):
            progress(min(1.0, current / max(1, total)), desc=message)
            _set_training_progress(
                current=current,
                total=total,
                mode=mode_label,
                text=message,
                started_at=started_at,
                running=True,
            )

        result, completed = train_lora(
            MANAGER,
            manifest,
            str(out),
            rank=int(rank),
            alpha=int(alpha),
            dropout=float(dropout),
            learning_rate=float(lr),
            length_mode=str(length_mode),
            training_steps=int(training_steps),
            training_epochs=int(training_epochs),
            gradient_accumulation=int(grad_accum),
            seed=int(seed),
            log=log,
            progress_cb=cb,
            save_every=save_every,
            resume_checkpoint=None if not resume_checkpoint or resume_checkpoint == "None" else resume_checkpoint,
            enable_eval_audio=bool(eval_enable),
            eval_text=str(eval_text or "This is a FireRedTTS3 training preview."),
            eval_reference_audio=eval_ref,
            eval_reference_transcript=str(eval_ref_text or ""),
            eval_language=str(eval_language or "English"),
        )
        unit = "steps" if str(length_mode) == "Steps" else "epochs"
        elapsed = max(0.0, time.monotonic() - started_at)
        with TRAIN_UI_LOCK:
            total = max(1, int(training_steps) if mode_label == "Steps" else int(training_epochs))
            TRAIN_UI_STATE.update(
                running=False,
                pct=100.0,
                current=total,
                total=total,
                mode=mode_label,
                text=f"Finished · adapter saved",
                eta_seconds=0.0,
                elapsed_seconds=elapsed,
            )
        log(f"[training] Adapter saved: {result}")
        return str(result), f"Training finished: {completed} {unit}.", console_html()
    except Exception as exc:
        with TRAIN_UI_LOCK:
            TRAIN_UI_STATE.update(
                running=False,
                text=f"Stopped / error: {type(exc).__name__}",
                eta_seconds=None,
            )
        log(traceback.format_exc())
        raise gr.Error(str(exc))



def request_stop_training():
    stop_training()
    with TRAIN_UI_LOCK:
        TRAIN_UI_STATE.update(
            running=False,
            text="Stop requested...",
            eta_seconds=None,
        )
    log("[training] Stop requested.")
    return "Stop requested.", console_html()



def _project_choices(selected="None"):
    choices = list_projects(TRAINING_PROJECTS)
    return gr.update(choices=choices, value=selected if selected in choices else "None")


def create_training_project(name):
    try:
        project, _ = project_create(TRAINING_PROJECTS, name)
        log(f"[project] Created: {project}")
        update = _project_choices(project)
        return update, update, f"Project created: {project}", console_html()
    except Exception as exc:
        raise gr.Error(str(exc))


def delete_training_project(name):
    try:
        project_delete(TRAINING_PROJECTS, name)
        log(f"[project] Deleted: {name}")
        update = _project_choices("None")
        return update, update, f"Project deleted: {name}", console_html()
    except Exception as exc:
        raise gr.Error(str(exc))


def save_training_project(
    name, source, language, manifest, output_name, preset, rank, alpha, dropout, lr,
    length_mode, training_steps, training_epochs, grad_accum, seed,
    save_steps, save_epochs,
    eval_enable, eval_text, eval_ref, eval_ref_text, eval_language,
    eval_asr_model, eval_asr_lang, eval_asr_batch,
):
    try:
        if not name or name == "None":
            raise ValueError("Create or select a Project first.")
        project_dir = TRAINING_PROJECTS / str(name)
        project_dir.mkdir(parents=True, exist_ok=True)

        persistent_eval_ref = ""
        if eval_ref:
            src = Path(str(eval_ref))
            if src.is_file():
                suffix = src.suffix.lower() or ".wav"
                dst = project_dir / f"eval_reference{suffix}"
                if src.resolve() != dst.resolve():
                    shutil.copy2(src, dst)
                persistent_eval_ref = str(dst.resolve())

        path = save_project(TRAINING_PROJECTS, name, {
            "source": source or "",
            "language": language,
            "manifest": manifest or "",
            "output_name": output_name or "",
            "vram_preset": preset,
            "rank": int(rank),
            "alpha": int(alpha),
            "dropout": float(dropout),
            "lr": float(lr),
            "length_mode": str(length_mode),
            "training_steps": int(training_steps),
            "training_epochs": int(training_epochs),
            "grad_accum": int(grad_accum),
            "seed": int(seed),
            "save_steps": int(save_steps),
            "save_epochs": int(save_epochs),
            "eval_enable": bool(eval_enable),
            "eval_text": str(eval_text or ""),
            "eval_reference_audio": persistent_eval_ref,
            "eval_reference_transcript": str(eval_ref_text or ""),
            "eval_language": str(eval_language or "English"),
            "eval_asr_model": str(eval_asr_model or ""),
            "eval_asr_language": str(eval_asr_lang or "Auto-detect"),
            "eval_asr_batch": int(eval_asr_batch),
        })
        log(f"[project] Saved: {path}")
        return f"Project saved: {path}", console_html()
    except Exception as exc:
        raise gr.Error(str(exc))



def load_training_project(name):
    d = load_project(TRAINING_PROJECTS, name)
    if not d:
        return [gr.update()] * 24
    eval_ref = d.get("eval_reference_audio", "")
    if eval_ref and not Path(eval_ref).is_file():
        eval_ref = ""
    return [
        gr.update(value=d.get("source", "")),
        gr.update(value=d.get("language", "English")),
        gr.update(value=d.get("manifest", "")),
        gr.update(value=d.get("output_name", "my_voice_lora")),
        gr.update(value=d.get("vram_preset", "24 GB")),
        gr.update(value=d.get("rank", 16)),
        gr.update(value=d.get("alpha", 32)),
        gr.update(value=d.get("dropout", 0.05)),
        gr.update(value=d.get("lr", 4e-5)),
        gr.update(value=d.get("length_mode", "Steps")),
        gr.update(value=d.get("training_steps", 1000)),
        gr.update(value=d.get("training_epochs", 30)),
        gr.update(value=d.get("grad_accum", 8)),
        gr.update(value=d.get("seed", 1234)),
        gr.update(value=d.get("save_steps", 100)),
        gr.update(value=d.get("save_epochs", 1)),
        gr.update(value=d.get("eval_enable", True)),
        gr.update(value=d.get("eval_text", "This is a FireRedTTS3 training preview.")),
        gr.update(value=eval_ref or None),
        gr.update(value=d.get("eval_reference_transcript", "")),
        gr.update(value=d.get("eval_language", "English")),
        gr.update(value=d.get("eval_asr_model", "Faster-Whisper large-v3  •  ~4.5 GB VRAM")),
        gr.update(value=d.get("eval_asr_language", "Auto-detect")),
        gr.update(value=d.get("eval_asr_batch", 8)),
    ]



def autotune_training_gui(manifest, preset):
    try:
        physical = None
        if torch.cuda.is_available():
            physical = torch.cuda.get_device_properties(0).total_memory / (1024**3)

        tuned = autotune(manifest, preset, physical)
        a = tuned["analysis"]

        sample_count = int(a["sample_count"])
        total_minutes = float(a["total_seconds"]) / 60.0
        mode = "Steps"
        suggested_steps = int(tuned["suggested_steps"])
        save_steps = int(tuned["save_steps"])
        suggested_epochs = int(tuned["suggested_epochs"])
        save_epochs = int(tuned["save_epochs"])

        status = (
            f"**AutoTune:** {sample_count} samples • {total_minutes:.1f} min • "
            f"median {a['median_seconds']:.1f}s • p90 {a['p90_seconds']:.1f}s. "
            f"**Recommended:** rank {tuned['rank']} • LR {tuned['lr']:.2g} • "
            f"gradient accumulation {tuned['grad']} • **{suggested_steps} steps** • "
            f"checkpoint/eval every {save_steps} steps. "
            f"Steps mode is preferred for predictable optimizer updates."
        )

        return (
            tuned["rank"],
            tuned["rank"] * 2,
            tuned["lr"],
            tuned["grad"],
            mode,
            gr.update(visible=True),
            gr.update(visible=False),
            suggested_steps,
            save_steps,
            suggested_epochs,
            save_epochs,
            status,
        )
    except Exception as exc:
        raise gr.Error(str(exc))



def list_training_checkpoints(output_name):
    root = TRAINING_OUTPUTS / str(output_name or "")
    out = ["None"]
    if root.is_dir():
        items=[]
        for d in root.glob("checkpoint-*"):
            if d.is_dir() and (d/"adapter_config.json").is_file():
                try: step=int(d.name.rsplit("-",1)[1])
                except Exception: step=0
                items.append((step,str(d)))
        out.extend(path for _,path in sorted(items, reverse=True))
    return out


def refresh_checkpoints(output_name, selected="None"):
    c=list_training_checkpoints(output_name)
    return gr.update(choices=c, value=selected if selected in c else "None")




def browse_dataset_folder(current_folder=""):
    """Native folder picker matching MOSS."""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root=tk.Tk(); root.withdraw(); root.attributes("-topmost",True)
        initial=current_folder if current_folder and Path(current_folder).is_dir() else str(ROOT)
        selected=filedialog.askdirectory(initialdir=initial); root.destroy()
        return selected or current_folder or ""
    except Exception as exc:
        log(f"Folder picker error: {exc}"); return current_folder or ""


TENSORBOARD_PROCESS=None
def launch_tensorboard(output_name):
    global TENSORBOARD_PROCESS
    import socket
    import time

    safe=(output_name or "my_voice_lora").strip().replace(" ","_")
    logdir=TRAINING_OUTPUTS/safe/"tensorboard"
    if not logdir.exists():
        raise gr.Error(
            f"No TensorBoard log exists yet for '{safe}'. Start training first."
        )

    url="http://127.0.0.1:6006"
    if TENSORBOARD_PROCESS is None or TENSORBOARD_PROCESS.poll() is not None:
        try:
            import pkg_resources
        except Exception as exc:
            raise gr.Error(
                "TensorBoard requires pkg_resources. Run install.bat once after Patch 12."
            ) from exc
        cmd=[
            sys.executable, "-m", "tensorboard.main",
            "--logdir", str(logdir),
            "--host", "127.0.0.1",
            "--port", "6006",
        ]
        log("[tensorboard] "+subprocess.list2cmdline(cmd))
        TENSORBOARD_PROCESS=subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

        # Wait briefly for the local server so the browser does not open on a
        # connection-refused page. This callback bypasses the Gradio queue.
        deadline=time.time()+8.0
        while time.time()<deadline:
            if TENSORBOARD_PROCESS.poll() is not None:
                output=""
                try:
                    output=TENSORBOARD_PROCESS.stdout.read() if TENSORBOARD_PROCESS.stdout else ""
                except Exception:
                    pass
                raise gr.Error(f"TensorBoard exited during startup. {output[-1200:]}")
            try:
                with socket.create_connection(("127.0.0.1",6006),timeout=0.25):
                    break
            except OSError:
                time.sleep(0.15)
        else:
            raise gr.Error("TensorBoard did not open port 6006 within 8 seconds.")

    webbrowser.open(url)
    log(f"[tensorboard] Ready: {url} · {logdir}")
    return f"TensorBoard running: {url}\nLogdir: {logdir}"


def training_length_mode_ui(mode):
    is_steps = str(mode) == "Steps"
    return gr.update(visible=is_steps), gr.update(visible=not is_steps)



ORANGE_PROGRESS_CSS = """
"""

with gr.Blocks(title=APP_TITLE) as demo:
    with gr.Row(elem_classes="title-section"):
        with gr.Column():
            gr.Markdown("# 🔥 FireRedTTS3 Easy GUI")
            gr.Markdown(
                "FireRedTTS3 Base + Instruct • local voice cloning, voice design and speech editing",
                elem_classes="tab-subtitle",
            )
    model_status = gr.State("Ready.")

    with gr.Row(elem_classes="global-toolbar"):
        unload_all_btn = gr.Button("🧹 Unload All Models", size="sm", variant="secondary")
        clear_outputs_btn = gr.Button("🗑️ Clear Outputs", size="sm", variant="stop")

    with gr.Tabs(elem_classes="tabs"):
        with gr.Tab("🎙️ Prep Samples"):
            with gr.Accordion("📖 Quick Guide", open=False, elem_classes=["quick-guide"]):
                gr.Markdown("""
Build a reusable **Voice Library** for zero-shot cloning and Dialogue Builder.

1. Upload/record a clean reference.
2. Enter its exact transcript or use Faster-Whisper.
3. Save it with a recognizable name.
4. The voice becomes immediately available in TTS / Voice Clone and Dialogue Builder.

Clean, single-speaker references and exact transcripts produce the most stable cloning.
""")
            with gr.Row():
                with gr.Column(scale=1, elem_classes="form-section"):
                    gr.Markdown("#### 📚 Voice Library")
                    with gr.Row():
                        voice_choice = gr.Dropdown(choices=list_voices(), value=NONE_VOICE, label="Saved Voice", scale=8)
                        voice_refresh = gr.Button("🔄", size="sm", scale=1)
                    voice_delete = gr.Button("🗑️ Delete", size="sm", variant="stop")
                    voice_msg = gr.Textbox(label="Voice Library Status", interactive=False)
                with gr.Column(scale=2, elem_classes="form-section"):
                    gr.Markdown("#### 🎙️ Prepare Sample")
                    voice_name = gr.Textbox(label="Voice Name", placeholder="Speaker name")
                    clone_ref = gr.Audio(label="Reference Audio", sources=["upload","microphone"], type="filepath", elem_classes=["audio-safe-space"])
                    clone_ref_text = gr.Textbox(label="Reference Transcript", lines=4)
                    with gr.Accordion("🛰️ Faster-Whisper Transcription (Optional)", open=False):
                        asr_model = gr.Dropdown(choices=list(WHISPER_MODELS), value="Faster-Whisper large-v3  •  ~4.5 GB VRAM", label="Model")
                        asr_lang = gr.Dropdown(choices=list(WHISPER_LANGS), value="Auto-detect", label="Language")
                        asr_batch = gr.Slider(1,32,value=8,step=1,label="Batch Size")
                        voice_transcribe = gr.Button("Transcribe Now")
                        asr_status = gr.Textbox(label="Transcription Status", interactive=False)
                    voice_save = gr.Button("💾 Save Voice", variant="primary", elem_classes="green-btn")

        with gr.Tab("🔊 Inference"):
            with gr.Accordion("📖 Quick Guide", open=False, elem_classes=["quick-guide"]):
                gr.Markdown("""
Models download **on demand**. FireRedTTS3 natively uses FlashAttention 2 in its Qwen/RedAE attention blocks.  
Use **PyTorch SDPA** only as a compatibility fallback. **torch.compile** is optional/experimental and compiles the repeatedly used DiT + PatchEncoder modules; the first generation can be slower while compilation occurs.
""")
            gr.Markdown("### 🚀 Acceleration Engines")
            with gr.Row(elem_classes="form-section"):
                acceleration_attn = gr.Dropdown(
                    ["FlashAttention 2 (Native / Recommended)", "PyTorch SDPA (Compatibility)"],
                    value="FlashAttention 2 (Native / Recommended)", label="Attention Backend",
                )
                acceleration_compile = gr.Checkbox(value=False, label="torch.compile (Experimental)")
                acceleration_compile_mode = gr.Dropdown(
                    ["default","reduce-overhead"],
                    value="default", label="Compile Mode",
                )

            with gr.Tabs():
                with gr.Tab("TTS / Voice Clone"):
                    with gr.Accordion("📖 Quick Guide",open=False,elem_classes=["quick-guide"]):
                        gr.Markdown("""
Choose a saved voice or provide a custom reference plus its exact transcript.

Long-form generation uses the MOSS-style **Chunk Mode**: None, Paragraph/Sentence Auto, Periods, Paragraphs, Lines or Speaker turns. Each chunk is synthesized independently and merged with the selected silence.

A trained LoRA adapter can be selected directly from the adapter dropdown.
""")
                    gr.Markdown("### 🎙️ Zero-Shot Voice Cloning")
                    with gr.Row():
                        with gr.Column(scale=1,elem_classes="form-section"):
                            with gr.Row():
                                tts_voice=gr.Dropdown(choices=list_voices(),value=NONE_VOICE,label="Voice Library",scale=8)
                                tts_voice_refresh=gr.Button("🔄",size="sm",scale=1)
                            tts_ref=gr.Audio(label="Custom Reference Audio",sources=["upload","microphone"],type="filepath",elem_classes=["audio-safe-space"])
                            tts_ref_text=gr.Textbox(label="Reference Transcript",lines=3)
                            tts_lang=gr.Dropdown(choices=LANGUAGES,value="Auto-detect",label="Language / Dialect",info="Auto-detect may choose an incorrect accent. For best pronunciation and accent, select the target language explicitly.")
                            with gr.Row():
                                tts_adapter=gr.Dropdown(choices=_adapter_choices(),value="None",label="LoRA Adapter",scale=8)
                                tts_adapter_refresh=gr.Button("🔄",size="sm",scale=1)
                        with gr.Column(scale=2,elem_classes="form-section"):
                            tts_text=gr.Textbox(label="Text",lines=9)
                            with gr.Accordion("⚙️ Generation Parameters",open=False):
                                tts_cfg=gr.Slider(0.5,4.0,value=2.0,step=0.1,label="Inference CFG")
                                tts_steps=gr.Slider(1,50,value=10,step=1,label="Diffusion Timesteps")
                                with gr.Row():
                                    tts_seed=gr.Number(value=1234,precision=0,label="Seed")
                                    tts_random_seed=gr.Checkbox(value=True,label="Random Seed")
                                tts_tn=gr.Checkbox(value=True,label="Text Normalization")
                            with gr.Accordion("📚 Long Text / Chunking",open=False):
                                tts_split=gr.Dropdown(CHUNK_CHOICES,value="None",label="Chunk Mode")
                                tts_pause=gr.Slider(0,5,value=0.25,step=0.05,label="Silence Between Chunks (seconds)")
                            tts_go=gr.Button("Generate Speech",variant="primary",size="lg")
                            tts_audio=gr.Audio(label="Generated Speech",type="filepath",elem_classes=["output-clean","audio-safe-space"])
                            tts_path=gr.Textbox(label="Saved WAV",interactive=False,elem_classes="output-path")

                with gr.Tab("Dialogue Builder"):
                    with gr.Accordion("📖 Quick Guide",open=False,elem_classes=["quick-guide"]):
                        gr.Markdown("""
Each row is synthesized independently with FireRedTTS3-Base and the resulting WAVs are joined in order, matching the MOSS Classic Dialogue Builder workflow.

Select a saved voice for each non-empty turn. **Pause Between Turns** is global. The selected LoRA adapter is reused across all turns.
""")
                    dialogue_count=gr.State(2)
                    with gr.Row(elem_classes="dialogue-toolbar"):
                        d_reset=gr.Button("Reset rows",size="sm",variant="secondary")
                        d_clear_all=gr.Button("Clear rows",size="sm",variant="secondary")
                        d_compact=gr.Button("Remove empty rows",size="sm",variant="secondary")
                        d_voice_refresh=gr.Button("🔄 Refresh Voices",size="sm")
                    d_values=[]; d_groups=[]; d_buttons=[]
                    for i in range(DIALOGUE_MAX_TURNS):
                        with gr.Group(visible=i<2,elem_classes="dialogue-turn-card") as grp:
                            with gr.Row():
                                with gr.Column(scale=2,min_width=180):
                                    dv=gr.Dropdown(choices=list_voices(),value=NONE_VOICE,label=f"Turn {i+1} · Voice")
                                with gr.Column(scale=6):
                                    dt=gr.Textbox(label=f"Text {i+1}",placeholder=f"Enter dialogue text for turn {i+1}...",lines=2)
                                with gr.Column(scale=2,min_width=210,elem_classes="dialogue-actions"):
                                    with gr.Row():
                                        b_add=gr.Button("➕",size="sm"); b_clone=gr.Button("📋",size="sm")
                                        b_up=gr.Button("⬆️",size="sm"); b_down=gr.Button("⬇️",size="sm")
                                    with gr.Row():
                                        b_clear=gr.Button("🧹",size="sm"); b_delete=gr.Button("🗑️",size="sm",variant="stop")
                        d_values.extend([dv,dt]); d_groups.append(grp); d_buttons.append((b_add,b_clone,b_up,b_down,b_clear,b_delete))
                    with gr.Row():
                        d_lang=gr.Dropdown(choices=LANGUAGES,value="Auto-detect",label="Language / Dialect",info="Auto-detect may choose an incorrect accent. For best pronunciation and accent, select the target language explicitly.")
                        d_pause=gr.Slider(0,5,value=0.5,step=0.05,label="Pause Between Turns (seconds)")
                    with gr.Accordion("⚙️ Generation Parameters",open=False):
                        d_cfg=gr.Slider(0.5,4.0,value=2.0,step=0.1,label="Inference CFG")
                        d_steps=gr.Slider(1,50,value=10,step=1,label="Diffusion Timesteps")
                        with gr.Row():
                            d_seed=gr.Number(value=1234,precision=0,label="Seed")
                            d_random_seed=gr.Checkbox(value=True,label="Random Seed")
                        d_tn=gr.Checkbox(value=True,label="Text Normalization")
                        with gr.Row():
                            d_adapter=gr.Dropdown(choices=_adapter_choices(),value="None",label="LoRA Adapter",scale=8)
                            d_adapter_refresh=gr.Button("🔄",size="sm",scale=1)
                    d_go=gr.Button("⚡ Generate Dialogue",variant="primary",size="lg")
                    d_audio=gr.Audio(label="Generated Dialogue",type="filepath",elem_classes=["output-clean","audio-safe-space"])
                    d_path=gr.Textbox(label="Saved WAV",interactive=False,elem_classes="output-path")
                    d_status=gr.Textbox(label="Dialogue Status",interactive=False,lines=2)

                with gr.Tab("Voice Design"):
                    with gr.Accordion("📖 Quick Guide",open=False,elem_classes=["quick-guide"]):
                        gr.Markdown("""
Describe the desired voice and enter the target text. **Language / Dialect** controls the text frontend.

Long text uses the same MOSS-style Chunk Mode as Voice Clone.
""")
                    gr.Markdown("### 🎨 Instruction-Controlled Voice Design")
                    with gr.Row():
                        with gr.Column(elem_classes="form-section"):
                            design_instruction=gr.Textbox(label="Voice Description",lines=5)
                            design_text=gr.Textbox(label="Text",lines=8)
                            design_lang=gr.Dropdown(choices=LANGUAGES,value="Auto-detect",label="Language / Dialect",info="Auto-detect may choose an incorrect accent. For best pronunciation and accent, select the target language explicitly.")
                            with gr.Accordion("⚙️ Generation Parameters",open=False):
                                design_cfg=gr.Slider(0.5,4.0,value=1.2,step=0.1,label="Inference CFG")
                                design_steps=gr.Slider(1,50,value=10,step=1,label="Diffusion Timesteps")
                                with gr.Row():
                                    design_seed=gr.Number(value=2,precision=0,label="Seed")
                                    design_random_seed=gr.Checkbox(value=True,label="Random Seed")
                                design_tn=gr.Checkbox(value=True,label="Text Normalization")
                            with gr.Accordion("📚 Long Text / Chunking",open=False):
                                design_split=gr.Dropdown(CHUNK_CHOICES,value="None",label="Chunk Mode")
                                design_pause=gr.Slider(0,5,value=0.25,step=0.05,label="Silence Between Chunks (seconds)")
                            design_go=gr.Button("Generate Voice",variant="primary")
                        with gr.Column(elem_classes="form-section"):
                            design_audio=gr.Audio(label="Generated Voice",type="filepath",elem_classes=["output-clean","audio-safe-space"])
                            design_plan=gr.Textbox(label="Generated Voice Plan",lines=6,interactive=False)
                            design_path=gr.Textbox(label="Saved WAV",interactive=False,elem_classes="output-path")

                with gr.Tab("Semantic Edit"):
                    with gr.Accordion("📖 Quick Guide",open=False,elem_classes=["quick-guide"]):
                        gr.Markdown("""
Semantic Edit works from **audio + natural-language edit instruction**. The released FireRedTTS3 API does not expose a language argument for this task.

Use the optional Faster-Whisper panel to transcribe the source speech and confirm/detect its language before editing. The transcript is diagnostic/context only; FireRedTTS3-Instruct consumes the audio directly.
""")
                    gr.Markdown("### ✂️ Semantic Speech Editing")
                    with gr.Row():
                        with gr.Column(elem_classes="form-section"):
                            semantic_audio=gr.Audio(label="Input Speech",sources=["upload","microphone"],type="filepath",elem_classes=["audio-safe-space"])
                            semantic_instruction=gr.Textbox(label="Edit Instruction",lines=5)
                            with gr.Accordion("🛰️ Transcription / Language Detection (Optional)",open=False):
                                sem_asr_model=gr.Dropdown(choices=list(WHISPER_MODELS),value="Faster-Whisper large-v3  •  ~4.5 GB VRAM",label="Model")
                                sem_asr_lang=gr.Dropdown(choices=list(WHISPER_LANGS),value="Auto-detect",label="Speech Language")
                                sem_asr_batch=gr.Slider(1,32,value=8,step=1,label="Batch Size")
                                sem_transcribe=gr.Button("Transcribe / Detect")
                                sem_transcript=gr.Textbox(label="Detected Transcript",lines=5,interactive=False)
                                sem_detected=gr.Textbox(label="Detection Status",interactive=False)
                            with gr.Accordion("⚙️ Generation Parameters",open=False):
                                semantic_cfg=gr.Slider(0.5,4.0,value=1.2,step=0.1,label="Inference CFG")
                                semantic_steps=gr.Slider(1,50,value=10,step=1,label="Diffusion Timesteps")
                                with gr.Row():
                                    semantic_seed=gr.Number(value=1234,precision=0,label="Seed")
                                    semantic_random_seed=gr.Checkbox(value=True,label="Random Seed")
                            semantic_go=gr.Button("Apply Semantic Edit",variant="primary")
                        with gr.Column(elem_classes="form-section"):
                            semantic_out=gr.Audio(label="Edited Speech",type="filepath",elem_classes=["output-clean","audio-safe-space"])
                            semantic_text=gr.Textbox(label="Model Edit Text / Mask",interactive=False,lines=6)
                            semantic_path=gr.Textbox(label="Saved WAV",interactive=False,elem_classes="output-path")

                with gr.Tab("Acoustic Edit"):
                    with gr.Accordion("📖 Quick Guide",open=False,elem_classes=["quick-guide"]):
                        gr.Markdown("Acoustic Edit changes speed, pitch or volume without rewriting the words. Random Seed and the common acceleration backend apply here as well.")
                    gr.Markdown("### 🎛️ Acoustic Speech Editing")
                    with gr.Row():
                        with gr.Column(elem_classes="form-section"):
                            acoustic_audio=gr.Audio(label="Input Speech",sources=["upload","microphone"],type="filepath",elem_classes=["audio-safe-space"])
                            acoustic_attr=gr.Radio(["Speed","Pitch","Volume"],value="Speed",label="Attribute")
                            acoustic_value=gr.Slider(0.5,2.0,value=1.0,step=0.1,label="Speed")
                            with gr.Accordion("⚙️ Generation Parameters",open=False):
                                acoustic_cfg=gr.Slider(0.5,4.0,value=1.2,step=0.1,label="Inference CFG")
                                acoustic_steps=gr.Slider(1,50,value=10,step=1,label="Diffusion Timesteps")
                                with gr.Row():
                                    acoustic_seed=gr.Number(value=1234,precision=0,label="Seed")
                                    acoustic_random_seed=gr.Checkbox(value=True,label="Random Seed")
                            acoustic_go=gr.Button("Apply Acoustic Edit",variant="primary")
                        with gr.Column(elem_classes="form-section"):
                            acoustic_out=gr.Audio(label="Edited Speech",type="filepath",elem_classes=["output-clean","audio-safe-space"])
                            acoustic_instruction=gr.Textbox(label="Exact Model Instruction",interactive=False)
                            acoustic_path=gr.Textbox(label="Saved WAV",interactive=False,elem_classes="output-path")

        with gr.Tab("📂 Dataset Preparation"):
            with gr.Accordion("📖 Quick Guide",open=False,elem_classes=["quick-guide"]):
                gr.Markdown("""
Prepare a reusable **voice-clone LoRA project**, following the same project workflow as MOSS.

1. Create or select a **Project**. Dataset and training settings are shared through the same project file.
2. Select the source folder containing single-speaker audio clips. Matching same-stem `.txt` transcripts are reused.
3. Optionally transcribe missing sidecars with Faster-Whisper.
4. Choose the correct FireRedTTS3 language/dialect and click **Prepare Dataset**.
5. The prepared JSONL is stored under `training/projects/<project>/` and is picked up by LoRA Training.

FireRedTTS3 has not published an official fine-tuning recipe; this remains an experimental Base/voice-clone workflow reconstructed from the released model graph.
""")
            with gr.Row(elem_classes="project-strip"):
                dataset_project=gr.Dropdown(choices=list_projects(TRAINING_PROJECTS),value="None",label="Project")
                dataset_new_project=gr.Textbox(label="New Project Name",placeholder="my_voice")
                dataset_create_project=gr.Button("Create Project")
                dataset_save_project=gr.Button("💾 Save Project",variant="secondary")
                dataset_delete_project=gr.Button("🗑️ Delete Project",variant="stop")
            with gr.Row():
                with gr.Row():
                    dataset_source=gr.Textbox(label="Source Audio Folder",scale=8)
                    dataset_browse_btn=gr.Button("📁 Browse",size="sm",scale=1)
                dataset_language=gr.Dropdown(choices=[x for x in LANGUAGES if x!="Auto-detect"],value="English",label="Language / Dialect")
            dataset_scan_btn=gr.Button("Build / Scan Dataset",variant="secondary")
            dataset_scan_status=gr.Textbox(label="Scan Status",interactive=False)
            with gr.Accordion("🛰️ Missing Transcript ASR",open=False):
                dataset_transcribe=gr.Checkbox(value=True,label="Transcribe Missing .txt Files")
                dataset_asr_model=gr.Dropdown(choices=list(WHISPER_MODELS),value="Faster-Whisper large-v3  •  ~4.5 GB VRAM",label="Model")
                dataset_asr_lang=gr.Dropdown(choices=list(WHISPER_LANGS),value="Auto-detect",label="Language")
                dataset_asr_batch=gr.Slider(1,32,value=8,step=1,label="Batch Size")
            dataset_prepare_btn=gr.Button("Prepare Dataset",variant="primary")
            dataset_manifest=gr.Textbox(label="Prepared Manifest",interactive=False)
            dataset_status=gr.Textbox(label="Project / Dataset Status",interactive=False)

        with gr.Tab("🚀 LoRA Training"):
            with gr.Accordion("📖 Quick Guide",open=False,elem_classes=["quick-guide"]):
                gr.Markdown("""
**Experimental — FireRedTTS3-Base voice cloning only.**

Choose one **Training Length Mode**:
- **Steps** → train for exactly N steps and save/evaluate every N steps.
- **Epochs** → train for exactly N epochs and save/evaluate every N epochs.

Only the controls for the selected mode are shown. Gradient accumulation and dataset-derived prompt/target audio limits are handled internally.
""")

            with gr.Row(elem_classes="project-strip"):
                with gr.Column(scale=4):
                    train_project=gr.Dropdown(
                        choices=list_projects(TRAINING_PROJECTS),value="None",label="Project"
                    )
                with gr.Column(scale=3):
                    train_preset=gr.Dropdown(
                        choices=list(VRAM_PRESETS),value="24 GB",label="VRAM Preset"
                    )
                with gr.Column(scale=2,min_width=155):
                    train_autotune=gr.Button("⚡ AutoTune",variant="secondary")
                with gr.Column(scale=2,min_width=155):
                    train_save_project=gr.Button("💾 Save Project",variant="secondary")
                with gr.Column(scale=2,min_width=155):
                    train_delete_project=gr.Button("🗑️ Delete Project",variant="stop")
            train_autotune_status=gr.Markdown()

            with gr.Row():
                with gr.Column(scale=1,elem_classes="form-section"):
                    gr.Markdown("#### 📦 Training Setup")
                    train_manifest=gr.Textbox(label="Prepared Dataset",interactive=False)
                    train_output=gr.Textbox(label="Adapter Name",value="my_voice_lora")
                    train_length_mode=gr.Dropdown(
                        ["Steps","Epochs"],value="Steps",label="Training Length Mode"
                    )

                    with gr.Group(visible=True) as train_steps_group:
                        with gr.Row():
                            train_steps=gr.Slider(
                                1,10000,value=1000,step=1,label="Training Steps"
                            )
                            train_save_steps=gr.Slider(
                                1,2000,value=100,step=1,label="Save Every N Steps"
                            )

                    with gr.Group(visible=False) as train_epochs_group:
                        with gr.Row():
                            train_epochs=gr.Slider(
                                1,2000,value=30,step=1,label="Training Epochs"
                            )
                            train_save_epochs=gr.Slider(
                                1,200,value=1,step=1,label="Save Every N Epochs"
                            )

                    with gr.Row():
                        train_resume=gr.Dropdown(
                            choices=["None"],value="None",label="Resume Checkpoint",scale=8
                        )
                        train_resume_refresh=gr.Button("🔄",size="sm",scale=1)

                with gr.Column(scale=1,elem_classes="form-section"):
                    gr.Markdown("#### ⚙️ LoRA Hyperparameters")
                    with gr.Row():
                        train_rank=gr.Dropdown([4,8,16,32,64],value=16,label="LoRA Rank")
                        train_alpha=gr.Slider(4,128,value=32,step=1,label="LoRA Alpha")
                    train_dropout=gr.Slider(0,0.2,value=0.05,step=0.01,label="LoRA Dropout")
                    train_lr=gr.Slider(
                        minimum=0.000001,
                        maximum=0.001,
                        value=0.00004,
                        step=0.000001,
                        label="Learning Rate",
                    )
                    with gr.Row():
                        train_grad_accum=gr.Slider(
                            1,64,value=8,step=1,label="Gradient Accumulation"
                        )
                        train_seed=gr.Number(value=1234,precision=0,label="Training Seed")

            with gr.Accordion("🎧 Eval Audio + TensorBoard",open=False):
                with gr.Row():
                    with gr.Column(scale=1):
                        train_eval_enable=gr.Checkbox(
                            value=True,label="Generate Eval Audio at Checkpoints"
                        )
                        train_eval_text=gr.Textbox(
                            value="This is a FireRedTTS3 training preview.",
                            label="Eval Text",lines=3,
                        )
                        train_eval_language=gr.Dropdown(
                            choices=[x for x in LANGUAGES if x!="Auto-detect"],
                            value="English",label="Eval Language",
                        )
                    with gr.Column(scale=1):
                        train_eval_ref=gr.Audio(
                            type="filepath",label="Eval Reference Audio",
                            elem_classes=["audio-safe-space"]
                        )
                        train_eval_ref_text=gr.Textbox(
                            label="Eval Reference Transcript",lines=3
                        )
                with gr.Row():
                    train_eval_asr_model=gr.Dropdown(
                        choices=list(WHISPER_MODELS),
                        value="Faster-Whisper large-v3  •  ~4.5 GB VRAM",
                        label="Transcription Model",
                    )
                    train_eval_asr_lang=gr.Dropdown(
                        choices=list(WHISPER_LANGS),value="Auto-detect",label="Speech Language"
                    )
                    train_eval_asr_batch=gr.Slider(
                        1,32,value=8,step=1,label="Batch Size"
                    )
                train_eval_transcribe=gr.Button(
                    "🛰️ Transcribe Eval Audio",variant="secondary"
                )
                train_eval_asr_status=gr.Textbox(
                    label="Eval Transcription Status",interactive=False
                )

            with gr.Row(elem_classes="global-toolbar"):
                train_btn=gr.Button("🚀 Start Training",variant="primary")
                train_stop=gr.Button("⏹️ Stop",variant="stop")
                tensorboard_btn=gr.Button("📊 TensorBoard",variant="secondary")
            train_progress=gr.HTML(training_progress_html())
            tensorboard_status=gr.Textbox(label="TensorBoard",interactive=False)
            with gr.Row():
                train_adapter_path=gr.Textbox(label="Saved Adapter",interactive=False)
                train_status=gr.Textbox(label="Training Status",interactive=False)

    with gr.Accordion("🖥️ Console",open=True,elem_classes=["console-accordion"]):
        cmd_console=gr.HTML(console_html())
    timer=gr.Timer(0.5,active=True)
    timer.tick(console_html,outputs=cmd_console)
    timer.tick(training_progress_html,outputs=train_progress)

    unload_all_btn.click(unload_all,outputs=[model_status,cmd_console])
    clear_outputs_btn.click(clear_outputs,outputs=[model_status,cmd_console])

    voice_choice.change(select_voice,inputs=voice_choice,outputs=[clone_ref,clone_ref_text])
    voice_refresh.click(refresh_voice_dropdown,inputs=voice_choice,outputs=voice_choice)
    voice_save.click(add_voice,inputs=[voice_name,clone_ref,clone_ref_text],outputs=[voice_choice,voice_msg,cmd_console])
    voice_delete.click(remove_voice,inputs=voice_choice,outputs=[voice_choice,clone_ref,clone_ref_text,cmd_console])
    voice_transcribe.click(transcribe_reference,inputs=[clone_ref,asr_model,asr_lang,asr_batch],outputs=[clone_ref_text,asr_status,cmd_console])

    tts_voice.change(select_voice,inputs=tts_voice,outputs=[tts_ref,tts_ref_text])
    tts_voice_refresh.click(refresh_voice_dropdown,inputs=tts_voice,outputs=tts_voice)
    tts_adapter_refresh.click(refresh_adapters,inputs=tts_adapter,outputs=tts_adapter)
    tts_go.click(
        clone_voice,
        inputs=[
            tts_voice,tts_ref,tts_ref_text,tts_text,tts_lang,tts_cfg,tts_steps,
            tts_seed,tts_random_seed,tts_tn,tts_split,tts_pause,
            acceleration_attn,acceleration_compile,acceleration_compile_mode,tts_adapter
        ],
        outputs=[tts_audio,tts_path,cmd_console],
    )

    d_outputs=[dialogue_count,*d_values,*d_groups,d_status]
    d_inputs=[dialogue_count,*d_values]
    d_reset.click(dialogue_reset_rows,outputs=d_outputs)
    d_clear_all.click(dialogue_clear_rows,inputs=d_inputs,outputs=d_outputs)
    d_compact.click(dialogue_compact_rows,inputs=d_inputs,outputs=d_outputs)
    d_voice_refresh.click(lambda: [gr.update(choices=list_voices()) for _ in range(DIALOGUE_MAX_TURNS)],outputs=d_values[::2])
    d_adapter_refresh.click(refresh_adapters,inputs=d_adapter,outputs=d_adapter)
    for i,(b_add,b_clone,b_up,b_down,b_clear,b_delete) in enumerate(d_buttons):
        b_add.click(lambda count,*vals,_i=i: dialogue_add_after(_i,count,*vals),inputs=d_inputs,outputs=d_outputs)
        b_clone.click(lambda count,*vals,_i=i: dialogue_clone_row(_i,count,*vals),inputs=d_inputs,outputs=d_outputs)
        b_up.click(lambda count,*vals,_i=i: dialogue_move_row(_i,-1,count,*vals),inputs=d_inputs,outputs=d_outputs)
        b_down.click(lambda count,*vals,_i=i: dialogue_move_row(_i,1,count,*vals),inputs=d_inputs,outputs=d_outputs)
        b_clear.click(lambda count,*vals,_i=i: dialogue_clear_row(_i,count,*vals),inputs=d_inputs,outputs=d_outputs)
        b_delete.click(lambda count,*vals,_i=i: dialogue_delete_row(_i,count,*vals),inputs=d_inputs,outputs=d_outputs)
    d_go.click(
        generate_dialogue,
        inputs=[
            d_lang,d_pause,d_cfg,d_steps,d_seed,d_random_seed,d_tn,
            acceleration_attn,acceleration_compile,acceleration_compile_mode,d_adapter,
            dialogue_count,*d_values
        ],
        outputs=[d_audio,d_path,d_status,cmd_console],
    )

    design_go.click(
        voice_design,
        inputs=[design_instruction,design_text,design_lang,design_cfg,design_steps,design_seed,design_random_seed,design_tn,
                design_split,design_pause,
                acceleration_attn,acceleration_compile,acceleration_compile_mode],
        outputs=[design_audio,design_plan,design_path,cmd_console],
    )
    sem_transcribe.click(transcribe_reference,inputs=[semantic_audio,sem_asr_model,sem_asr_lang,sem_asr_batch],outputs=[sem_transcript,sem_detected,cmd_console])
    semantic_go.click(
        semantic_edit,
        inputs=[semantic_audio,semantic_instruction,semantic_cfg,semantic_steps,semantic_seed,semantic_random_seed,
                acceleration_attn,acceleration_compile,acceleration_compile_mode],
        outputs=[semantic_out,semantic_text,semantic_path,cmd_console],
    )
    acoustic_attr.change(acoustic_slider,inputs=acoustic_attr,outputs=acoustic_value)
    acoustic_go.click(
        acoustic_edit,
        inputs=[acoustic_audio,acoustic_attr,acoustic_value,acoustic_cfg,acoustic_steps,acoustic_seed,acoustic_random_seed,
                acceleration_attn,acceleration_compile,acceleration_compile_mode],
        outputs=[acoustic_out,acoustic_instruction,acoustic_path,cmd_console],
    )

    dataset_browse_btn.click(browse_dataset_folder,inputs=dataset_source,outputs=dataset_source)
    dataset_scan_btn.click(dataset_scan,inputs=dataset_source,outputs=[dataset_scan_status,cmd_console])
    dataset_create_project.click(create_training_project,inputs=dataset_new_project,outputs=[dataset_project,train_project,dataset_status,cmd_console])
    dataset_delete_project.click(delete_training_project,inputs=dataset_project,outputs=[dataset_project,train_project,dataset_status,cmd_console])
    train_delete_project.click(delete_training_project,inputs=train_project,outputs=[dataset_project,train_project,train_status,cmd_console])

    project_load_outputs=[dataset_source,dataset_language,dataset_manifest,train_output,train_preset,train_rank,train_alpha,train_dropout,train_lr,train_length_mode,train_steps,train_epochs,train_grad_accum,train_seed,train_save_steps,train_save_epochs,train_eval_enable,train_eval_text,train_eval_ref,train_eval_ref_text,train_eval_language,train_eval_asr_model,train_eval_asr_lang,train_eval_asr_batch]
    dataset_project.change(load_training_project,inputs=dataset_project,outputs=project_load_outputs)
    train_project.change(load_training_project,inputs=train_project,outputs=project_load_outputs)

    project_save_inputs=[train_project,dataset_source,dataset_language,dataset_manifest,train_output,train_preset,train_rank,train_alpha,train_dropout,train_lr,train_length_mode,train_steps,train_epochs,train_grad_accum,train_seed,train_save_steps,train_save_epochs,train_eval_enable,train_eval_text,train_eval_ref,train_eval_ref_text,train_eval_language,train_eval_asr_model,train_eval_asr_lang,train_eval_asr_batch]
    train_save_project.click(save_training_project,inputs=project_save_inputs,outputs=[train_status,cmd_console])
    dataset_save_project.click(
        save_training_project,
        inputs=[dataset_project,dataset_source,dataset_language,dataset_manifest,train_output,train_preset,train_rank,train_alpha,train_dropout,train_lr,train_length_mode,train_steps,train_epochs,train_grad_accum,train_seed,train_save_steps,train_save_epochs,train_eval_enable,train_eval_text,train_eval_ref,train_eval_ref_text,train_eval_language,train_eval_asr_model,train_eval_asr_lang,train_eval_asr_batch],
        outputs=[dataset_status,cmd_console],
    )

    dataset_prepare_btn.click(
        dataset_prepare,
        inputs=[dataset_project,dataset_source,dataset_language,dataset_transcribe,dataset_asr_model,dataset_asr_lang,dataset_asr_batch],
        outputs=[dataset_manifest,dataset_status,cmd_console],
    )
    dataset_manifest.change(lambda x:x,inputs=dataset_manifest,outputs=train_manifest)
    train_autotune.click(
        autotune_training_gui,
        inputs=[train_manifest,train_preset],
        outputs=[
            train_rank,train_alpha,train_lr,train_grad_accum,
            train_length_mode,train_steps_group,train_epochs_group,
            train_steps,train_save_steps,train_epochs,train_save_epochs,
            train_autotune_status,
        ],
    )
    train_resume_refresh.click(refresh_checkpoints,inputs=[train_output,train_resume],outputs=train_resume)
    train_length_mode.change(training_length_mode_ui,inputs=train_length_mode,outputs=[train_steps_group,train_epochs_group])
    train_btn.click(
        run_lora_training,
        inputs=[
            train_project,train_manifest,train_output,train_rank,train_alpha,train_dropout,train_lr,
            train_length_mode,train_steps,train_epochs,train_grad_accum,train_seed,
            train_save_steps,train_save_epochs,train_resume,
            train_eval_enable,train_eval_text,train_eval_ref,train_eval_ref_text,train_eval_language
        ],
        outputs=[train_adapter_path,train_status,cmd_console],
    )
    train_stop.click(request_stop_training,outputs=[train_status,cmd_console],queue=False)
    train_eval_transcribe.click(
        transcribe_reference,
        inputs=[train_eval_ref,train_eval_asr_model,train_eval_asr_lang,train_eval_asr_batch],
        outputs=[train_eval_ref_text,train_eval_asr_status,cmd_console],
    )
    tensorboard_btn.click(launch_tensorboard,inputs=train_output,outputs=tensorboard_status,queue=False)

if __name__=="__main__":
    log(f"[system] Torch {torch.__version__}; CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log(f"[system] GPU: {torch.cuda.get_device_name(0)}")
    demo.queue(default_concurrency_limit=1).launch(
        inbrowser=True,server_name="127.0.0.1",show_error=True,
        allowed_paths=[str(OUTPUTS_DIR),str(ROOT/"voices"),str(TRAINING_ROOT)],css=CSS + ORANGE_PROGRESS_CSS,
    )
