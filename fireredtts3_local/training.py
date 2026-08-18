from __future__ import annotations

import json
import math
import random
import shutil
import threading
import warnings
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
import torchaudio


warnings.filterwarnings(
    "ignore",
    message=r"In 2\.9, this function's implementation will be changed to use .*torchaudio\.load_with_torchcodec.*",
    category=UserWarning,
)

AUDIO_EXTS = {".wav", ".flac", ".mp3", ".m4a", ".ogg", ".opus"}
STOP_EVENT = threading.Event()


def _safe_name(name: str) -> str:
    out = "".join(c if c.isalnum() or c in "-_" else "_" for c in (name or "").strip())
    return out.strip("_") or "project"


def scan_dataset(source_dir: str):
    src = Path(source_dir).expanduser()
    if not src.exists():
        raise ValueError("Dataset source folder does not exist.")
    files = sorted(p for p in src.rglob("*") if p.is_file() and p.suffix.lower() in AUDIO_EXTS)
    with_text = sum(1 for p in files if p.with_suffix(".txt").exists())
    return files, with_text


def prepare_dataset(
    source_dir: str,
    project_name: str,
    projects_root: Path,
    language: str,
    asr_manager=None,
    asr_model: str | None = None,
    asr_language: str = "Auto-detect",
    asr_batch_size: int = 8,
    transcribe_missing: bool = True,
    log=print,
):
    files, with_text = scan_dataset(source_dir)
    if not files:
        raise ValueError("No supported audio files found.")
    if not project_name or str(project_name) == "None":
        raise ValueError("Create or select a Project first.")
    project = Path(projects_root) / _safe_name(project_name)
    audio_dir = project / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    manifest = project / "manifest.jsonl"
    rows = []
    for index, src in enumerate(files, 1):
        text_path = src.with_suffix(".txt")
        transcript = text_path.read_text(encoding="utf-8").strip() if text_path.exists() else ""
        if not transcript and transcribe_missing:
            if asr_manager is None:
                raise ValueError(f"Missing transcript for {src.name} and ASR is disabled.")
            transcript, _ = asr_manager.transcribe(
                str(src), asr_model, asr_language, int(asr_batch_size)
            )
        if not transcript:
            log(f"[dataset] Skipping {src.name}: no transcript.")
            continue
        dst = audio_dir / f"{index:05d}_{src.stem}{src.suffix.lower()}"
        if src.resolve() != dst.resolve():
            shutil.copy2(src, dst)
        row = {"audio": str(dst.resolve()), "text": transcript, "language": language}
        rows.append(row)
        log(f"[dataset] Prepared {index}/{len(files)}: {src.name}")
    if not rows:
        raise ValueError("No usable samples were prepared.")
    with manifest.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    meta_path = project / "project.json"
    meta = {}
    if meta_path.is_file():
        try:
            loaded = json.loads(meta_path.read_text(encoding="utf-8-sig"))
            if isinstance(loaded, dict):
                meta.update(loaded)
        except Exception:
            pass
    meta.update({"name": project.name, "samples": len(rows), "language": language, "manifest": str(manifest)})
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return project, manifest, len(rows)


def load_manifest(path: str | Path):
    p = Path(path)
    rows = []
    with p.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def stop_training():
    STOP_EVENT.set()


def _target_modules():
    # Qwen3 projections + FireRed DiT/PatchEncoder attention/MLP projections.
    return [
        "q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj",
        "to_q","to_k","to_v","linear",
    ]


def _pad_latents(latents: torch.Tensor, patch_size: int):
    rem = latents.shape[1] % patch_size
    if rem:
        latents = F.pad(latents, (0, 0, 0, patch_size-rem))
    return latents


def _teacher_forced_flow_loss(tts, prompt_audio, prompt_sr, prompt_text, target_audio, target_sr, target_text, language):
    """Experimental reconstruction of FireRedTTS3 Base's published flow objective.

    The released repository does not include its official training loss. This objective
    follows the public inference equations: teacher-forced Qwen conditioning plus
    rectified-flow velocity prediction in the DiT latent space.
    """
    core = tts.tts_core
    device = tts.device
    patch = core.patch_size

    # Encode audio targets with frozen frontend.
    with torch.no_grad():
        pa = prompt_audio[:1]
        pa = torchaudio.functional.resample(pa, prompt_sr, tts.redae.sample_rate)
        pa = tts.redae.pad_to_multiple_of(pa, tts.redae.downsample_rate * patch).to(device)
        prompt_latents = tts.redae.encode(pa, tts.redae.sample_rate).float()
        spk = tts.spk_extractor.forward(pa, tts.redae.sample_rate).to(device)

        ta = target_audio[:1]
        ta = torchaudio.functional.resample(ta, target_sr, tts.redae.sample_rate)
        ta = tts.redae.pad_to_multiple_of(ta, tts.redae.downsample_rate * patch).to(device)
        target_latents = _pad_latents(tts.redae.encode(ta, tts.redae.sample_rate).float(), patch)

    lang_tag = f"<|{language}|>"
    text_in = f"{lang_tag}<|sot|>{prompt_text}{target_text}<|eot|>"
    text_tokens = tts._tokenize_text(text_in)

    # Teacher-forced patch sequence. Gradients are required through LoRA-equipped
    # backbone/patch encoder/DiT, while RedAE/CampPlus remain frozen.
    prompt_patches = core.patch_encoder(prompt_latents)
    target_patches = target_latents.view(1, -1, patch, target_latents.shape[-1])
    target_patch_embeds = []
    for j in range(target_patches.shape[1]):
        target_patch_embeds.append(core.patch_encoder(target_patches[:, j]))
    target_patch_embeds = torch.cat(target_patch_embeds, dim=1) if target_patch_embeds else None

    text_emb = core.backbone_llm.embed_tokens(text_tokens)
    spk_llm = core.spk_proj_llm(spk).unsqueeze(1)
    teacher_inputs = [spk_llm, text_emb, prompt_patches]
    if target_patch_embeds is not None and target_patch_embeds.shape[1] > 1:
        teacher_inputs.append(target_patch_embeds[:, :-1])
    input_embeds = torch.cat(teacher_inputs, dim=1)

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        backbone = core.backbone_llm.forward(
            inputs_embeds=input_embeds,
            use_cache=False,
        ).last_hidden_state
    patch_seq_start = spk_llm.shape[1] + text_emb.shape[1]
    prompt_patch_count = prompt_patches.shape[1]
    dit_spk = core.spk_proj_dit(spk)

    # Stochastic patch training keeps the experimental trainer within consumer
    # GPU memory. Oversample the final patch so the stop head sees enough
    # positive targets; uniform sampling made stop_loss almost always zero.
    patch_count = int(target_patches.shape[1])
    if patch_count <= 1:
        j = 0
    elif random.random() < 0.25:
        j = patch_count - 1
    else:
        j = random.randrange(patch_count - 1)
    current = target_patches[:, j]
    full_clean = torch.cat([prompt_latents, target_latents], dim=1)
    prompt_latent_count = prompt_latents.shape[1]
    hist_end = prompt_latent_count + j * patch
    hist = F.pad(full_clean[:, :hist_end], (0, 0, core.history_length, 0))[:, -core.history_length:]

    cond_end = patch_seq_start + prompt_patch_count + j
    cond_start = max(patch_seq_start, cond_end - (core.history_patches + 1))
    cond_h = backbone[:, cond_start:cond_end]
    if cond_h.shape[1] < core.history_patches + 1:
        cond_h = F.pad(cond_h, (0,0,core.history_patches+1-cond_h.shape[1],0))
    cond_h = core.dit_head(cond_h[:, -(core.history_patches+1):])

    noise = torch.randn_like(current)
    t = 0.05 + 0.90 * torch.rand((1,), device=device, dtype=current.dtype)
    xt_current = noise * (1.0 - t.view(1,1,1)) + current * t.view(1,1,1)
    xt = torch.cat([hist, xt_current], dim=1)
    cond = torch.cat([
        cond_h.repeat_interleave(patch, dim=1),
        dit_spk.unsqueeze(1).repeat(1, core.history_length + patch, 1),
    ], dim=-1)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        pred = core.dit(torch.cat([xt, cond], dim=-1), t)
    target_v = current - noise
    flow_loss = F.mse_loss(pred[:, -patch:].float(), target_v.float())

    h_for_stop = backbone[:, cond_end-1]
    stop_target = torch.ones(1, device=device) if j == target_patches.shape[1]-1 else torch.zeros(1, device=device)
    stop_loss = F.binary_cross_entropy_with_logits(core.stop_head(h_for_stop).view(-1).float(), stop_target.float())
    return flow_loss + 0.05 * stop_loss, flow_loss.detach(), stop_loss.detach()


def train_lora(
    manager,
    manifest_path: str,
    output_dir: str,
    rank: int = 16,
    alpha: int = 32,
    dropout: float = 0.05,
    learning_rate: float = 1e-4,
    length_mode: str = "Steps",
    training_steps: int = 1000,
    training_epochs: int = 10,
    gradient_accumulation: int = 1,
    seed: int = 1234,
    log=print,
    progress_cb=None,
    save_every: int = 100,
    resume_checkpoint: str | None = None,
    enable_eval_audio: bool = False,
    eval_text: str = "This is a FireRedTTS3 training preview.",
    eval_reference_audio: str | None = None,
    eval_reference_transcript: str = "",
    eval_language: str = "English",
):
    from peft import LoraConfig, get_peft_model, PeftModel
    from torch.utils.tensorboard import SummaryWriter

    rows = load_manifest(manifest_path)
    if len(rows) < 2:
        raise ValueError("Experimental LoRA training requires at least 2 prepared clips.")

    # Internal dataset-driven clipping policy.
    durations = []
    for row in rows:
        try:
            info = torchaudio.info(row["audio"])
            if info.sample_rate > 0 and info.num_frames > 0:
                durations.append(float(info.num_frames) / float(info.sample_rate))
        except Exception:
            pass
    if durations:
        values = sorted(durations)
        median = values[len(values) // 2]
        p90 = values[min(len(values) - 1, int(round((len(values) - 1) * 0.90)))]
        auto_prompt_seconds = max(1.5, min(6.0, median))
        auto_target_seconds = max(3.0, min(15.0, p90))
    else:
        auto_prompt_seconds = 3.0
        auto_target_seconds = 8.0
    log(
        f"[training] Automatic audio limits: reference ≤{auto_prompt_seconds:.1f}s · "
        f"target ≤{auto_target_seconds:.1f}s"
    )

    STOP_EVENT.clear()
    random.seed(seed)
    torch.manual_seed(seed)

    mode = "Epochs" if str(length_mode) == "Epochs" else "Steps"
    training_steps = max(1, int(training_steps))
    training_epochs = max(1, int(training_epochs))
    save_every = max(1, int(save_every))
    accum = max(1, int(gradient_accumulation))
    steps_per_epoch = max(1, math.ceil(len(rows) / accum))
    total_planned = training_steps if mode == "Steps" else training_epochs * steps_per_epoch

    if mode == "Steps":
        log(f"[training] Steps mode: {training_steps} steps · save/eval every {save_every} steps")
    else:
        log(f"[training] Epochs mode: {training_epochs} epochs · save/eval every {save_every} epochs")

    tts = manager.load_base(log=log, attention_backend="flash_attention_2", torch_compile=False)
    core = tts.tts_core

    tts.redae.eval()
    tts.spk_extractor.eval()
    for p in tts.redae.parameters():
        p.requires_grad_(False)
    for p in tts.spk_extractor.parameters():
        p.requires_grad_(False)

    resume_path = Path(resume_checkpoint) if resume_checkpoint and str(resume_checkpoint) != "None" else None
    if resume_path is not None and (resume_path / "adapter_config.json").is_file():
        log(f"[training] Resuming adapter from {resume_path}")
        tts.tts_core = PeftModel.from_pretrained(core, str(resume_path), is_trainable=True)
        core = tts.tts_core
    elif not hasattr(core, "peft_config"):
        cfg = LoraConfig(
            r=int(rank),
            lora_alpha=int(alpha),
            lora_dropout=float(dropout),
            target_modules="all-linear",
            bias="none",
        )
        tts.tts_core = get_peft_model(core, cfg)
        core = tts.tts_core

    core.train()
    params = [p for p in core.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError("No trainable LoRA parameters were created.")
    opt = torch.optim.AdamW(
        params,
        lr=float(learning_rate),
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.01,
    )

    # Warmup + cosine decay prevents the early all-linear LoRA updates from
    # hitting the model at full LR before the optimizer statistics stabilize.
    warmup_steps = max(20, min(100, int(round(total_planned * 0.05))))
    min_lr_ratio = 0.10

    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return max(0.05, float(current_step + 1) / float(warmup_steps))
        progress = min(
            1.0,
            max(0.0, (current_step - warmup_steps) / max(1, total_planned - warmup_steps)),
        )
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)

    micro_step = 0
    step = 0
    completed_epochs = 0
    if resume_path is not None and (resume_path / "trainer_state.pt").is_file():
        state = torch.load(resume_path / "trainer_state.pt", map_location="cpu", weights_only=False)
        if isinstance(state, dict) and state.get("optimizer"):
            opt.load_state_dict(state["optimizer"])
            if state.get("scheduler"):
                try:
                    scheduler.load_state_dict(state["scheduler"])
                except Exception as exc:
                    log(f"[training][WARN] Scheduler state restore skipped: {exc}")
            step = int(state.get("step", state.get("optimizer_step", 0)))
            micro_step = int(state.get("micro_step", 0))
            completed_epochs = int(state.get("epoch", 0))
            log(
                f"[training] Resumed at "
                + (f"step {step}" if mode == "Steps" else f"epoch {completed_epochs}")
            )

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(out / "tensorboard"))

    def save_checkpoint(epoch_value: int):
        name = (
            f"checkpoint-step-{step:06d}"
            if mode == "Steps"
            else f"checkpoint-epoch-{epoch_value:04d}"
        )
        ckpt = out / name
        ckpt.mkdir(parents=True, exist_ok=True)
        log(
            f"[training] Saving checkpoint at "
            + (f"step {step}" if mode == "Steps" else f"epoch {epoch_value}")
            + "..."
        )
        core.save_pretrained(str(ckpt))
        torch.save(
            {
                "step": step,
                "optimizer_step": step,
                "micro_step": micro_step,
                "epoch": epoch_value,
                "length_mode": mode,
                "optimizer": opt.state_dict(),
                "scheduler": scheduler.state_dict(),
            },
            ckpt / "trainer_state.pt",
        )
        log(f"[training] Checkpoint saved: {ckpt}")

        if enable_eval_audio and eval_reference_audio and Path(eval_reference_audio).is_file():
            try:
                core.eval()
                ref, ref_sr = torchaudio.load(eval_reference_audio)
                with torch.inference_mode():
                    wav, wav_sr = tts.generate(
                        language=eval_language,
                        prompt_text=eval_reference_transcript,
                        prompt_audio=ref,
                        prompt_audio_sr=ref_sr,
                        text=eval_text,
                        seed=seed,
                        do_split=False,
                    )
                eval_dir = out / "eval_audio"
                eval_dir.mkdir(parents=True, exist_ok=True)
                eval_path = eval_dir / (
                    f"step-{step:06d}.wav"
                    if mode == "Steps"
                    else f"epoch-{epoch_value:04d}.wav"
                )
                torchaudio.save(str(eval_path), wav.detach().cpu(), wav_sr)
                writer.add_audio("eval/generated", wav.detach().cpu(), step, sample_rate=wav_sr)
                writer.flush()
                log(f"[training] Eval audio saved: {eval_path}")
            except Exception as exc:
                log(f"[training][WARN] Eval audio failed: {exc}")
            finally:
                core.train()

    opt.zero_grad(set_to_none=True)

    # Step metrics are averaged over every micro-batch participating in the
    # optimizer update. EMA metrics are the primary TensorBoard curves.
    accum_loss = 0.0
    accum_flow = 0.0
    accum_stop = 0.0
    accum_count = 0
    ema_loss = None
    ema_flow = None
    ema_stop = None
    ema_beta = 0.90

    while not STOP_EVENT.is_set():
        if mode == "Steps" and step >= training_steps:
            break
        if mode == "Epochs" and completed_epochs >= training_epochs:
            break

        order = list(range(len(rows)))
        random.shuffle(order)
        epoch_number = completed_epochs + 1

        for batch_index, ri in enumerate(order):
            if STOP_EVENT.is_set():
                break
            if mode == "Steps" and step >= training_steps:
                break

            target = rows[ri]
            prompt = rows[random.choice([i for i in range(len(rows)) if i != ri])]
            target_audio, target_sr = torchaudio.load(target["audio"])
            prompt_audio, prompt_sr = torchaudio.load(prompt["audio"])

            target_audio = target_audio[:, :max(1, int(target_sr * auto_target_seconds))]
            prompt_audio = prompt_audio[:, :max(1, int(prompt_sr * auto_prompt_seconds))]

            loss, flow, stop = _teacher_forced_flow_loss(
                tts,
                prompt_audio,
                prompt_sr,
                prompt["text"],
                target_audio,
                target_sr,
                target["text"],
                target.get("language") or "English",
            )
            (loss / accum).backward()
            micro_step += 1
            accum_loss += float(loss.detach().item())
            accum_flow += float(flow.detach().item())
            accum_stop += float(stop.detach().item())
            accum_count += 1

            is_last_batch = (batch_index == len(order) - 1)
            do_step = ((batch_index + 1) % accum == 0) or is_last_batch
            if not do_step:
                continue

            grad_norm = torch.nn.utils.clip_grad_norm_(params, 0.5)
            opt.step()
            scheduler.step()
            opt.zero_grad(set_to_none=True)
            step += 1

            step_loss = accum_loss / max(1, accum_count)
            step_flow = accum_flow / max(1, accum_count)
            step_stop = accum_stop / max(1, accum_count)
            ema_loss = step_loss if ema_loss is None else ema_beta * ema_loss + (1.0 - ema_beta) * step_loss
            ema_flow = step_flow if ema_flow is None else ema_beta * ema_flow + (1.0 - ema_beta) * step_flow
            ema_stop = step_stop if ema_stop is None else ema_beta * ema_stop + (1.0 - ema_beta) * step_stop
            current_lr = float(opt.param_groups[0]["lr"])

            accum_loss = accum_flow = accum_stop = 0.0
            accum_count = 0

            if mode == "Steps":
                msg = (
                    f"[training] step={step}/{training_steps} "
                    f"loss={step_loss:.6f} ema={ema_loss:.6f} "
                    f"flow={step_flow:.6f} stop={step_stop:.6f} "
                    f"lr={current_lr:.2e}"
                )
                progress_value = step
            else:
                msg = (
                    f"[training] epoch={epoch_number}/{training_epochs} "
                    f"loss={step_loss:.6f} ema={ema_loss:.6f} "
                    f"flow={step_flow:.6f} stop={step_stop:.6f} "
                    f"lr={current_lr:.2e}"
                )
                progress_value = min(
                    total_planned,
                    completed_epochs * steps_per_epoch + min(
                        steps_per_epoch, (batch_index // accum) + 1
                    ),
                )

            log(msg)

            # Primary charts are smoothed. Raw step averages remain available
            # for diagnosing variance without making the main graphs unreadable.
            writer.add_scalar("train/loss", float(ema_loss), step)
            writer.add_scalar("train/flow_loss", float(ema_flow), step)
            writer.add_scalar("train/stop_loss", float(ema_stop), step)
            writer.add_scalar("train_raw/loss", float(step_loss), step)
            writer.add_scalar("train_raw/flow_loss", float(step_flow), step)
            writer.add_scalar("train_raw/stop_loss", float(step_stop), step)
            writer.add_scalar("train/learning_rate", current_lr, step)
            writer.add_scalar("train/grad_norm", float(grad_norm), step)
            writer.flush()
            if progress_cb:
                progress_cb(progress_value, max(1, total_planned), msg)

            if mode == "Steps" and step % save_every == 0:
                save_checkpoint(epoch_number)

        completed_epochs += 1
        if mode == "Epochs" and completed_epochs % save_every == 0:
            save_checkpoint(completed_epochs)

    core.save_pretrained(str(out))
    torch.save(
        {
            "step": step,
            "optimizer_step": step,
            "micro_step": micro_step,
            "epoch": completed_epochs,
            "length_mode": mode,
            "optimizer": opt.state_dict(),
            "scheduler": scheduler.state_dict(),
        },
        out / "trainer_state.pt",
    )
    meta = {
        "experimental": True,
        "objective": "teacher-forced rectified-flow reconstruction from released inference equations",
        "rank": int(rank),
        "alpha": int(alpha),
        "dropout": float(dropout),
        "length_mode": mode,
        "steps": step,
        "epochs": completed_epochs,
        "training_steps": training_steps,
        "training_epochs": training_epochs,
        "save_every": save_every,
        "optimizer": "AdamW(beta1=0.9,beta2=0.95,weight_decay=0.01)",
        "warmup_steps": warmup_steps,
        "scheduler": "cosine_to_10pct",
        "manifest": str(Path(manifest_path).resolve()),
    }
    (out / "firered_lora_metadata.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    core.eval()
    writer.close()
    manager.unload(log)
    return out, step if mode == "Steps" else completed_epochs
