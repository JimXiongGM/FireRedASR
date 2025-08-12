#!/usr/bin/env python3

import os
import uuid
import shutil
import subprocess
from typing import Optional, Tuple, List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import kaldiio
import numpy as np
from datetime import datetime

from fireredasr.speech2text import FireRedAsr
from fireredasr.audio2vtt import (
    energy_vad_segments,
    refine_segments_energy_minima,
    write_vtt,
    restore_punctuation_cn,
)


app = FastAPI(title="FireRedASR VTT Service")


class TranscribeRequest(BaseModel):
    file_path: str
    asr_type: str = "llm"
    model_dir: Optional[str] = "pretrained_models/FireRedASR-LLM-L"
    output_dir: Optional[str] = "out/vtt_api"
    # Segmentation
    seg_method: str = "energy_vad"  # fixed/energy_vad
    segment_ms: int = 15000  # used when seg_method=fixed
    vad_frame_ms: int = 25
    vad_hop_ms: int = 10
    vad_energy_threshold_db: float = -45.0
    vad_min_speech_ms: int = 200
    vad_max_speech_ms: int = 30000
    vad_max_silence_ms: int = 250
    vad_pad_ms: int = 120
    # Boundary refinement
    refine_boundaries: int = 1
    refine_window_ms: int = 120
    refine_eval_win_ms: int = 20
    refine_step_ms: int = 5
    # Decode Options
    use_gpu: int = 1
    batch_size: int = 2
    beam_size: int = 3
    decode_max_len: int = 0
    decode_min_len: int = 0
    repetition_penalty: float = 3.0
    llm_length_penalty: float = 1.0
    temperature: float = 1.0
    # Punctuation
    restore_punctuation: int = 1
    punct_max_clause_len: int = 18


_GLOBAL_MODEL: Optional[FireRedAsr] = None
_GLOBAL_MODEL_KEY: Optional[Tuple[str, str]] = None  # (asr_type, model_dir)


def _ensure_model(asr_type: str, model_dir: str) -> FireRedAsr:
    global _GLOBAL_MODEL, _GLOBAL_MODEL_KEY
    key = (asr_type, os.path.abspath(model_dir))
    if _GLOBAL_MODEL is None or _GLOBAL_MODEL_KEY != key:
        if not os.path.isdir(model_dir):
            raise HTTPException(status_code=400, detail=f"model_dir not found: {model_dir}")
        _GLOBAL_MODEL = FireRedAsr.from_pretrained(asr_type, model_dir)
        _GLOBAL_MODEL_KEY = key
    return _GLOBAL_MODEL


def _is_video(path: str) -> bool:
    ext = os.path.splitext(path)[1].lower()
    return ext in {".mp4", ".mkv", ".mov", ".avi", ".flv", ".webm", ".m4v"}


def _extract_wav(input_path: str) -> str:
    abs_in = os.path.abspath(input_path)
    if not os.path.exists(abs_in):
        raise HTTPException(status_code=400, detail=f"file_path not found: {input_path}")
    os.makedirs("out/tmp", exist_ok=True)
    tmp_wav = os.path.abspath(os.path.join("out/tmp", f"{uuid.uuid4().hex}.wav"))
    cmd = [
        "ffmpeg", "-y", "-i", abs_in,
        "-ar", "16000", "-ac", "1", "-acodec", "pcm_s16le", "-f", "wav", tmp_wav,
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="ffmpeg not found in PATH")
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"ffmpeg failed: {e.stderr.decode(errors='ignore')[:512]}")
    return tmp_wav


def _maybe_to_wav(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".wav":
        return os.path.abspath(path)
    # convert any non-wav (audio/video) to wav via ffmpeg
    return _extract_wav(path)


def _build_vtt_for_file(req: TranscribeRequest) -> str:
    model = _ensure_model(req.asr_type, req.model_dir)

    inp_abs = os.path.abspath(req.file_path)
    wav_path = _maybe_to_wav(inp_abs) if _is_video(inp_abs) or os.path.splitext(inp_abs)[1].lower() != ".wav" else inp_abs

    # Load waveform
    sample_rate, wav_np = kaldiio.load_mat(wav_path)
    if wav_np.dtype != np.float32:
        wav_np = wav_np.astype(np.float32)
    # Normalize to [-1,1]
    if wav_np.dtype == np.int16:
        wav_np = wav_np / 32768.0
    else:
        max_abs = float(np.max(np.abs(wav_np))) if wav_np.size > 0 else 1.0
        if max_abs > 1.0:
            wav_np = wav_np / max_abs

    # Segmentation
    if req.seg_method == "energy_vad":
        segs = energy_vad_segments(
            sample_rate,
            wav_np,
            frame_ms=req.vad_frame_ms,
            hop_ms=req.vad_hop_ms,
            energy_threshold_db=req.vad_energy_threshold_db,
            min_speech_ms=req.vad_min_speech_ms,
            max_speech_ms=req.vad_max_speech_ms,
            max_silence_ms=req.vad_max_silence_ms,
            pad_ms=req.vad_pad_ms,
        )
    else:
        # fixed segmentation from audio2vtt
        from fireredasr.audio2vtt import fixed_segment
        segs = fixed_segment(sample_rate, wav_np, req.segment_ms)

    if req.refine_boundaries:
        segs = refine_segments_energy_minima(
            sample_rate,
            wav_np,
            segs,
            refine_window_ms=req.refine_window_ms,
            refine_eval_win_ms=req.refine_eval_win_ms,
            refine_step_ms=req.refine_step_ms,
        )

    # Batch inference
    all_texts: List[str] = []
    batch: List[Tuple[int, np.ndarray]] = []
    for (st, et, seg) in segs:
        batch.append(seg)
        if len(batch) == req.batch_size:
            results = model.transcribe(
                ["seg_%06d" % (len(all_texts) + i) for i in range(len(batch))],
                batch,
                {
                    "use_gpu": req.use_gpu,
                    "beam_size": req.beam_size,
                    "nbest": 1,
                    "decode_max_len": req.decode_max_len,
                    "softmax_smoothing": 1.0,
                    "aed_length_penalty": 0.0,
                    "eos_penalty": 1.0,
                    "decode_min_len": req.decode_min_len,
                    "repetition_penalty": req.repetition_penalty,
                    "llm_length_penalty": req.llm_length_penalty,
                    "temperature": req.temperature,
                },
            )
            all_texts.extend([r["text"] for r in results])
            batch = []

    if batch:
        results = model.transcribe(
            ["seg_%06d" % (len(all_texts) + i) for i in range(len(batch))],
            batch,
            {
                "use_gpu": req.use_gpu,
                "beam_size": req.beam_size,
                "nbest": 1,
                "decode_max_len": req.decode_max_len,
                "softmax_smoothing": 1.0,
                "aed_length_penalty": 0.0,
                "eos_penalty": 1.0,
                "decode_min_len": req.decode_min_len,
                "repetition_penalty": req.repetition_penalty,
                "llm_length_penalty": req.llm_length_penalty,
                "temperature": req.temperature,
            },
        )
        all_texts.extend([r["text"] for r in results])

    # Build VTT segments
    vtt_segments: List[Tuple[float, float, str]] = []
    for (st, et, _), text in zip(segs, all_texts):
        if req.restore_punctuation:
            text = restore_punctuation_cn(text, max_clause_len=req.punct_max_clause_len)
        vtt_segments.append((st, et, text))

    # Output path
    os.makedirs(req.output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(inp_abs))[0]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    vtt_path = os.path.abspath(os.path.join(req.output_dir, f"{base}_{ts}.vtt"))
    write_vtt(vtt_path, vtt_segments)

    # Cleanup temp wav if created
    if wav_path != inp_abs and os.path.basename(wav_path).startswith("out"):
        try:
            os.remove(wav_path)
        except Exception:
            pass

    return vtt_path


@app.post("/transcribe")
def transcribe(req: TranscribeRequest):
    vtt_path = _build_vtt_for_file(req)
    return {"vtt_path": vtt_path}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8100)


