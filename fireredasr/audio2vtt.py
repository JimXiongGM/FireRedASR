#!/usr/bin/env python3

import argparse
import os
from typing import List, Tuple

import kaldiio
import numpy as np

from fireredasr.speech2text import FireRedAsr, get_wav_info


def format_vtt_timestamp(seconds: float) -> str:
    total_ms = int(round(seconds * 1000.0))
    ms = total_ms % 1000
    total_sec = total_ms // 1000
    s = total_sec % 60
    total_min = total_sec // 60
    m = total_min % 60
    h = total_min // 60
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def fixed_segment(sample_rate: int, wav: np.ndarray, segment_ms: int) -> List[Tuple[float, float, Tuple[int, np.ndarray]]]:
    segment_len = int(sample_rate * (segment_ms / 1000.0))
    segments = []
    num_samples = wav.shape[0]
    start = 0
    while start < num_samples:
        end = min(start + segment_len, num_samples)
        seg_wav = wav[start:end]
        start_sec = start / sample_rate
        end_sec = end / sample_rate
        segments.append((start_sec, end_sec, (sample_rate, seg_wav)))
        start = end
    return segments


def energy_vad_segments(
    sample_rate: int,
    wav: np.ndarray,
    frame_ms: int = 25,
    hop_ms: int = 10,
    energy_threshold_db: float = -40.0,
    min_speech_ms: int = 300,
    max_speech_ms: int = 30000,
    max_silence_ms: int = 200,
    pad_ms: int = 100,
) -> List[Tuple[float, float, Tuple[int, np.ndarray]]]:
    # Normalize to float32 [-1, 1]
    x = wav
    if x.dtype != np.float32:
        x = x.astype(np.float32)
    max_abs = float(np.max(np.abs(x))) if x.size > 0 else 1.0
    if wav.dtype == np.int16:
        x = x / 32768.0
    elif max_abs > 1.0:
        x = x / max_abs

    eps = 1e-10
    frame_len = max(1, int(sample_rate * (frame_ms / 1000.0)))
    hop = max(1, int(sample_rate * (hop_ms / 1000.0)))
    num_samples = x.shape[0]
    if num_samples == 0:
        return []

    # Compute RMS energy per frame in dBFS
    rms_list: List[float] = []
    for start in range(0, max(1, num_samples - frame_len + 1), hop):
        frame = x[start : start + frame_len]
        if frame.shape[0] < frame_len:
            pad = np.zeros((frame_len - frame.shape[0],), dtype=frame.dtype)
            frame = np.concatenate([frame, pad], axis=0)
        rms = float(np.sqrt(np.mean(frame * frame) + eps))
        db = 20.0 * np.log10(rms + eps)
        rms_list.append(db)
    energies_db = np.array(rms_list, dtype=np.float32)

    # Voice mask
    is_speech = energies_db > energy_threshold_db
    if is_speech.size == 0:
        return []

    # Build initial runs of speech
    boundaries = np.flatnonzero(np.diff(np.concatenate(([0], is_speech.view(np.int8), [0]))))
    run_starts = boundaries[0::2]
    run_ends = boundaries[1::2]

    # Merge runs across short gaps
    merged: List[Tuple[int, int]] = []
    max_gap = int(round(max_silence_ms / hop_ms))
    for s, e in zip(run_starts, run_ends):
        if not merged:
            merged.append((s, e))
            continue
        prev_s, prev_e = merged[-1]
        gap = s - prev_e
        if gap <= max_gap:
            merged[-1] = (prev_s, e)
        else:
            merged.append((s, e))

    # Filter short runs and enforce max length by splitting
    min_len_f = int(round(min_speech_ms / hop_ms))
    max_len_f = int(round(max_speech_ms / hop_ms))
    final_runs: List[Tuple[int, int]] = []
    for s, e in merged:
        length = e - s
        if length < min_len_f:
            continue
        if length <= max_len_f:
            final_runs.append((s, e))
        else:
            # Split into chunks of size max_len_f
            cur = s
            while cur < e:
                nxt = min(cur + max_len_f, e)
                final_runs.append((cur, nxt))
                cur = nxt

    # Convert frames to sample indices with padding
    pad_samp = int(round(sample_rate * (pad_ms / 1000.0)))
    segments: List[Tuple[float, float, Tuple[int, np.ndarray]]] = []
    for s_f, e_f in final_runs:
        start_samp = s_f * hop
        end_samp = min(num_samples, e_f * hop + frame_len)
        start_samp = max(0, start_samp - pad_samp)
        end_samp = min(num_samples, end_samp + pad_samp)
        seg_wav = wav[start_samp:end_samp]
        segments.append((start_samp / sample_rate, end_samp / sample_rate, (sample_rate, seg_wav)))
    return segments


def write_vtt(vtt_path: str, segments: List[Tuple[float, float, str]]) -> None:
    lines = ["WEBVTT", ""]
    for i, (st, et, text) in enumerate(segments, start=1):
        lines.append(str(i))
        lines.append(f"{format_vtt_timestamp(st)} --> {format_vtt_timestamp(et)}")
        lines.append(text if text else "")
        lines.append("")
    os.makedirs(os.path.dirname(vtt_path), exist_ok=True)
    with open(vtt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def restore_punctuation_cn(text: str, max_clause_len: int = 18) -> str:
    if not text:
        return text
    t = text.strip()
    # Normalize ASCII punctuation to Chinese
    trans = str.maketrans({
        ",": "，",
        ";": "；",
        ":": "：",
        "?": "？",
        "!": "！",
        "(" : "（",
        ")" : "）",
    })
    t = t.translate(trans)

    # Conservative: only ensure sentence ends with proper ending punctuation.
    end_punct = set("。！？……")
    if not any(ch in end_punct for ch in t):
        t = t + "。"
    return t

def refine_segments_energy_minima(
    sample_rate: int,
    wav: np.ndarray,
    segments: List[Tuple[float, float, Tuple[int, np.ndarray]]],
    refine_window_ms: int,
    refine_eval_win_ms: int,
    refine_step_ms: int,
) -> List[Tuple[float, float, Tuple[int, np.ndarray]]]:
    win = max(1, int(sample_rate * (refine_window_ms / 1000.0)))
    eval_half = max(1, int(sample_rate * (refine_eval_win_ms / 1000.0) / 2.0))
    step = max(1, int(sample_rate * (refine_step_ms / 1000.0)))

    num_samples = wav.shape[0]

    def local_energy(center: int) -> float:
        a = max(0, center - eval_half)
        b = min(num_samples, center + eval_half)
        seg = wav[a:b]
        if seg.size == 0:
            return 1e9
        # robust RMS: guard against NaNs/Infs
        seg = np.nan_to_num(seg, nan=0.0, posinf=0.0, neginf=0.0)
        mean_sq = float(np.mean(seg * seg)) if seg.size > 0 else 0.0
        mean_sq = max(mean_sq, 1e-12)
        return float(np.sqrt(mean_sq))

    refined: List[Tuple[float, float, Tuple[int, np.ndarray]]] = []
    for (st_s, et_s, (_sr, _seg_wav)) in segments:
        st = int(round(st_s * sample_rate))
        et = int(round(et_s * sample_rate))
        if st >= et:
            refined.append((st_s, et_s, (_sr, _seg_wav)))
            continue

        st_lo = max(0, st - win)
        st_hi = min(num_samples - 1, st + win)
        best_st = st
        best_e = local_energy(st)
        for c in range(st_lo, st_hi + 1, step):
            e = local_energy(c)
            if e < best_e:
                best_e = e
                best_st = c

        et_lo = max(0, et - win)
        et_hi = min(num_samples - 1, et + win)
        best_et = et
        best_e2 = local_energy(et)
        for c in range(et_lo, et_hi + 1, step):
            e = local_energy(c)
            if e < best_e2:
                best_e2 = e
                best_et = c

        # Ensure valid ordering
        if best_st >= best_et - max(1, step):
            best_st, best_et = st, et

        new_seg = wav[best_st:best_et]
        refined.append((best_st / sample_rate, best_et / sample_rate, (sample_rate, new_seg)))

    return refined


def main(args: argparse.Namespace) -> None:
    wavs = get_wav_info(args)
    model = FireRedAsr.from_pretrained(args.asr_type, args.model_dir)

    for uttid, wav_path in wavs:
        sr, wav_np = kaldiio.load_mat(wav_path)
        if args.seg_method == "fixed":
            segs = fixed_segment(sr, wav_np, args.segment_ms)
        else:
            segs = energy_vad_segments(
                sr,
                wav_np,
                frame_ms=args.vad_frame_ms,
                hop_ms=args.vad_hop_ms,
                energy_threshold_db=args.vad_energy_threshold_db,
                min_speech_ms=args.vad_min_speech_ms,
                max_speech_ms=args.vad_max_speech_ms,
                max_silence_ms=args.vad_max_silence_ms,
                pad_ms=args.vad_pad_ms,
            )

        if args.refine_boundaries:
            segs = refine_segments_energy_minima(
                sr,
                wav_np,
                segs,
                refine_window_ms=args.refine_window_ms,
                refine_eval_win_ms=args.refine_eval_win_ms,
                refine_step_ms=args.refine_step_ms,
            )

        # Batch inference per file
        all_texts: List[str] = []
        batch: List[Tuple[int, np.ndarray]] = []
        batch_starts: List[float] = []
        batch_ends: List[float] = []
        for (st, et, seg) in segs:
            batch.append(seg)
            batch_starts.append(st)
            batch_ends.append(et)
            if len(batch) == args.batch_size:
                results = model.transcribe(
                    [f"{uttid}_{len(all_texts)+i:04d}" for i in range(len(batch))],
                    batch,
                    {
                        "use_gpu": args.use_gpu,
                        "beam_size": args.beam_size,
                        "nbest": args.nbest,
                        "decode_max_len": args.decode_max_len,
                        "softmax_smoothing": args.softmax_smoothing,
                        "aed_length_penalty": args.aed_length_penalty,
                        "eos_penalty": args.eos_penalty,
                        "decode_min_len": args.decode_min_len,
                        "repetition_penalty": args.repetition_penalty,
                        "llm_length_penalty": args.llm_length_penalty,
                        "temperature": args.temperature,
                    },
                )
                all_texts.extend([r["text"] for r in results])
                batch = []
                batch_starts = []
                batch_ends = []

        if batch:
            results = model.transcribe(
                [f"{uttid}_{len(all_texts)+i:04d}" for i in range(len(batch))],
                batch,
                {
                    "use_gpu": args.use_gpu,
                    "beam_size": args.beam_size,
                    "nbest": args.nbest,
                    "decode_max_len": args.decode_max_len,
                    "softmax_smoothing": args.softmax_smoothing,
                    "aed_length_penalty": args.aed_length_penalty,
                    "eos_penalty": args.eos_penalty,
                    "decode_min_len": args.decode_min_len,
                    "repetition_penalty": args.repetition_penalty,
                    "llm_length_penalty": args.llm_length_penalty,
                    "temperature": args.temperature,
                },
            )
            all_texts.extend([r["text"] for r in results])

        # Assemble VTT segments
        vtt_segments: List[Tuple[float, float, str]] = []
        for (st, et, _), text in zip(segs, all_texts):
            if args.restore_punctuation:
                text = restore_punctuation_cn(text, max_clause_len=args.punct_max_clause_len)
            vtt_segments.append((st, et, text))

        out_dir = args.output_dir
        os.makedirs(out_dir, exist_ok=True)
        vtt_path = os.path.join(out_dir, f"{uttid}.vtt")
        write_vtt(vtt_path, vtt_segments)
        print(f"Saved: {vtt_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--asr_type', type=str, required=True, choices=["aed", "llm"])
    parser.add_argument('--model_dir', type=str, required=True)

    # Input / Output
    parser.add_argument("--wav_path", type=str)
    parser.add_argument("--wav_paths", type=str, nargs="*")
    parser.add_argument("--wav_dir", type=str)
    parser.add_argument("--wav_scp", type=str)
    parser.add_argument("--output_dir", type=str, default="out/vtt")

    # Segmentation
    parser.add_argument("--seg_method", type=str, default="energy_vad", choices=["fixed", "energy_vad"])
    parser.add_argument("--segment_ms", type=int, default=15000, help="Used when seg_method=fixed")
    parser.add_argument("--vad_frame_ms", type=int, default=25)
    parser.add_argument("--vad_hop_ms", type=int, default=10)
    parser.add_argument("--vad_energy_threshold_db", type=float, default=-40.0)
    parser.add_argument("--vad_min_speech_ms", type=int, default=300)
    parser.add_argument("--vad_max_speech_ms", type=int, default=30000)
    parser.add_argument("--vad_max_silence_ms", type=int, default=200)
    parser.add_argument("--vad_pad_ms", type=int, default=100)

    # Boundary refinement
    parser.add_argument("--refine_boundaries", type=int, default=1)
    parser.add_argument("--refine_window_ms", type=int, default=120)
    parser.add_argument("--refine_eval_win_ms", type=int, default=20)
    parser.add_argument("--refine_step_ms", type=int, default=5)

    # Punctuation restoration (heuristic)
    parser.add_argument("--restore_punctuation", type=int, default=1)
    parser.add_argument("--punct_max_clause_len", type=int, default=18)

    # Decode Options (reusing existing)
    parser.add_argument('--use_gpu', type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--beam_size", type=int, default=1)
    parser.add_argument("--decode_max_len", type=int, default=0)
    # FireRedASR-AED
    parser.add_argument("--nbest", type=int, default=1)
    parser.add_argument("--softmax_smoothing", type=float, default=1.0)
    parser.add_argument("--aed_length_penalty", type=float, default=0.0)
    parser.add_argument("--eos_penalty", type=float, default=1.0)
    # FireRedASR-LLM
    parser.add_argument("--decode_min_len", type=int, default=0)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--llm_length_penalty", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=1.0)

    args = parser.parse_args()
    main(args)


