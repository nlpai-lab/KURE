"""Inference throughput (pairs per second, PPS) for `evaluate.py --speed`.

Same protocol as instructkr/reranker-simple-benchmark (`evaluate_reranker.py --speed`), which the
KURE-Reranker model cards report. PPS is timed on the very pairs the NDCG pass scored (golds + BM25
top-50 per query), tapped from the model's predict while mteb runs; whole queries are sampled with
seed 0 until `n` pairs, so every model is timed on the same set. Batch size starts at 8 and doubles
until the first OOM; at each size the sample is repeated to a multiple of it and ordered
longest-first, one warmup, then PASSES timed passes. Only the model's forward is timed (CUDA events
around every call), so tokenization and host-side work are excluded. PPS = pairs / summed forward
time, and the best batch size is reported. The batch search follows the ettin-reranker speed
benchmark (https://huggingface.co/blog/ettin-reranker#speed).
"""

from __future__ import annotations

import itertools
import os
import random
import subprocess

import torch

START_BS = 8
PASSES = 3


def set_attention(hf) -> str | None:
    """Switch to flash_attention_2, or sdpa where it is unavailable; return what is live."""
    for impl in ("flash_attention_2", "sdpa"):
        try:
            hf.set_attn_implementation(impl)
            break
        except Exception as e:
            print(f"[speed] attn {impl} unavailable: {e}", flush=True)
    return getattr(hf.config, "_attn_implementation", None)


class PairTap:
    """Records the (query, doc) pairs the CrossEncoder is asked to score while `on`."""

    def __init__(self, model):
        predict = model.predict
        self.pairs, self.on = [], False

        def tapped(pairs, *args, **kwargs):
            if self.on:
                self.pairs.extend((p[0], p[1]) for p in pairs)
            return predict(pairs, *args, **kwargs)

        model.predict = tapped

    def take(self) -> list:
        pairs, self.pairs = self.pairs, []
        return pairs


def sample_pairs(pairs: list, n: int, seed: int = 0) -> list:
    """Whole queries (all their candidates), shuffled with `seed`, until >= n pairs."""
    by_query: dict[str, list] = {}
    for q, d in pairs:
        by_query.setdefault(q, []).append((q, d))
    queries = sorted(by_query)
    random.Random(seed).shuffle(queries)
    out = []
    for q in queries:
        if len(out) >= n:
            break
        out.extend(by_query[q])
    return out


class ForwardTimer:
    """Replaces `module.forward` on the instance, so every forward is bracketed by CUDA events while `on`."""

    def __init__(self, module):
        self.orig, self.events, self.on = module.forward, [], False
        module.forward = self._timed

    def _timed(self, *args, **kwargs):
        if not self.on:
            return self.orig(*args, **kwargs)
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        out = self.orig(*args, **kwargs)
        end.record()
        self.events.append((start, end))
        return out

    def take_ms(self) -> float:
        torch.cuda.synchronize()
        ms = sum(s.elapsed_time(e) for s, e in self.events)
        self.events = []
        return ms


def measure(model, pairs: list) -> dict:
    timer = ForwardTimer(model)
    sweep = {}
    with torch.no_grad():
        for bs in (START_BS * 2 ** i for i in itertools.count()):
            n = -(-len(pairs) // bs) * bs
            sample = sorted((pairs[i % len(pairs)] for i in range(n)), key=lambda p: -len(p[0]) - len(p[1]))
            try:
                model.predict(sample[:2 * bs], batch_size=bs, show_progress_bar=False)  # warmup + OOM probe
                timer.on = True
                for _ in range(PASSES):
                    model.predict(sample, batch_size=bs, show_progress_bar=False)
                ms = timer.take_ms()
            except torch.cuda.OutOfMemoryError:
                sweep[bs] = None
                break
            finally:
                timer.on, timer.events = False, []
                torch.cuda.empty_cache()
            sweep[bs] = PASSES * n / (ms / 1000)
            print(f"[speed]   bs={bs}: {sweep[bs]:.1f} pairs/s", flush=True)
    model.forward = timer.orig
    ok = {b: v for b, v in sweep.items() if v is not None}
    best = max(ok, key=ok.get) if ok else None
    return {"pps": ok.get(best), "pps_batch_size": best, "pps_sweep": {str(b): v for b, v in sweep.items()}}


def gpu_shared(device: str) -> bool:
    """Whether another process is on this GPU -- if so the timing is suspect."""
    uuid = str(torch.cuda.get_device_properties(device).uuid)
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout
    pids = {int(pid) for gpu, pid in (line.split(", ") for line in out.splitlines() if line) if uuid in gpu}
    return bool(pids - {os.getpid()})


def speed_record(model, tap: PairTap, attn: str | None, device: str, n: int) -> dict:
    """Time the pairs `tap` collected for one task; the fields evaluate.py adds to its json."""
    pairs = sample_pairs(tap.take(), n)
    shared = gpu_shared(device)
    res = measure(model, pairs)
    shared = shared or gpu_shared(device)
    return {**res, "_pps_n_pairs": len(pairs), "_pps_n_queries": len({q for q, _ in pairs}),
            "_pps_passes": PASSES, "_pps_dtype": "bfloat16", "_pps_attn": attn,
            "_pps_gpu": torch.cuda.get_device_name(device), "_pps_gpu_shared": shared}
