# Serving KURE-v2

Runnable versions of the serving configurations benchmarked in the [KURE-v2 model card](https://huggingface.co/nlpai-lab/KURE-v2): encode a Korean retrieval dataset, build an index, and get quality + latency + index size in one command.

```
run.py      end-to-end experiment (encode -> build -> search -> report)
indexes.py  the index backends (exact MaxSim, PLAID, asymmetric binary, binary IVF)
```

## Run

No setup: the `# /// script` block at the top of `run.py` declares its own dependencies (sentence-transformers >= 6.0, fast-plaid, faiss-cpu, torch 2.8/cu128), so `uv run deploy/late-interaction/run.py` ignores the project venv and resolves an isolated, cached environment for the script on first run. This is deliberate: the project venv pins pylate, which pins fast-plaid to the 1.4.6.x line, while these measurements use fast-plaid 1.6.0 and the sentence-transformers MultiVectorEncoder encode path, the exact stack behind the model-card numbers.

```bash
uv run deploy/late-interaction/run.py --index maxsim                     # exact MaxSim (quality ceiling)
uv run deploy/late-interaction/run.py --index plaid --nbits 4            # PLAID, n-bit residuals
uv run deploy/late-interaction/run.py --index plaid --nbits 4 --pool 3   # PLAID + token pooling x3
uv run deploy/late-interaction/run.py --index binary                     # asymmetric binary, exhaustive
uv run deploy/late-interaction/run.py --index binary --pool 3            # token pooling x3 + binary
uv run deploy/late-interaction/run.py --index plaid-binary               # PLAID candidates + binary rerank
uv run deploy/late-interaction/run.py --index plaid-binary --pool 3      # PLAID + pooling x3 + binary
uv run deploy/late-interaction/run.py --index binary-ivf                 # binary IVF + exact rerank
```

Each run prints nDCG@10, index size and build time, batch-1 query-encode and search latency (mean / p95), and the end-to-end QPS of the serial encode + search pipeline:

```
=== nlpai-lab/KURE-v2 | plaid | yjoonjang/markers_bm (720 docs, 714 queries) ===
nDCG@10:            0.96xx
index size:          xx.x MB (build x.xs)
query encode (bs=1): xx.x ms mean / xx.x ms p95
index search (bs=1): xx.x ms mean / xx.x ms p95
end-to-end:          xx.x QPS, xx.x ms p95
```

It also appends the same numbers as one JSON line to `results/<dataset>.jsonl` next to the script (config, dataset size, quality, latency, device), so repeated runs accumulate into a comparable log; `results/markers_bm.jsonl` in the repo holds reference runs of every backend on one A100 80GB. `--out <path>` redirects, `--out none` skips.

## Backends

| `--index` | What it does | Trade-off |
|---|---|---|
| `maxsim` | Exhaustive exact MaxSim over bf16 token vectors | Quality ceiling; largest index, slowest at scale |
| `plaid` | PLAID (fast-plaid): centroid + 4-bit residual codes, centroid-driven candidate generation | ~10x smaller, near-exact quality |
| `binary` | Asymmetric binary quantization: documents keep 1 sign bit per dimension, queries stay full precision, scoring is the exact query x {-1,+1} dot | 32x smaller than fp32; exhaustive scan |
| `plaid-binary` | Two-stage: PLAID (from the original vectors) retrieves `--rerank-depth` candidates, re-scored with the exact asymmetric binary MaxSim | Binary-quality results without the exhaustive scan; stores PLAID index + 1-bit codes |
| `binary-ivf` | Same 1-bit codes; candidates via faiss binary IVF (Hamming over inverted lists), top candidates re-scored with the exact asymmetric MaxSim | Binary-level size with sublinear search; the query is never quantized in the score that ranks the output |

`--pool N` applies hierarchical token pooling at document-encode time (clusters each document's tokens and averages within clusters, shrinking every index ~Nx); it composes with any backend. Queries are never pooled.

## Options

- `--dataset`: any BeIR-style HF dataset with `corpus` / `queries` / `default` (qrels) configs. Default `yjoonjang/markers_bm` (AutoRAG, small enough for a quick pass).
- `--limit N`: cap corpus and query count for a smoke run (e.g. `--limit 100` on CPU).
- `--device`: defaults to cuda when available. GPU is strongly recommended for encoding.
- Backend knobs use the benchmark defaults: `--nbits 4` (PLAID), `--nprobe 32`, `--topk-tokens 128`, `--rerank-depth 1000` (binary IVF).

## Relation to the model card numbers

The model card's Serving section reports the same configurations measured under a fixed protocol (9 Korean MTEB retrieval tasks, single A100 80GB, batch-1 serial encode + search, warmup + repeated timed queries). This directory reproduces the mechanics with a lighter protocol on one dataset, so expect the same ordering between methods rather than identical numbers. One caveat at demo scale: on a corpus this small (720 documents) PLAID's centroid store and per-call overhead dominate, so it shows up larger and slower than exhaustive scans here; its size and speed advantages appear on large corpora (see the model card's MIRACL figures).
