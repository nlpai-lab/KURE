# Result files backing the leaderboard

Per-task mteb result JSONs for every model in the README leaderboard, over the nine
MTEB(kor, v2) retrieval tasks. Regenerate the table with:

```bash
uv run python eval/dense/make_leaderboard.py
```

## Sources

- **Official [MTEB results repository](https://github.com/embeddings-benchmark/results)**
  (directory name = the model revision): KURE-v2, KURE-v1, KoE5, comsat-embed-ko-8b-preview,
  Qwen3-Embedding-8B/4B, harrier-oss-v1-27b, snowflake-arctic-embed-l-v2.0-ko, F2LLM-v2-8B,
  PIXIE-Rune-v1.5, BGE-m3-ko, bge-m3.
- **Local mteb runs** (directory name `local`; models not yet in the official repository):
  mLateOn, pplx-embed-v1-late-0.6b, colbert-ko-0.1b, colbert-ko-v1, colbert-ko-en-v2,
  KURE-v2-unsupervised, Nemotron-3-Embed-8B-BF16, Nemotron-3-Embed-1B-BF16,
  llama-nemotron-embed-1b-v2, pplx-embed-v1-4b, pplx-embed-v1-0.6b, Qwen3-Embedding-0.6B.
- **BAAI/bge-m3 exception**: the official files for PublicHealthQA, MrTidyRetrieval,
  MultiLongDocRetrieval, and MIRACLRetrieval do not include the Korean subsets, so those four
  files come from this repository's own earlier evaluation run.
