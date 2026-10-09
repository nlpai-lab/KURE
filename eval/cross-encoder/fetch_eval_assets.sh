#!/usr/bin/env bash
# Fetch the stage-1 candidate pools of instructkr/reranker-simple-benchmark: the cached BM25 top-1000
# document ids per query (stage1_pools/<task>_id.jsonl, 121 MB). They are the benchmark's artifacts,
# so they are gitignored and pulled on demand. Run once before evaluate.py.
set -eu
cd "$(dirname "$0")"
RAW=https://raw.githubusercontent.com/instructkr/reranker-simple-benchmark/main/eval
mkdir -p stage1_pools
for t in AutoRAGRetrieval BelebeleRetrieval Ko-StrategyQA LawIRKo MIRACLRetrieval \
         MrTidyRetrieval MultiLongDocRetrieval PublicHealthQA SQuADKorV1Retrieval; do
    curl -fsSL -o "stage1_pools/${t}_id.jsonl" "$RAW/results/stage1/top_1k_qrels/${t}_id.jsonl"
done
echo "fetched $(ls stage1_pools | wc -l) stage-1 pools"
