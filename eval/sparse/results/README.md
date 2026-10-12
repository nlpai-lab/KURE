# Sparse retrieval results

Korean results are in `<model>/ko`, English results in `<model>/en`.

```bash
uv run python eval/sparse/make_leaderboard.py      # ko
uv run python eval/sparse/make_leaderboard_en.py   # en
```

## MTEB(kor, v2) retrieval: per-task nDCG@10

| Model | Params | Avg | AutoRAG | PublicHealthQA | Ko-StrategyQA | LawIRKo | SQuADKorV1 | Belebele | MrTidy | MLDR | MIRACL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **nlpai-lab/KURE-Sparse-base** | 593M | **0.7984** | 0.9445 | 0.8099 | **0.8188** | 0.7244 | **0.9815** | **0.9703** | 0.6255 | **0.5716** | **0.7395** |
| omai-research/milco-650m | 593M | 0.7824 | 0.9483 | **0.8219** | 0.8143 | 0.6724 | 0.9700 | 0.9623 | **0.6997** | 0.4318 | 0.7209 |
| **nlpai-lab/KURE-Sparse-nano** | 149M | 0.7716 | 0.9360 | 0.7711 | 0.7924 | 0.7226 | 0.9774 | 0.9608 | 0.5618 | 0.5133 | 0.7091 |
| telepix/PIXIE-Splade-v1.5 | 149M | 0.7608 | 0.9412 | 0.7910 | 0.7993 | **0.7667** | 0.9687 | 0.9651 | 0.5132 | 0.4938 | 0.6085 |
| yjoonjang/splade-ko-v1 | 149M | 0.7526 | 0.9423 | 0.7977 | 0.7583 | 0.6509 | 0.9754 | 0.9576 | 0.5299 | 0.5162 | 0.6451 |
| telepix/PIXIE-Splade-v1.0 | 149M | 0.6636 | **0.9597** | 0.7860 | 0.7352 | 0.5330 | 0.9417 | 0.9413 | 0.2909 | 0.3991 | 0.3858 |
| opensearch-project/opensearch-neural-sparse-encoding-multilingual-v1 | 167M | 0.5444 | 0.6482 | 0.5506 | 0.5553 | 0.3554 | 0.7770 | 0.7321 | 0.5526 | 0.1219 | 0.6064 |

## BEIR + MS MARCO: per-task nDCG@10

BEIR-13 is the average without MS MARCO.

| Model | Params | Avg | BEIR-13 | ArguAna | ClimateFEVER | DBPedia | FEVER | FiQA | HotpotQA | MS MARCO | NFCorpus | NQ | Quora | SCIDOCS | SciFact | TREC-COVID | Touche-2020 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **nlpai-lab/KURE-Sparse-base** | 593M | **0.5782** | **0.5864** | 0.5896 | **0.3959** | **0.4837** | **0.9325** | **0.4720** | **0.8084** | **0.4713** | **0.3678** | **0.6758** | 0.8252 | 0.1741 | 0.7183 | **0.8260** | **0.3535** |
| Linkup-Platform/linkup-sparseup-embed-v1 | 188M | 0.5550 | 0.5645 | 0.5648 | 0.3747 | 0.4150 | 0.8969 | 0.4687 | 0.7761 | 0.4318 | 0.3523 | 0.5717 | 0.8656 | **0.1852** | **0.7376** | 0.7823 | 0.3478 |
| SparseUp (reported) | 188M | 0.554 | 0.564 | 0.568 | 0.361 | 0.415 | 0.897 | 0.469 | 0.772 | 0.431 | 0.351 | 0.571 | 0.866 | 0.184 | 0.739 | 0.784 | 0.350 |
| omai-research/milco-650m | 593M | 0.5477 | 0.5545 | **0.6157** | 0.3079 | 0.4501 | 0.8353 | 0.4262 | 0.7780 | 0.4595 | 0.3629 | 0.6495 | **0.8818** | 0.1704 | 0.7020 | 0.7476 | 0.2813 |
| **nlpai-lab/KURE-Sparse-nano** | 149M | 0.5244 | 0.5317 | 0.5075 | 0.2865 | 0.4214 | 0.9079 | 0.4080 | 0.7476 | 0.4289 | 0.3340 | 0.5647 | 0.7825 | 0.1581 | 0.7120 | 0.7637 | 0.3183 |
| telepix/PIXIE-Splade-v1.5 | 149M | 0.4040 | 0.4149 | 0.2185 | 0.1302 | 0.3519 | 0.6475 | 0.2753 | 0.5756 | 0.2629 | 0.3325 | 0.3858 | 0.6878 | 0.1468 | 0.6694 | 0.7068 | 0.2650 |
