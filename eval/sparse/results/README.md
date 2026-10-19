# Sparse retrieval results

Korean results are in `<model>/mteb_kor`, English results in `<model>/beir_eng` (BEIR) and `<model>/rteb_eng` (RTEB).

```bash
uv run python eval/sparse/make_leaderboard_mteb_kor.py   # MTEB(kor, v2)
uv run python eval/sparse/make_leaderboard_beir_eng.py   # BEIR
uv run python eval/sparse/make_leaderboard_rteb_eng.py   # RTEB
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

BEIR-13 is the average without MS MARCO. Query and document dims and FLOPs are averaged over all 14 tasks.

| Model | Params | Avg | BEIR-13 | ArguAna | ClimateFEVER | DBPedia | FEVER | FiQA | HotpotQA | MS MARCO | NFCorpus | NQ | Quora | SCIDOCS | SciFact | TREC-COVID | Touche-2020 | Query dims | Document dims | FLOPs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **nlpai-lab/KURE-Sparse-base** | 593M | **0.5782** | **0.5864** | 0.5896 | **0.3959** | **0.4837** | **0.9325** | **0.4720** | **0.8084** | **0.4713** | 0.3678 | **0.6758** | 0.8252 | 0.1741 | 0.7183 | 0.8260 | 0.3535 | 215.6 | 1006.2 | 52.71 |
| utahnlp/laconic-1b | 1.2B | 0.5718 | 0.5831 | 0.6127 | 0.3952 | 0.4705 | 0.8871 | 0.4427 | 0.8018 | 0.4256 | **0.3958** | 0.6642 | 0.8072 | **0.2397** | **0.7596** | **0.8300** | 0.2737 | 140.4 | 1143.6 | 50.43 |
| Linkup-Platform/linkup-sparseup-embed-v1 | 188M | 0.5550 | 0.5645 | 0.5648 | 0.3747 | 0.4150 | 0.8969 | 0.4687 | 0.7761 | 0.4318 | 0.3523 | 0.5717 | 0.8656 | 0.1852 | 0.7376 | 0.7823 | 0.3478 | 81.8 | 260.5 | 8.11 |
| SparseUp (reported) | 188M | 0.554 | 0.564 | 0.568 | 0.361 | 0.415 | 0.897 | 0.469 | 0.772 | 0.431 | 0.351 | 0.571 | 0.866 | 0.184 | 0.739 | 0.784 | 0.350 | - | - | - |
| omai-research/milco-650m | 593M | 0.5477 | 0.5545 | **0.6157** | 0.3079 | 0.4501 | 0.8353 | 0.4262 | 0.7780 | 0.4595 | 0.3629 | 0.6495 | **0.8818** | 0.1704 | 0.7020 | 0.7476 | 0.2813 | 134.4 | 529.6 | 19.63 |
| opensearch-project/opensearch-neural-sparse-encoding-doc-v3-gte | 137M | 0.5373 | 0.5459 | 0.5196 | 0.3123 | 0.4553 | 0.8598 | 0.4069 | 0.7158 | 0.4259 | 0.3592 | 0.5817 | 0.8734 | 0.1675 | 0.7251 | 0.7330 | **0.3866** | 21.0 | 208.4 | 1.58 |
| **nlpai-lab/KURE-Sparse-nano** | 149M | 0.5244 | 0.5317 | 0.5075 | 0.2865 | 0.4214 | 0.9079 | 0.4080 | 0.7476 | 0.4289 | 0.3340 | 0.5647 | 0.7825 | 0.1581 | 0.7120 | 0.7637 | 0.3183 | 201.5 | 978.8 | 36.51 |
| naver/splade-v3 | 110M | 0.5163 | 0.5202 | 0.4880 | 0.2566 | 0.4514 | 0.8098 | 0.3810 | 0.6924 | 0.4667 | 0.3630 | 0.5867 | 0.8138 | 0.1557 | 0.7171 | 0.7346 | 0.3119 | 72.5 | 210.7 | 7.56 |
| telepix/PIXIE-Splade-v1.5 | 149M | 0.4040 | 0.4149 | 0.2185 | 0.1302 | 0.3519 | 0.6475 | 0.2753 | 0.5756 | 0.2629 | 0.3325 | 0.3858 | 0.6878 | 0.1468 | 0.6694 | 0.7068 | 0.2650 | 329.9 | 1326.0 | 118.57 |

## RTEB(eng): per-task nDCG@10

The eight public non-code tasks of RTEB(eng, beta). Each task score is the mean over its splits and subsets, so the average is not the RTEB leaderboard number, which also covers RTEB's private and code tasks.

| Model | Params | Avg | CUREv1 | ChatDoctor | FinQA | AILACasedocs | LegalSummarization | HC3Finance | FinanceBench | AILAStatutes | Query dims | Document dims | FLOPs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| utahnlp/laconic-1b | 1.2B | **0.5415** | 0.5340 | 0.5272 | 0.6727 | **0.3436** | 0.6064 | 0.5893 | 0.7449 | **0.3142** | 227.1 | 1331.6 | 107.21 |
| **nlpai-lab/KURE-Sparse-base** | 593M | 0.5093 | **0.5996** | 0.3914 | **0.7838** | 0.2298 | 0.5494 | **0.5951** | **0.7525** | 0.1727 | 787.0 | 1899.1 | 277.55 |
| Linkup-Platform/linkup-sparseup-embed-v1 | 188M | 0.4941 | 0.4010 | **0.5424** | 0.5805 | 0.2406 | 0.6430 | 0.5900 | 0.6852 | 0.2700 | 152.4 | 325.7 | 25.06 |
| **nlpai-lab/KURE-Sparse-nano** | 149M | 0.4750 | 0.3591 | 0.4730 | 0.7042 | 0.2860 | 0.6399 | 0.4438 | 0.7005 | 0.1939 | 359.8 | 1274.5 | 109.86 |
| opensearch-project/opensearch-neural-sparse-encoding-doc-v3-gte | 137M | 0.4721 | 0.3004 | 0.5015 | 0.7344 | 0.2596 | **0.6554** | 0.4265 | 0.6855 | 0.2139 | 82.7 | 359.2 | 12.75 |
| naver/splade-v3 | 110M | 0.4596 | 0.3306 | 0.5036 | 0.6374 | 0.3153 | 0.5663 | 0.4118 | 0.6878 | 0.2241 | 203.2 | 286.7 | 35.00 |
| omai-research/milco-650m | 593M | 0.4301 | 0.5674 | 0.2344 | 0.7399 | 0.2197 | 0.3078 | 0.5451 | 0.6513 | 0.1747 | 548.0 | 1375.1 | 163.65 |
| telepix/PIXIE-Splade-v1.5 | 149M | 0.3630 | 0.3414 | 0.2178 | 0.5751 | 0.2082 | 0.5216 | 0.3042 | 0.5180 | 0.2181 | 860.1 | 1835.6 | 391.32 |
