# Dense retrieval results

Korean results are in `<model>/mteb_kor`, English RTEB results in `<model>/rteb_eng`.

```bash
uv run python eval/dense/make_leaderboard_mteb_kor.py   # MTEB(kor, v2)
uv run python eval/dense/make_leaderboard_rteb_eng.py   # RTEB
```

## Per-task nDCG@10

| Model | Type | Params | Avg | AutoRAG | PublicHealthQA | Ko-StrategyQA | LawIRKo | SQuADKorV1 | Belebele | MrTidy | MLDR | MIRACL |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **nlpai-lab/KURE-v2** | Late-interaction | 154M | **0.8160** | **0.9718** | 0.8228 | 0.8070 | 0.7550 | **0.9846** | 0.9660 | 0.5974 | **0.7158** | 0.7237 |
| nvidia/Nemotron-3-Embed-8B-BF16 | Dense | 8.0B | 0.8127 | 0.9311 | 0.8898 | **0.8405** | 0.8481 | 0.9650 | 0.9714 | **0.6540** | 0.4785 | 0.7363 |
| yjoonjang/colbert-ko-en-v2 | Late-interaction | 149M | 0.8063 | 0.9686 | 0.8222 | 0.7940 | 0.7181 | **0.9846** | 0.9687 | 0.5783 | 0.6992 | 0.7230 |
| **nlpai-lab/KURE-v3** | Dense | 1.1B | 0.8027 | 0.9324 | 0.8639 | 0.8192 | 0.8158 | 0.9666 | 0.9673 | 0.6320 | 0.4781 | **0.7489** |
| sionic-ai/comsat-embed-ko-8b-preview | Dense | 7.6B | 0.7927 | 0.8518 | 0.8871 | 0.8394 | 0.8164 | 0.9168 | **0.9853** | 0.6253 | 0.5157 | 0.6964 |
| lightonai/mLateOn | Late-interaction | 307M | 0.7906 | 0.9392 | 0.8061 | 0.7905 | 0.6431 | 0.9803 | 0.9608 | 0.5817 | 0.7005 | 0.7135 |
| Qwen/Qwen3-Embedding-8B | Dense | 7.6B | 0.7826 | 0.8276 | 0.8721 | 0.8363 | 0.8171 | 0.9063 | 0.9824 | 0.6187 | 0.5046 | 0.6783 |
| nvidia/Nemotron-3-Embed-1B-BF16 | Dense | 1.1B | 0.7817 | 0.9074 | 0.8642 | 0.8136 | 0.7758 | 0.9505 | 0.9629 | 0.6084 | 0.4202 | 0.7325 |
| Qwen/Qwen3-Embedding-4B | Dense | 4.0B | 0.7737 | 0.8431 | 0.8693 | 0.8270 | 0.7769 | 0.9044 | 0.9522 | 0.6076 | 0.5022 | 0.6803 |
| perplexity-ai/pplx-embed-v1-4b | Dense | 4.0B | 0.7705 | 0.8321 | 0.9042 | 0.8326 | 0.7736 | 0.9402 | 0.9624 | 0.5778 | 0.4343 | 0.6775 |
| microsoft/harrier-oss-v1-27b | Dense | 27.0B | 0.7667 | 0.8176 | 0.8971 | 0.8361 | **0.8737** | 0.9204 | 0.9546 | 0.5306 | 0.4046 | 0.6653 |
| dragonkue/snowflake-arctic-embed-l-v2.0-ko | Dense | 568M | 0.7653 | 0.9093 | 0.8337 | 0.8050 | 0.7735 | 0.9447 | 0.9518 | 0.5712 | 0.4304 | 0.6685 |
| codefuse-ai/F2LLM-v2-8B | Dense | 7.6B | 0.7638 | 0.7678 | **0.9380** | 0.8371 | 0.8405 | 0.8874 | 0.9513 | 0.6162 | 0.4047 | 0.6313 |
| telepix/PIXIE-Rune-v1.5 | Dense | 568M | 0.7618 | 0.8927 | 0.8426 | 0.8064 | 0.7705 | 0.9457 | 0.9617 | 0.5492 | 0.4482 | 0.6393 |
| **nlpai-lab/KURE-v1** | Dense | 568M | 0.7616 | 0.8708 | 0.8193 | 0.7999 | 0.7426 | 0.9357 | 0.9502 | 0.5909 | 0.4637 | 0.6816 |
| kakaocorp/kanana-2-3b-embedding | Dense | 3.2B | 0.7584 | 0.8393 | 0.8772 | 0.8329 | 0.8057 | 0.9246 | 0.9437 | 0.5967 | 0.3590 | 0.6466 |
| nvidia/llama-nemotron-embed-1b-v2 | Dense | 1.2B | 0.7564 | 0.8708 | 0.8177 | 0.8055 | 0.7329 | 0.9345 | 0.9519 | 0.5988 | 0.3952 | 0.7004 |
| dragonkue/BGE-m3-ko | Dense | 568M | 0.7547 | 0.8738 | 0.8155 | 0.7959 | 0.7322 | 0.9414 | 0.9503 | 0.6099 | 0.3899 | 0.6833 |
| BAAI/bge-m3 | Dense | 568M | 0.7509 | 0.8301 | 0.8041 | 0.7941 | 0.7174 | 0.9038 | 0.9315 | 0.6471 | 0.4287 | 0.7015 |
| jinaai/jina-embeddings-v5-text-small-retrieval | Dense | 596M | 0.7419 | 0.8409 | 0.8664 | 0.8096 | 0.7483 | 0.8926 | 0.9346 | 0.5779 | 0.3315 | 0.6752 |
| perplexity-ai/pplx-embed-v1-0.6b | Dense | 596M | 0.7390 | 0.8414 | 0.8255 | 0.8100 | 0.7444 | 0.9174 | 0.9393 | 0.5524 | 0.3179 | 0.7026 |
| perplexity-ai/pplx-embed-v1-late-0.6b | Late-interaction | 596M | 0.7381 | 0.8557 | 0.8089 | 0.7973 | 0.7285 | 0.9696 | 0.9548 | 0.5400 | 0.2816 | 0.7064 |
| **nlpai-lab/KoE5** | Dense | 560M | 0.7337 | 0.8434 | 0.8351 | 0.8001 | 0.7756 | 0.8980 | 0.9425 | 0.5841 | 0.3015 | 0.6235 |
| **nlpai-lab/KURE-v2-unsupervised** | Late-interaction | 154M | 0.7283 | 0.8822 | 0.8240 | 0.7818 | 0.7680 | 0.9427 | 0.9525 | 0.3496 | 0.6130 | 0.4411 |
| google/embeddinggemma-300m | Dense | 308M | 0.7254 | 0.8187 | 0.8170 | 0.8008 | 0.6850 | 0.9114 | 0.9430 | 0.5253 | 0.3844 | 0.6434 |
| kakaocorp/kanana-2-1.3b-embedding | Dense | 1.3B | 0.7172 | 0.8108 | 0.8647 | 0.8146 | 0.7653 | 0.9014 | 0.9222 | 0.5540 | 0.2088 | 0.6129 |
| Qwen/Qwen3-Embedding-0.6B | Dense | 596M | 0.7140 | 0.8338 | 0.8107 | 0.7586 | 0.7526 | 0.8490 | 0.9124 | 0.4870 | 0.4199 | 0.6023 |
| dragonkue/colbert-ko-0.1b | Late-interaction | 149M | 0.6776 | 0.9700 | 0.7482 | 0.7364 | 0.4475 | 0.9794 | 0.9644 | 0.3966 | 0.2872 | 0.5685 |
| yjoonjang/colbert-ko-v1 | Late-interaction | 149M | 0.6282 | 0.9557 | 0.6783 | 0.6560 | 0.4823 | 0.9594 | 0.9154 | 0.3279 | 0.2214 | 0.4575 |

## KURE-v3 at Matryoshka dimensions

Embeddings truncated to their first N dimensions (`truncate_dim`). nDCG@10 per task, with the average Recall@10.

| Dimension | Avg | Avg Recall@10 | AutoRAG | PublicHealthQA | Ko-StrategyQA | LawIRKo | SQuADKorV1 | Belebele | MrTidy | MLDR | MIRACL |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2048 | 0.8027 | 0.8857 | 0.9324 | 0.8639 | 0.8192 | 0.8158 | 0.9666 | 0.9673 | 0.6320 | 0.4781 | 0.7489 |
| 1024 | 0.7953 | 0.8804 | 0.9257 | 0.8602 | 0.8127 | 0.8085 | 0.9632 | 0.9664 | 0.6154 | 0.4700 | 0.7353 |
| 768 | 0.7911 | 0.8752 | 0.9129 | 0.8685 | 0.8090 | 0.8002 | 0.9607 | 0.9626 | 0.6142 | 0.4582 | 0.7335 |
| 512 | 0.7834 | 0.8656 | 0.9135 | 0.8569 | 0.8038 | 0.7941 | 0.9555 | 0.9622 | 0.5992 | 0.4382 | 0.7270 |
| 256 | 0.7547 | 0.8479 | 0.9034 | 0.8554 | 0.7709 | 0.7593 | 0.9322 | 0.9373 | 0.5777 | 0.3648 | 0.6913 |

## RTEB(eng): per-task nDCG@10

The eight public non-code tasks of RTEB(eng, beta). Each task score is the mean over its splits (CUREv1: its English subset only, as RTEB(eng) defines it), so the average is not the RTEB leaderboard number, which also covers RTEB's private and code tasks.

| Model | Type | Params | Avg | CUREv1 | ChatDoctor | FinQA | AILACasedocs | LegalSummarization | HC3Finance | FinanceBench | AILAStatutes |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| nvidia/Nemotron-3-Embed-8B-BF16 | Dense | 8.0B | **0.7405** | **0.6818** | 0.7695 | **0.8849** | **0.4908** | **0.7677** | **0.7990** | **0.9502** | 0.5804 |
| sionic-ai/comsat-embed-ko-8b-preview | Dense | 7.6B | 0.6942 | 0.6146 | 0.7337 | 0.6476 | 0.4689 | 0.6987 | 0.7084 | 0.8430 | **0.8385** |
| Qwen/Qwen3-Embedding-8B | Dense | 7.6B | 0.6887 | 0.6294 | 0.7323 | 0.6339 | 0.4409 | 0.7069 | 0.7196 | 0.8280 | 0.8190 |
| **nlpai-lab/KURE-v3** | Dense | 1.1B | 0.6885 | 0.6518 | 0.6713 | 0.8632 | 0.4267 | 0.7180 | 0.7018 | 0.8753 | 0.5997 |
| perplexity-ai/pplx-embed-v1-4b | Dense | 4.0B | 0.6829 | 0.5774 | 0.7407 | 0.8358 | 0.4546 | 0.6806 | 0.7289 | 0.8438 | 0.6012 |
| nvidia/Nemotron-3-Embed-1B-BF16 | Dense | 1.1B | 0.6676 | 0.6169 | 0.6809 | 0.8242 | 0.4287 | 0.7267 | 0.6444 | 0.8540 | 0.5649 |
| Qwen/Qwen3-Embedding-4B | Dense | 4.0B | 0.6601 | 0.6104 | 0.7138 | 0.6339 | 0.3938 | 0.6612 | 0.6877 | 0.7662 | 0.8140 |
| codefuse-ai/F2LLM-v2-8B | Dense | 7.6B | 0.6426 | 0.5541 | **0.7975** | 0.6108 | 0.4498 | 0.6499 | 0.7596 | 0.8744 | 0.4448 |
| perplexity-ai/pplx-embed-v1-0.6b | Dense | 596M | 0.6407 | 0.5888 | 0.6885 | 0.7333 | 0.3865 | 0.6804 | 0.6822 | 0.7875 | 0.5788 |
| jinaai/jina-embeddings-v5-text-small-retrieval | Dense | 596M | 0.6111 | 0.5701 | 0.7113 | 0.5563 | 0.4378 | 0.6485 | 0.6220 | 0.8076 | 0.5352 |
| Qwen/Qwen3-Embedding-0.6B | Dense | 596M | 0.5874 | 0.5175 | 0.6208 | 0.5740 | 0.3525 | 0.6234 | 0.5471 | 0.7375 | 0.7261 |
| lightonai/mLateOn | Late-interaction | 307M | 0.5738 | 0.6268 | 0.6024 | 0.7887 | 0.2721 | 0.6831 | 0.6501 | 0.7722 | 0.1952 |
| google/embeddinggemma-300m | Dense | 308M | 0.5568 | 0.6012 | 0.6342 | 0.5437 | 0.3468 | 0.6833 | 0.5997 | 0.7748 | 0.2706 |
| telepix/PIXIE-Rune-v1.5 | Dense | 568M | 0.5488 | 0.5740 | 0.5976 | 0.6228 | 0.3590 | 0.6711 | 0.5252 | 0.7865 | 0.2540 |
| kakaocorp/kanana-2-3b-embedding | Dense | 3.2B | 0.5421 | 0.6032 | 0.6192 | 0.5292 | 0.3288 | 0.6345 | 0.5717 | 0.7978 | 0.2524 |
| perplexity-ai/pplx-embed-v1-late-0.6b | Late-interaction | 596M | 0.5322 | 0.5998 | 0.4097 | 0.6860 | 0.2432 | 0.6695 | 0.6757 | 0.7089 | 0.2648 |
| **nlpai-lab/KURE-v1** | Dense | 568M | 0.5246 | 0.5566 | 0.5665 | 0.5977 | 0.3685 | 0.6300 | 0.5080 | 0.7457 | 0.2235 |
| BAAI/bge-m3 | Dense | 568M | 0.5194 | 0.5531 | 0.5797 | 0.5570 | 0.3597 | 0.6329 | 0.5100 | 0.6667 | 0.2961 |
| kakaocorp/kanana-2-1.3b-embedding | Dense | 1.3B | 0.5013 | 0.5813 | 0.5418 | 0.4834 | 0.2229 | 0.6324 | 0.5120 | 0.7556 | 0.2806 |
| **nlpai-lab/KURE-v2** | Late-interaction | 154M | 0.4803 | 0.5900 | 0.3671 | 0.7870 | 0.1782 | 0.6316 | 0.4694 | 0.6780 | 0.1415 |
