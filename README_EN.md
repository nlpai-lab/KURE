# 🔎 KURE: Korea University Retrieval Embedding model

## Update Logs
- 2024.12.21: [🤗 KURE-v1](https://huggingface.co/nlpai-lab/KURE-v1) and the [MTEB-ko-retrieval Leaderboard](https://github.com/nlpai-lab/KURE/blob/main/README_EN.md#mteb-ko-retrieval-leaderboard) have been released.
- 2024.10.02: [🤗 KoE5](https://huggingface.co/nlpai-lab/KoE5) and [🤗 ko-triplet-v1.0](https://huggingface.co/datasets/nlpai-lab/ko-triplet-v1.0) have been released.

---

<br>

KURE is a Korean-specific embedding model developed by Korea University's [NLP & AI Lab](http://nlp.korea.ac.kr/) and [HIAI Research Institute](http://hiai.korea.ac.kr).

We are excited to release KURE.
<br/>

## How to Use KURE Models
### With sentence-transformers
```bash
pip install sentence-transformers
```

You can run the model with the following example code.

```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub

model = SentenceTransformer("nlpai-lab/KURE-v1")
# model = SentenceTransformer("nlpai-lab/KoE5")

# Run inference
sentences = [
    '헌법과 법원조직법은 어떤 방식을 통해 기본권 보장 등의 다양한 법적 모색을 가능하게 했어',
    '4. 시사점과 개선방향 앞서 살펴본 바와 같이 우리 헌법과 ｢법원조직 법｣은 대법원 구성을 다양화하여 기본권 보장과 민주주의 확립에 있어 다각적인 법적 모색을 가능하게 하는 것을 근본 규범으로 하고 있다. 더욱이 합의체로서의 대법원 원리를 채택하고 있는 것 역시 그 구성의 다양성을 요청하는 것으로 해석된다. 이와 같은 관점에서 볼 때 현직 법원장급 고위법관을 중심으로 대법원을 구성하는 관행은 개선할 필요가 있는 것으로 보인다.',
    '연방헌법재판소는 2001년 1월 24일 5:3의 다수견해로 「법원조직법」 제169조 제2문이 헌법에 합치된다는 판결을 내렸음 ○ 5인의 다수 재판관은 소송관계인의 인격권 보호, 공정한 절차의 보장과 방해받지 않는 법과 진실 발견 등을 근거로 하여 텔레비전 촬영에 대한 절대적인 금지를 헌법에 합치하는 것으로 보았음 ○ 그러나 나머지 3인의 재판관은 행정법원의 소송절차는 특별한 인격권 보호의 이익도 없으며, 텔레비전 공개주의로 인해 법과 진실 발견의 과정이 언제나 위태롭게 되는 것은 아니라면서 반대의견을 제시함 ○ 왜냐하면 행정법원의 소송절차에서는 소송당사자가 개인적으로 직접 심리에 참석하기보다는 변호사가 참석하는 경우가 많으며, 심리대상도 사실문제가 아닌 법률문제가 대부분이기 때문이라는 것임 □ 한편, 연방헌법재판소는 「연방헌법재판소법」(Bundesverfassungsgerichtsgesetz: BVerfGG) 제17a조에 따라 제한적이나마 재판에 대한 방송을 허용하고 있음 ○ 「연방헌법재판소법」 제17조에서 「법원조직법」 제14절 내지 제16절의 규정을 준용하도록 하고 있지만, 녹음이나 촬영을 통한 재판공개와 관련하여서는 「법원조직법」과 다른 내용을 규정하고 있음',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 1024]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# Results for KURE-v1
# tensor([[1.0000, 0.6967, 0.5306],
#         [0.6967, 1.0000, 0.4427],
#         [0.5306, 0.4427, 1.0000]])

# Results for KoE5
# tensor([[1.0000, 0.6721, 0.3897],
#        [0.6721, 1.0000, 0.3740],
#        [0.3897, 0.3740, 1.0000]])
```

<br/>

## MTEB-ko-retrieval Leaderboard
We evaluated our models on all Korean Retrieval Benchmarks registered in [MTEB](https://github.com/embeddings-benchmark/mteb).
### Korean Retrieval Benchmark
- [Ko-StrategyQA](https://huggingface.co/datasets/taeminlee/Ko-StrategyQA): Korean ODQA multi-hop retrieval dataset (Translated from StrategyQA).
- [AutoRAGRetrieval](https://huggingface.co/datasets/yjoonjang/markers_bm): A Korean document retrieval dataset constructed by parsing PDFs from 5 domains: finance, public, medical, legal, and commerce.
- [MIRACLRetrieval](https://huggingface.co/datasets/miracl/miracl): Wikipedia-based Korean document retrieval dataset.
- [PublicHealthQA](https://huggingface.co/datasets/xhluca/publichealth-qa): Korean document retrieval dataset for the medical and public health domains.
- [BelebeleRetrieval](https://huggingface.co/datasets/facebook/belebele): FLORES-200 based Korean document retrieval dataset.
- [MrTidyRetrieval](https://huggingface.co/datasets/mteb/mrtidy): Wikipedia-based Korean document retrieval dataset.
- [MultiLongDocRetrieval](https://huggingface.co/datasets/Shitao/MLDR): Korean long-document retrieval dataset from various domains.
- [XPQARetrieval](https://huggingface.co/datasets/jinaai/xpqa): Korean document retrieval dataset from various domains.

### Evaluation code
You can add a model to `evaluate.py` to evaluate it using MTEB.
```bash
cd eval
pip install -r requirements.txt
python evaluate.py
```

### Leaderboard
We visualize the evaluation results for all models on all tasks via streamlit.
```bash
streamlit run leaderboard.py
```

Below are the average results for all models across all benchmark datasets.
Detailed results can be found in the `eval/results` folder.

### Top-k 1
| Model                                   | Average Recall | Average Precision | Average NDCG | Average F1 |
|-----------------------------------------|----------------------|------------------------|-------------------|-----------------|
| **nlpai-lab/KURE-v1**                   | **0.52640**          | **0.60551**            | **0.60551**       | **0.55784**     |
| dragonkue/BGE-m3-ko                     | 0.52361              | 0.60394                | 0.60394           | 0.55535         |
| BAAI/bge-m3                             | 0.51778              | 0.59846                | 0.59846           | 0.54998         |
| Snowflake/snowflake-arctic-embed-l-v2.0 | 0.51246              | 0.59384                | 0.59384           | 0.54489         |
| nlpai-lab/KoE5                          | 0.50157              | 0.57790                | 0.57790           | 0.53178         |
| intfloat/multilingual-e5-large          | 0.50052              | 0.57727                | 0.57727           | 0.53122         |
| jinaai/jina-embeddings-v3               | 0.48287              | 0.56068                | 0.56068           | 0.51361         |
| BAAI/bge-multilingual-gemma2            | 0.47904              | 0.55472                | 0.55472           | 0.50916         |
| intfloat/multilingual-e5-large-instruct | 0.47842              | 0.55435                | 0.55435           | 0.50826         |
| intfloat/multilingual-e5-base           | 0.46950              | 0.54490                | 0.54490           | 0.49947         |
| intfloat/e5-mistral-7b-instruct         | 0.46772              | 0.54394                | 0.54394           | 0.49781         |
| Alibaba-NLP/gte-multilingual-base       | 0.46469              | 0.53744                | 0.53744           | 0.49353         |
| Alibaba-NLP/gte-Qwen2-7B-instruct       | 0.46633              | 0.53625                | 0.53625           | 0.49429         |
| openai/text-embedding-3-large           | 0.44884              | 0.51688                | 0.51688           | 0.47572         |
| Salesforce/SFR-Embedding-2_R            | 0.43748              | 0.50815                | 0.50815           | 0.46504         |
| upskyy/bge-m3-korean                    | 0.43125              | 0.50245                | 0.50245           | 0.45945         |
| jhgan/ko-sroberta-multitask             | 0.33788              | 0.38497                | 0.38497           | 0.35678         |

### Top-k 3
| Model                                   | Average Recall | Average Precision | Average NDCG | Average F1 |
|-----------------------------------------|----------------------|------------------------|-------------------|-----------------|
| **nlpai-lab/KURE-v1**                   | **0.68678**          | **0.28711**            | **0.65538**       | **0.39835**     |
| dragonkue/BGE-m3-ko                     | 0.67834              | 0.28385                | 0.64950           | 0.39378         |
| BAAI/bge-m3                             | 0.67526              | 0.28374                | 0.64556           | 0.39291         |
| Snowflake/snowflake-arctic-embed-l-v2.0 | 0.67128              | 0.28193                | 0.64042           | 0.39072         |
| intfloat/multilingual-e5-large          | 0.65807              | 0.27777                | 0.62822           | 0.38423         |
| nlpai-lab/KoE5                          | 0.65174              | 0.27329                | 0.62369           | 0.37882         |
| BAAI/bge-multilingual-gemma2            | 0.64415              | 0.27416                | 0.61105           | 0.37782         |
| jinaai/jina-embeddings-v3               | 0.64116              | 0.27165                | 0.60954           | 0.37511         |
| intfloat/multilingual-e5-large-instruct | 0.64353              | 0.27040                | 0.60790           | 0.37453         |
| Alibaba-NLP/gte-multilingual-base       | 0.63744              | 0.26404                | 0.59695           | 0.36764         |
| Alibaba-NLP/gte-Qwen2-7B-instruct       | 0.63163              | 0.25937                | 0.59237           | 0.36263         |
| intfloat/multilingual-e5-base           | 0.62099              | 0.26144                | 0.59179           | 0.36203         |
| intfloat/e5-mistral-7b-instruct         | 0.62087              | 0.26144                | 0.58917           | 0.36188         |
| openai/text-embedding-3-large           | 0.61035              | 0.25356                | 0.57329           | 0.35270         |
| Salesforce/SFR-Embedding-2_R            | 0.60001              | 0.25253                | 0.56346           | 0.34952         |
| upskyy/bge-m3-korean                    | 0.59215              | 0.25076                | 0.55722           | 0.34623         |
| jhgan/ko-sroberta-multitask             | 0.46930              | 0.18994                | 0.43293           | 0.26696         |

### Top-k 5
| Model                                   | Average Recall | Average Precision | Average NDCG | Average F1 |
|-----------------------------------------|----------------------|------------------------|-------------------|-----------------|
| **nlpai-lab/KURE-v1**                   | **0.73851**          | **0.19130**            | **0.67479**       | **0.29903**     |
| dragonkue/BGE-m3-ko                     | 0.72517              | 0.18799                | 0.66692           | 0.29401         |
| BAAI/bge-m3                             | 0.72954              | 0.18975                | 0.66615           | 0.29632         |
| Snowflake/snowflake-arctic-embed-l-v2.0 | 0.72962              | 0.18875                | 0.66236           | 0.29542         |
| nlpai-lab/KoE5                          | 0.70820              | 0.18287                | 0.64499           | 0.28628         |
| intfloat/multilingual-e5-large          | 0.70124              | 0.18316                | 0.64402           | 0.28588         |
| BAAI/bge-multilingual-gemma2            | 0.70258              | 0.18556                | 0.63338           | 0.28851         |
| jinaai/jina-embeddings-v3               | 0.69933              | 0.18256                | 0.63133           | 0.28505         |
| intfloat/multilingual-e5-large-instruct | 0.69018              | 0.17838                | 0.62486           | 0.27933         |
| Alibaba-NLP/gte-multilingual-base       | 0.69365              | 0.17789                | 0.61896           | 0.27879         |
| intfloat/multilingual-e5-base           | 0.67250              | 0.17406                | 0.61119           | 0.27247         |
| Alibaba-NLP/gte-Qwen2-7B-instruct       | 0.67447              | 0.17114                | 0.60952           | 0.26943         |
| intfloat/e5-mistral-7b-instruct         | 0.67449              | 0.17484                | 0.60935           | 0.27349         |
| openai/text-embedding-3-large           | 0.66365              | 0.17004                | 0.59389           | 0.26677         |
| Salesforce/SFR-Embedding-2_R            | 0.65622              | 0.17018                | 0.58494           | 0.26612         |
| upskyy/bge-m3-korean                    | 0.65477              | 0.17015                | 0.58073           | 0.26589         |
| jhgan/ko-sroberta-multitask             | 0.53136              | 0.13264                | 0.45879           | 0.20976         |

### Top-k 10
| Model                                   | Average Recall | Average Precision | Average NDCG | Average F1 |
|-----------------------------------------|----------------------|------------------------|-------------------|-----------------|
| **nlpai-lab/KURE-v1**                   | **0.79682**          | **0.10624**            | **0.69473**       | **0.18524**     |
| dragonkue/BGE-m3-ko                     | 0.78450              | 0.10492                | 0.68748           | 0.18288         |
| BAAI/bge-m3                             | 0.79195              | 0.10592                | 0.68723           | 0.18456         |
| Snowflake/snowflake-arctic-embed-l-v2.0 | 0.78669              | 0.10462                | 0.68189           | 0.18260         |
| intfloat/multilingual-e5-large          | 0.75902              | 0.10147                | 0.66370           | 0.17693         |
| nlpai-lab/KoE5                          | 0.75296              | 0.09937                | 0.66012           | 0.17369         |
| BAAI/bge-multilingual-gemma2            | 0.76153              | 0.10364                | 0.65330           | 0.18003         |
| jinaai/jina-embeddings-v3               | 0.76277              | 0.10240                | 0.65290           | 0.17843         |
| intfloat/multilingual-e5-large-instruct | 0.74851              | 0.09888                | 0.64451           | 0.17283         |
| Alibaba-NLP/gte-multilingual-base       | 0.75631              | 0.09938                | 0.64025           | 0.17363         |
| Alibaba-NLP/gte-Qwen2-7B-instruct       | 0.74092              | 0.09607                | 0.63258           | 0.16847         |
| intfloat/multilingual-e5-base           | 0.73512              | 0.09717                | 0.63216           | 0.16977         |
| intfloat/e5-mistral-7b-instruct         | 0.73795              | 0.09777                | 0.63076           | 0.17078         |
| openai/text-embedding-3-large           | 0.72946              | 0.09571                | 0.61670           | 0.16739         |
| Salesforce/SFR-Embedding-2_R            | 0.71662              | 0.09546                | 0.60589           | 0.16651         |
| upskyy/bge-m3-korean                    | 0.71895              | 0.09583                | 0.60258           | 0.16712         |
| jhgan/ko-sroberta-multitask             | 0.61225              | 0.07826                | 0.48687           | 0.13757         |
<br/>

## Training Details
- KURE-v1 is a model fine-tuned from [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3).
- KoE5 is a model fine-tuned from [intfloat/multilingual-e5-large](https://huggingface.co/intfloat/multilingual-e5-large).

### Training Data
**KURE-v1**
- Korean query-document-hard_negative(5) pairs
- Approx. 2,000,000 examples

**KoE5**
- [ko-triplet-v1.0](https://huggingface.co/datasets/nlpai-lab/ko-triplet-v1.0)
- Korean query-document-hard_negative(1) pairs (open data)
- Approx. 700,000+ examples

### Training Procedure
**KURE-v1**
- loss: [CachedGISTEmbedLoss](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#cachedgistembedloss)
- batch size: 4096
- learning rate: 2e-05
- epochs: 1

**KoE5**
- loss: [CachedMultipleNegativesRankingLoss](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#cachedmultiplenegativesrankingloss)
- batch size: 512
- learning rate: 1e-05
- epochs: 1

<br/>

## Important Notes
- When using KoE5, you must add a prefix for each input: (e.g., `query: {query}`, `passage: {document}`)
  
## License
- ```MIT```

## Citation
If you find our paper or models helpful, please consider citing them as follows:
```text
@misc{KURE,
  publisher = {Youngjoon Jang, Junyoung Son, Taemin Lee},
  year = {2024},
  url = {https://github.com/nlpai-lab/KURE}
},

@misc{KoE5,
  author = {NLP & AI Lab and Human-Inspired AI research},
  title = {KoE5: A New Dataset and Model for Improving Korean Embedding Performance},
  year = {2024},
  publisher = {Youngjoon Jang, Junyoung Son, Taemin Lee},
  journal = {GitHub repository},
  howpublished = {\url{https://drive.google.com/file/d/1wB02XGFH5v18iJYSYB0oJkWFYxH0ftoJ/view}},
}
```
```
