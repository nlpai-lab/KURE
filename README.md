# 🔎 KURE: Korea University Retrieval Embedding model

## Update Logs
- 2024.12.21: [🤗 KURE-v1](https://huggingface.co/nlpai-lab/KURE-v1), [MTEB-ko-retrieval Leaderboard](https://github.com/nlpai-lab/KURE?tab=readme-ov-file#mteb-ko-retrieval-leaderboard) 공개
- 2024.10.02: [🤗 KoE5](https://huggingface.co/nlpai-lab/KoE5), [🤗 ko-triplet-v1.0](https://huggingface.co/datasets/nlpai-lab/ko-triplet-v1.0) 공개

---

<br>

KURE는 고려대학교 [NLP & AI 연구실](http://nlp.korea.ac.kr/)과 [HIAI 연구소](http://hiai.korea.ac.kr)가 개발한 한국어 특화 임베딩 모델입니다.

KURE를 공개합니다.  
<br/>

## KURE 모델 실행 코드
### sentence-transformers로 실행
```bash
pip install sentence-transformers
```

아래 예제 코드로 실행해볼 수 있습니다.

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
[MTEB](https://github.com/embeddings-benchmark/mteb)에 등록된 모든 Korean Retrieval Benchmark에 대한 평가를 진행하였습니다.
### Korean Retrieval Benchmark
| Dataset                                                               | Description                                                                                             | Average Length (characters) |
|-----------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------|-----------------------------|
| [Ko-StrategyQA](https://huggingface.co/datasets/taeminlee/Ko-StrategyQA) | 한국어 ODQA multi-hop 검색 데이터셋 (StrategyQA 번역 데이터셋)                                                   | 305.15                      |
| [AutoRAGRetrieval](https://huggingface.co/datasets/yjoonjang/markers_bm) | 금융, 공공, 의료, 법률, 커머스 5개 분야에 대해, pdf를 파싱하여 구성한 한국어 문서 검색 데이터셋         | 823.60                      |
| [MIRACLRetrieval](https://huggingface.co/datasets/miracl/miracl)         | Wikipedia 기반의 한국어 문서 검색 데이터셋                                                              | 166.63                      |
| [PublicHealthQA](https://huggingface.co/datasets/xhluca/publichealth-qa)   | 의료 및 공중보건 도메인에 대한 한국어 문서 검색 데이터셋                                                  | 339.00                      |
| [BelebeleRetrieval](https://huggingface.co/datasets/facebook/belebele)   | FLORES-200 기반의 한국어 문서 검색 데이터셋                                                             | 243.11                      |
| [MrTidyRetrieval](https://huggingface.co/datasets/mteb/mrtidy)           | Wikipedia 기반의 한국어 문서 검색 데이터셋                                                              | 166.90                      |
| [MultiLongDocRetrieval](https://huggingface.co/datasets/Shitao/MLDR)   | 다양한 도메인의 한국어 장문 검색 데이터셋                                                               | 13,813.44                   |
<!-- - [XPQARetrieval](https://huggingface.co/datasets/jinaai/xpqa): 다양한 도메인의 한국어 문서 검색 데이터셋 -->

<details>
<summary>XPQARetrieval 데이터셋 제외 이유</summary>

- 저희 평가에서는 [XPQARetrieval](https://huggingface.co/datasets/jinaai/xpqa) 데이터셋을 제외하여 평가하였습니다. XPQA는 Cross-Lingual QA 능력을 평가하기 위한 데이터셋으로, 질의를 기반으로 근거 문서를 찾아야 하는 검색 태스크 평가에 사용하기에는 부적절하다고 판단하였습니다.
- XPQARetrieval 데이터셋의 예시는 다음과 같습니다.
```json
{
	"query": "미개봉인가요?",
	"document": "아니요. 리뉴얼된 제품입니다."
},
{
	"query": "아이패드에어 3와 호환이 가능합니까?",
	"document": "네, 가능합니다."
}
```

</details>

### Evaluation code
`evaluate.py`에 모델을 추가하여 mteb를 활용한 평가를 진행할 수 있습니다.
```bash
cd eval
pip install -r requirements.txt
python evaluate.py
```

### Leaderboard
streamlit을 통해 모든 모델의 모든 태스크에 대한 평가 결과를 시각화합니다.
```bash
streamlit run leaderboard.py
```

### Average Results
아래는 모든 모델의, 모든 벤치마크 데이터셋에 대한 평균 결과입니다.
자세한 결과는 `eval/results`폴더에서 확인하실 수 있습니다.
| Model                                         | Parameters | Average Recall@10 | Average Precision@10 | Average NDCG@10 | Average F1@10 |
|-----------------------------------------------|------------|----------------|-------------------|--------------|------------|
| Qwen/Qwen3-Embedding-8B                   | 8B     | 0.86157    | 0.11302       | 0.76349  | 0.19715|
| Qwen/Qwen3-Embedding-4B                       | 4B         | 0.85261        | 0.11111           | 0.74844      | 0.19412    |
| telepix/PIXIE-Rune-Preview                    | 0.6B       | 0.83771        | 0.10890           | 0.74201      | 0.19041    |
| **nlpai-lab/KURE-v1**                             | **0.6B**       | **0.83997**        | **0.11020**           | **0.73947**      | **0.19232**    |
| dragonkue/snowflake-arctic-embed-l-v2.0-ko    | 0.6B       | 0.83460        | 0.10850           | 0.73855      | 0.18973    |
| telepix/PIXIE-Spell-Preview-1.7B              | 1.7B       | 0.83403        | 0.10681           | 0.73420      | 0.18730    |
| BAAI/bge-m3                                   | 0.6B       | 0.83988        | 0.11057           | 0.73388      | 0.19286    |
| dragonkue/BGE-m3-ko                           | 0.6B       | 0.82568        | 0.10862           | 0.73122      | 0.18952    |
| Snowflake/snowflake-arctic-embed-l-v2.0       | 0.6B       | 0.82096        | 0.10678           | 0.71785      | 0.18672    |
| telepix/PIXIE-Spell-Preview-0.6B              | 0.6B       | 0.80978        | 0.10314           | 0.71058      | 0.18106    |
| intfloat/multilingual-e5-large                | 0.6B       | 0.80120        | 0.10528           | 0.70750      | 0.18380    |
| FronyAI/frony-embed-medium-arctic-ko-v2.5     | 0.6B       | 0.81407        | 0.10461           | 0.70672      | 0.18327    |
| nlpai-lab/KoE5                                | 0.6B       | 0.79659        | 0.10339           | 0.70430      | 0.18094    |
| google/embeddinggemma-300m                    | 0.3B       | 0.80696        | 0.10673           | 0.69438      | 0.18600    |
| BAAI/bge-multilingual-gemma2                  | 9.4B       | 0.80229        | 0.10735           | 0.69314      | 0.18667    |
| Qwen/Qwen3-Embedding-0.6B                     | 0.6B       | 0.79388        | 0.10120           | 0.68950      | 0.17753    |
| Alibaba-NLP/gte-multilingual-base             | 0.3B       | 0.80663        | 0.10421           | 0.68786      | 0.18231    |
| jinaai/jina-embeddings-v3                     | 0.6B       | 0.79560        | 0.10462           | 0.68721      | 0.18258    |
| SamilPwC-AXNode-GenAI/PwC-Embedding_expr      | 0.6B       | 0.78495        | 0.10347           | 0.68462      | 0.18049    |
| nomic-ai/nomic-embed-text-v2-moe              | 0.5B       | 0.77598        | 0.10220           | 0.67987      | 0.17828    |
| intfloat/multilingual-e5-large-instruct       | 0.6B       | 0.78293        | 0.10138           | 0.67985      | 0.17749    |
| intfloat/multilingual-e5-base                 | 0.3B       | 0.77666        | 0.10085           | 0.67094      | 0.17644    |
| Alibaba-NLP/gte-Qwen2-7B-instruct             | 7.6B       | 0.77758        | 0.09880           | 0.66886      | 0.17357    |
| intfloat/e5-mistral-7b-instruct               | 7.1B       | 0.77057        | 0.10010           | 0.66493      | 0.17510    |
| openai/text-embedding-3-large                 | Unkown       | 0.76420        | 0.09828           | 0.65134      | 0.17217    |
| upskyy/bge-m3-korean                          | 0.6B       | 0.76324        | 0.10026           | 0.64339      | 0.17500    |
| Salesforce/SFR-Embedding-2_R                  | 2.6B       | 0.74958        | 0.09797           | 0.63906      | 0.17113    |
| jhgan/ko-sroberta-multitask                   | 0.1B       | 0.64784        | 0.08123           | 0.51648      | 0.14304    |
<br/>

### MultiLongDoc Results
아래는 평균적으로 길이가 긴 문서들인 MultiLongDocRetrieval 데이터셋에 대한 평가 결과입니다.
| Model                                         | Parameters | Average Recall@10 | Average Precision@10 | Average NDCG@10 | Average F1@10 |
|-----------------------------------------------|------------|-------------------|----------------------|-----------------|---------------|
| Qwen/Qwen3-Embedding-8B                   | 8B     | 0.65250       | 0.06525          | 0.51027     | 0.11864   |
| Qwen/Qwen3-Embedding-4B                       | 4B         | 0.61000           | 0.06100              | 0.48661         | 0.11091       |
| Alibaba-NLP/gte-multilingual-base             | 0.3B       | 0.61250           | 0.06125              | 0.47568         | 0.11136       |
| telepix/PIXIE-Spell-Preview-1.7B              | 1.7B       | 0.58750           | 0.05875              | 0.47479         | 0.10682       |
| **nlpai-lab/KURE-v1**                             | **0.6B**       | **0.58000**           | **0.05800**              | **0.46369**         | **0.10545**       |
| telepix/PIXIE-Rune-Preview                    | 0.6B       | 0.56250           | 0.05625              | 0.43975         | 0.10227       |
| dragonkue/snowflake-arctic-embed-l-v2.0-ko    | 0.6B       | 0.55250           | 0.05525              | 0.43045         | 0.10045       |
| BAAI/bge-m3                                   | 0.6B       | 0.55500           | 0.05550              | 0.42870         | 0.10091       |
| Qwen/Qwen3-Embedding-0.6B                     | 0.6B       | 0.53000           | 0.05300              | 0.42282         | 0.09636       |
| telepix/PIXIE-Spell-Preview-0.6B              | 0.6B       | 0.53500           | 0.05350              | 0.42105         | 0.09727       |
| dragonkue/BGE-m3-ko                           | 0.6B       | 0.48250           | 0.04825              | 0.38987         | 0.08773       |
| FronyAI/frony-embed-medium-arctic-ko-v2.5     | 0.6B       | 0.51250           | 0.05125              | 0.38870         | 0.09318       |
| Snowflake/snowflake-arctic-embed-l-v2.0       | 0.6B       | 0.51500           | 0.05150              | 0.38639         | 0.09364       |
| google/embeddinggemma-300m                    | 0.3B       | 0.40750           | 0.04075              | 0.31604         | 0.07409       |
| jinaai/jina-embeddings-v3                     | 0.6B       | 0.42000           | 0.04200              | 0.31399         | 0.07636       |
| nlpai-lab/KoE5                                | 0.6B       | 0.40000           | 0.04000              | 0.30146         | 0.07273       |
| Alibaba-NLP/gte-Qwen2-7B-instruct             | 7.6B       | 0.38250           | 0.03825              | 0.29841         | 0.06955       |
| openai/text-embedding-3-large                 | Unkown     | 0.40250           | 0.04025              | 0.29794         | 0.07318       |
| BAAI/bge-multilingual-gemma2                  | 9.4B       | 0.40500           | 0.04050              | 0.29649         | 0.07364       |
| nomic-ai/nomic-embed-text-v2-moe              | 0.5B       | 0.35000           | 0.03500              | 0.27140         | 0.06364       |
| intfloat/multilingual-e5-large-instruct       | 0.6B       | 0.36250           | 0.03625              | 0.26746         | 0.06591       |
| SamilPwC-AXNode-GenAI/PwC-Embedding_expr      | 0.6B       | 0.34000           | 0.03400              | 0.26056         | 0.06182       |
| intfloat/e5-mistral-7b-instruct               | 7.1B       | 0.34250           | 0.03425              | 0.25993         | 0.06227       |
| intfloat/multilingual-e5-large                | 0.6B       | 0.36000           | 0.03600              | 0.25838         | 0.06545       |
| Salesforce/SFR-Embedding-2_R                  | 2.6B       | 0.33750           | 0.03375              | 0.25574         | 0.06136       |
| intfloat/multilingual-e5-base                 | 0.3B       | 0.31000           | 0.03100              | 0.23126         | 0.05636       |
| upskyy/bge-m3-korean                          | 0.6B       | 0.31000           | 0.03100              | 0.22378         | 0.05636       |
| jhgan/ko-sroberta-multitask                   | 0.1B       | 0.29500           | 0.02950              | 0.21082         | 0.05364       |
<br/>

### Conclusion
KURE-v1은 두 평가 결과에서 모두 상위권의 성능을 보입니다.
- **전체 평균:** 다양한 길이와 도메인의 데이터셋을 종합한 평가에서 비슷한 크기의 모델들 중 우수한 성능을 기록합니다.
- **장문(MultiLongDoc Results):** 평균 길이가 13,000자가 넘는 장문 데이터셋에서도 0.6B 크기의 모델들 중 가장 우수한 검색 능력을 보입니다.

이를 통해 KURE-v1은 문서 길이에 구애받지 않고 강건한 성능을 보이는 모델임을 확인할 수 있습니다.

## Training Details
- KURE-v1은 [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3)를 기반으로 fine-tuning된 모델입니다.
- KoE5는 [intfloat/multilingual-e5-large](https://huggingface.co/intfloat/multilingual-e5-large)를 기반으로 fine-tuning된 모델입니다.

### Training Data
**KURE-v1**
- 한국어 query-document-hard_negatives(5개) 데이터 쌍 
- 약 1,500,000 examples

**KoE5**
- [ko-triplet-v1.0](https://huggingface.co/datasets/nlpai-lab/ko-triplet-v1.0)
- 한국어 query-document-hard_negative(1개) 데이터 쌍 (open data)
- 약 700,000 examples

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

## 주의사항
- KoE5 사용 시, prefix를 붙여 주어야 합니다. (query: {query}, passage: {document})
  
## License
- ```MIT```

## Citation
If you find our paper or models helpful, please consider cite as follows:
```text
@misc{KURE,
  publisher = {Youngjoon Jang, Junyoung Son, Taemin Lee},
  year = {2024},
  url = {https://github.com/nlpai-lab/KURE}
},

@misc{KoE5,
  author = {NLP & AI Lab and Human-Inspired AI research},
  title = {KoE5: 한국어 임베딩 성능 향상을 위한 새로운 데이터셋 및 모델},
  year = {2024},
  publisher = {Youngjoon Jang, Junyoung Son, Taemin Lee},
  journal = {GitHub repository},
  howpublished = {\url{https://drive.google.com/file/d/1wB02XGFH5v18iJYSYB0oJkWFYxH0ftoJ/view}},
}
```
