# KURE-v2 서빙

[English](README.md) | [한국어](README_ko.md)

[KURE-v2 모델 카드](https://huggingface.co/nlpai-lab/KURE-v2)의 서빙 구성을 그대로 실행해볼 수 있는 코드입니다. 실행 방법과 백엔드별 옵션은 [README.md](README.md#run)를 참고하세요.

## 벤치마크 결과

KURE-v2는 후기 상호작용 모델로, 문서를 토큰 벡터의 집합으로 저장하기 때문에 배포에서는 색인 크기와 검색 비용이 실질적인 관건입니다. 9개 한국어 MTEB 검색 태스크에서 KURE-v2를 여러 ANN 백엔드·압축 기법으로, [faiss HNSW](https://faiss.ai/cpp_api/struct/structfaiss_1_1IndexHNSW.html)로 서빙되는 단일 벡터 5종과 함께 벤치마크했습니다. 모든 수치는 종단간(배치 1 질의 인코딩 + 색인 검색)이며 A100 80GB 1대에서 직렬로 측정했습니다.

<p align="center">
  <img src="../../assets/deploy_overview.png" width="100%" alt="색인 크기 대비(좌) / 종단간 QPS 대비(우) 평균 nDCG@10">
</p>

그림이 보여주는 두 가지:

- Hierarchical Token Pooling(x2)은 nDCG 손실 없이 색인을 절반으로 줄입니다(24.2 → 12.5 GB). Asymmetric Binary Quantization (문서 토큰 1-bit, 질의 bf16)는 0.98 하락으로 4.8배 축소합니다. 둘을 결합하면(pooling x3 + Asym. Binary) 9개 코퍼스 전체 색인이 **1.7 GB로, 모든 Single-Vector HNSW 색인(fp16 벡터, 7.0-25.4 GB)보다 작으면서도** 최고 Single-Vector 모델보다 높은 품질(79.57 대 79.10)을 유지합니다.
- 실제 query는 텍스트로 도착합니다: 4B-8B Single-Vector 모델은 query 인코딩에만 38-40 ms를 써서 색인이 아무리 빨라도 약 25 QPS에 고정됩니다. KURE-v2는 13.8 ms(154M)에 인코딩하므로 MUVERA를 제외한 모든 구성이 **44-57 QPS로, 8B Single-Vector 모델의 약 2배 처리량을 더 높은 품질로** 제공합니다.

### 대규모 문서 집합에서의 성능, 지연, 인덱스

<p align="center">
  <img src="../../assets/bigcorpus_miracl.png" width="70%" alt="MIRACL(150만 문서): 품질, 종단간 p95 지연, 색인 크기">
</p>

가장 큰 코퍼스(MIRACL, 약 150만 문서)에서 전수 1-bit 스캔의 비용은 문서 수에 비례합니다: p95가 156 ms까지 오르고, 토큰을 3배 줄여도 74 ms에 그칩니다. 같은 1-bit 색인 위에서 faiss [BinaryIVF](https://faiss.ai/cpp_api/struct/structfaiss_1_1IndexBinaryIVF.html) (해밍 검색)로 후보를 생성하고 정확한 비대칭 MaxSim으로 재순위화하면 **같은 2.2 GB 색인에서 p95가 38 ms로, 4B-8B 단일 벡터(42 ms)보다 짧은 꼬리 지연을 더 높은 nDCG로** 달성합니다. 대규모 컬렉션에서는 전수 스캔이 아니라 후보 생성형 색인(PLAID 또는 BinaryIVF)을 사용하세요.

<details>
<summary><b>측정 상세</b></summary>

- **하드웨어**: NVIDIA A100 80GB 1대, AMD EPYC 7513 2개(64코어), RAM 1.2 TB.
- **소프트웨어**: faiss-cpu 1.15.0, fast-plaid 1.6.0, sentence-transformers 6.0.0, PyTorch 2.8.0.
- **프로토콜**: 배치 1, 직렬. 색인 검색 지연: 워밍업 10회 후 태스크 전체 질의를 각 1회 측정(QPS = 평균 지연의 역수). 질의 인코딩 지연: 워밍업 5회 후 50회 측정. 종단간 = 인코딩 + 검색.
- **정밀도**: 인코딩은 bf16; 색인은 방법별 저장 형식(HNSW fp16 벡터, PLAID 4-bit 잔차, 이진화 1-bit).
- **색인 크기**: 디스크에 직렬화된 색인 전체(벡터·그래프·코드북 포함, 외부 문서 ID 매핑 제외). PLAID 색인은 freeze된 상태(fast-plaid `freeze()`)로, 검색용 merged 코드/잔차만 세고 샤드 빌드 복사본과 1,000문서 이하 코퍼스에서 fast-plaid가 보관하는 원본 임베딩은 제외합니다.
- **태스크**: 9개 한국어 MTEB 검색 태스크; MLDR은 dev/test 평균; nDCG@10 x100.
- **HNSW**: `IndexHNSWSQ`(fp16 저장 벡터, bf16 임베딩에 무손실; 정규화 임베딩의 내적), M=32, efConstruction=200, efSearch=64.
- **PLAID**: nbits=4, 나머지는 fast-plaid 기본값(kmeans_niters=4, n_ivf_probe=8, n_full_scores=4096). nbits=2/1은 14.2/9.1 GB, 81.40/80.70 nDCG.
- **MUVERA**: num_repetitions=10, num_simhash_projections=6, final_projection_dimension=8192, 상위 1,000개 정확 MaxSim 재순위화.
- **BinaryIVF**: nlist=floor(sqrt(총 토큰 수)), 상한 65,536, nprobe=32, 질의 토큰당 상위 128개 해밍 토큰, 상위 1,000개 문서 정확 비대칭 MaxSim 재순위화.
- **토큰 풀링**: 계층적(Ward linkage), pool_factor 2-3, 문서에만 적용.
</details>
