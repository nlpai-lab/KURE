#!/bin/bash
# KURE Evaluation Pipeline
#
# 특징:
# - GPU 0이 모델1 끝나면 바로 모델2 시작 (다른 GPU 기다리지 않음)
# - OOM 발생 시 배치 사이즈 자동 절반으로 재시도 (최대 3회)
# - 완료된 태스크 자동 스킵

MODELS=(
    # "BAAI/bge-m3"
	# "Alibaba-NLP/gte-multilingual-base"
	# "google/embeddinggemma-300m"
    # "nlpai-lab/KURE-v1"
    # "jinaai/jina-embeddings-v3"
	# "dragonkue/snowflake-arctic-embed-l-v2.0-ko"
	# "dragonkue/BGE-m3-ko"
	# "telepix/PIXIE-Rune-v1.0"
	# "intfloat/multilingual-e5-large"
	# "intfloat/multilingual-e5-large-instruct"
	# "intfloat/e5-mistral-7b-instruct"
	# "SamilPwC-AXNode-GenAI/PwC-Embedding_expr"
	# "Snowflake/snowflake-arctic-embed-l-v2.0"
    # "Qwen/Qwen3-Embedding-0.6B"
	# "Qwen/Qwen3-Embedding-4B"
	# "Qwen/Qwen3-Embedding-8B"
	# "nvidia/llama-nemotron-embed-1b-v2"
	# "nvidia/llama-embed-nemotron-8b"
	# "Alibaba-NLP/gte-Qwen2-1.5B-instruct"
	# "Alibaba-NLP/gte-Qwen2-7B-instruct"
	# "jinaai/jina-embeddings-v5-text-small"
	# "jinaai/jina-embeddings-v5-text-nano"
	# "microsoft/harrier-oss-v1-270m"
	# "microsoft/harrier-oss-v1-0.6b"
	# "microsoft/harrier-oss-v1-27b"
	# "perplexity-ai/pplx-embed-v1-0.6b"
	"perplexity-ai/pplx-embed-v1-4b"
)

# GPU 설정 (필요시 수정)
export CUDA_VISIBLE_DEVICES=6,7

echo "=========================================="
echo "KURE Pipeline Evaluation"
echo "Models: ${MODELS[@]}"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "=========================================="

# 파이프라인 실행 (GPU별 독립 큐, GPU 자동 감지)
uv run python eval/run_pipeline.py \
    --models "${MODELS[@]}" \
    --output_dir eval/results

echo "=========================================="
echo "Evaluation Complete"
echo "=========================================="

# 사용 예시:
#
# 1. 기본 실행:
#    ./eval/run.sh
#
# 2. MRL 차원 지정:
#    uv run python eval/run_pipeline.py --models jinaai/jina-embeddings-v3:256 jinaai/jina-embeddings-v3:512
#
# 3. 특정 GPU만 사용:
#    uv run python eval/run_pipeline.py --models BAAI/bge-m3 --gpus 0 1
#
# 4. 완료된 태스크 재평가:
#    uv run python eval/run_pipeline.py --models BAAI/bge-m3 --no_skip
