"""Benchmarking all datasets constituting the MTEB Korean leaderboard & average scores"""
from __future__ import annotations

import os
import logging
import argparse

import torch

from sentence_transformers import SentenceTransformer
from sentence_transformers.models import StaticEmbedding

import mteb
from mteb import MTEB, get_tasks
from dotenv import load_dotenv

from mteb.encoder_interface import PromptType
from mteb.models.sentence_transformer_wrapper import SentenceTransformerWrapper
from mteb.models.instruct_wrapper import instruct_wrapper

load_dotenv() # for OPENAI

parser = argparse.ArgumentParser(description="Extract contexts")
parser.add_argument('--quantize', default=False, type=bool, help='quantize embeddings')
args = parser.parse_args()

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger("main")

TASK_LIST_CLASSIFICATION = []

TASK_LIST_CLUSTERING = []

TASK_LIST_PAIR_CLASSIFICATION = []

TASK_LIST_RERANKING = []

TASK_LIST_RETRIEVAL = [
    "Ko-StrategyQA",
    "AutoRAGRetrieval",
    "MIRACLRetrieval", # 시간이 오래 걸림 주의
    "PublicHealthQA",
    "BelebeleRetrieval",
    "MrTidyRetrieval", # 시간이 오래 걸림 주의
    "MultiLongDocRetrieval",
    "XPQARetrieval",
    "Tatoeba"
]

TASK_LIST_STS = []

TASK_LIST = (
    TASK_LIST_CLASSIFICATION
    + TASK_LIST_CLUSTERING
    + TASK_LIST_PAIR_CLASSIFICATION
    + TASK_LIST_RERANKING
    + TASK_LIST_RETRIEVAL
    + TASK_LIST_STS
)

model_names = [
    # my_model_directory
]
model_names = [
    "Salesforce/SFR-Embedding-2_R", # 4096
    "Alibaba-NLP/gte-Qwen2-7B-instruct", # 8192
    "BAAI/bge-multilingual-gemma2", # 8192
    "intfloat/e5-mistral-7b-instruct", # 32768
    "intfloat/multilingual-e5-large-instruct", # 512
    "openai/text-embedding-3-large", # 8191
    "Alibaba-NLP/gte-multilingual-base", 
    "intfloat/multilingual-e5-base", # 512
    "intfloat/multilingual-e5-large", # 512
    "jinaai/jina-embeddings-v3", # 8192
    "jhgan/ko-sroberta-multitask", # 128
    "BAAI/bge-m3", # 8192
    "nlpai-lab/KoE5", # 512
    "dragonkue/BGE-m3-ko", # 8192
    "Snowflake/snowflake-arctic-embed-l-v2.0", # 8192,
    "nlpai-lab/KURE-v1" # 8192
] + model_names

def evaluate_model(model_name):
    try:
        model = None
        if not os.path.exists(model_name): # hf에 등록된 모델의 경우
            if "m2v" in model_name: # model2vec 모델의 경우: 모델명에 m2v를 포함시켜주어야 model2vec 모델로 인식합니다.
                static_embedding = StaticEmbedding.from_model2vec(model_name)
                model = SentenceTransformer(modules=[static_embedding])
            else:
                if model_name == "nlpai-lab/KoE5":
                    # mE5 기반의 모델이므로, 해당 프롬프트를 추가시킵니다.
                    model_prompts = {
                        PromptType.query.value: "query: ",
                        PromptType.passage.value: "passage: ",
                    }
                    model = SentenceTransformerWrapper(model=model_name, model_prompts=model_prompts)
                elif model_name == "BAAI/bge-multilingual-gemma2":
                    # mbge-gemma2의 경우, mteb에서 지원하지 않습니다. 따라서, instruct_wrapper를 사용합니다.
                    instruction_template = '<instruct>{instruction}\n<query>'
                    model = instruct_wrapper(
                            model_name_or_path=model_name,
                            instruction_template=instruction_template,
                            attn="cccc",
                            pooling_method="lasttoken",
                            mode="embedding",
                            torch_dtype=torch.float16,
                            normalized=True,
                    )
                elif model_name == "Snowflake/snowflake-arctic-embed-l-v2.0":
                    # mteb에서 Snowflake 모델을 지원하지 않으므로, Snowflake에서 사용하는 "query: " prefix를 임의로 추가합니다.
                    model_prompts = {
                        PromptType.query.value: "query: ",
                    }
                    model = SentenceTransformerWrapper(model=model_name, model_prompts=model_prompts)
                else:
                    # mteb에 등록된 모델의 경우, 프롬프트/prefix 등을 포함하여 평가할 수 있습니다. 등록되지 않은 경우, sentence-transformer를 사용하여 불러옵니다.
                    model = mteb.get_model(model_name)
        else: # 직접 학습한 모델의 경우
            file_name = os.path.join(model_name, "model.safetensors")
            if os.path.exists(file_name):
                if "m2v" in model_name: # model2vec의 경우: 모델명에 m2v를 포함시켜주어야 model2vec 모델로 인식합니다.
                    static_embedding = StaticEmbedding.from_model2vec(model_name)
                    model = SentenceTransformer(modules=[static_embedding])
                else:
                    model = mteb.get_model(model_name)

        if model:
            print(f"Running task: {TASK_LIST} / {model_name}")
            evaluation = MTEB(
                tasks=get_tasks(tasks=TASK_LIST, languages=["kor-Kore", "kor-Hang", "kor_Hang"])
            )

            # 48GB VRAM 기준 적합한 batch sizes
            if "multilingual-e5" in model_name or "KoE5" in model_name:
                batch_size = 512
            elif "jina" in model_name:
                batch_size = 8
            elif "bge-m3" in model_name or "KURE" in model_name:
                batch_size = 32
            elif "gemma2" in model_name:
                batch_size = 256 
            elif "Salesforce" in model_name or "Snowflake" in model_name:
                batch_size = 128
            else:
                batch_size = 64
              
            evaluation.run(
                model,
                output_folder=f"results/{model_name}",
                encode_kwargs={"batch_size": batch_size},
            )
    except Exception as ex:
        print(ex)

if __name__ == "__main__":
    for model_name in model_names:
        evaluate_model(model_name)
