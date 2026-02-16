"""Benchmarking all datasets constituting the MTEB Korean leaderboard & average scores"""
from __future__ import annotations

import os
import logging
from multiprocessing import Process, current_process
import torch
import hashlib

from sentence_transformers import SentenceTransformer
from sentence_transformers.models import StaticEmbedding

import mteb
from mteb import MTEB, get_tasks
from mteb.encoder_interface import PromptType
from mteb.models.sentence_transformer_wrapper import SentenceTransformerWrapper
from mteb.models.instruct_wrapper import instruct_wrapper

import argparse
from dotenv import load_dotenv
from setproctitle import setproctitle
import traceback
import logging

# load_dotenv() # for OPENAI

parser = argparse.ArgumentParser(description="Extract contexts")
parser.add_argument('--quantize', default=False, type=bool, help='quantize embeddings')
args = parser.parse_args()

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger("main")

# MIRACL, MrTidy는 평가 시 시간이 오래 걸리기 때문에, 태스크별로 나누어 multiprocessing으로 평가합니다.
# 필요 시 GPU 번호를 다르게 조정해 주세요.

TASK_LIST_RETRIEVAL_GPU_MAPPING = {
	0: [
		"BelebeleRetrieval",
		"XPQARetrieval",
		"MultiLongDocRetrieval",
		"Ko-StrategyQA",
		"AutoRAGRetrieval",
		"PublicHealthQA",
	],
	1: ["MIRACLRetrieval"],
	2: ["MrTidyRetrieval"],
}

model_names = [
	# my_model_directory
]
model_names = [
	# "Salesforce/SFR-Embedding-2_R", # 4096
	# "Alibaba-NLP/gte-Qwen2-7B-instruct", # 8192
	# "intfloat/e5-mistral-7b-instruct", # 32768
	# "intfloat/multilingual-e5-large-instruct", # 512
	# "openai/text-embedding-3-large", # 8191
	# "Alibaba-NLP/gte-multilingual-base", # 8192
	# "upskyy/bge-m3-korean", #8192
	# "intfloat/multilingual-e5-base", # 512
	# "intfloat/multilingual-e5-large", # 512
	# "jhgan/ko-sroberta-multitask", # 128
	# "BAAI/bge-multilingual-gemma2", # 8192
	# "BAAI/bge-m3", # 8192
	# "nlpai-lab/KoE5", # 512
	# "jinaai/jina-embeddings-v3", # 8192
	# "nomic-ai/nomic-embed-text-v2-moe", # 512
	# "dragonkue/BGE-m3-ko", # 8192
	# "Snowflake/snowflake-arctic-embed-l-v2.0", # 8192,
	# "nlpai-lab/KURE-v1", # 8192,
	# "dragonkue/snowflake-arctic-embed-l-v2.0-ko", # 8192
	# "Qwen/Qwen3-Embedding-0.6B", # 32768
	# "Qwen/Qwen3-Embedding-4B", # 32768
	# "Qwen/Qwen3-Embedding-8B", # 32768
	# "FronyAI/frony-embed-medium-arctic-ko-v2.5", # 8192
	# "telepix/PIXIE-Spell-Preview-0.6B", # 32768
	# "telepix/PIXIE-Rune-Preview", # 8192 
	# "telepix/PIXIE-Spell-Preview-1.7B", # 32768
	# "google/embeddinggemma-300m", # 2048
	# "SamilPwC-AXNode-GenAI/PwC-Embedding_expr" # 512
] + model_names

save_path = "./RESULTS_DEV"

def evaluate_model(model_name, gpu_id, tasks):
	import torch
	try:
		device = torch.device(f"cuda:{str(gpu_id)}") 
		torch.cuda.set_device(device)
		os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

		model = None
		if not os.path.exists(model_name): # hf에 등록된 모델의 경우
			if "m2v" in model_name: # model2vec의 경우: 모델명에 m2v를 포함시켜주어야 model2vec 모델로 인식합니다.
				static_embedding = StaticEmbedding.from_model2vec(model_name)
				model = SentenceTransformer(modules=[static_embedding], device=device)
			else:
				if model_name == "nlpai-lab/KoE5" or "KU-HIAI-ONTHEIT" in model_name:
					# mE5 기반의 모델이므로, 해당 프롬프트를 추가시킵니다.
					model_prompts = {
						PromptType.query.value: "query: ",
						PromptType.document.value: "passage: ",
					}
					model = SentenceTransformerWrapper(model=model_name, model_prompts=model_prompts, device=device)
				elif "snowflake" in model_name.lower() or "pixie" in model_name.lower():
					model_prompts = {
						PromptType.query.value: "query: ",
					}
					model = SentenceTransformerWrapper(model=model_name, model_prompts=model_prompts, device=device)
				elif "frony" in model_name:
					model_prompts = {
						PromptType.query.value: "<Q>",
						PromptType.document.value: "<P>",
					}
					model = SentenceTransformerWrapper(model_name, model_prompts=model_prompts, device=device)
				else:
					# mteb에 등록된 모델의 경우, 프롬프트/prefix 등을 포함하여 평가할 수 있습니다. 등록되지 않은 경우, sentence-transformers를 사용하여 불러옵니다.
					model = mteb.get_model(model_name, device=device)
		else: # 직접 학습한 모델의 경우
			file_name = os.path.join(model_name, "model.safetensors")
			if os.path.exists(file_name):
				if "m2v" in model_name:
					static_embedding = StaticEmbedding.from_model2vec(model_name)
					model = SentenceTransformer(modules=[static_embedding], device=device)
				else:
					model = mteb.get_model(model_name, device=device)

		if model:
			output_folder_name = os.path.basename(model_name)
			if os.path.isdir(model_name) and len(output_folder_name) > 100:
				model_hash = hashlib.md5(model_name.encode()).hexdigest()[:6]
				output_folder_name = f"{output_folder_name[:93]}_{model_hash}"

			if os.path.isdir(model_name):
				try:
					model.model_meta.name = output_folder_name
				except AttributeError:
					logger.warning("Could not override model_meta.name. Path might still be too long.")
			
			setproctitle(f"{output_folder_name}-{gpu_id}")
			print(f"Running tasks: {tasks} / {model_name} on GPU {gpu_id} in process {current_process().name}")
			evaluation = MTEB(
				tasks=get_tasks(tasks=tasks, languages=["kor-Kore", "kor-Hang", "kor_Hang"])
			)
			# 48GB VRAM 기준 적합한 batch sizes
			if "multilingual-e5" in model_name or "KoE5" in model_name or "ontheit" in model_name:
				batch_size = 512
			elif "jina" in model_name:
				batch_size = 8
			elif "bge-m3" in model_name or "Snowflake" in model_name:
				batch_size = 64
			elif "gemma2" in model_name:
				batch_size = 256 
			elif "Salesforce" in model_name:
				batch_size = 8
			elif "embeddinggemma" in model_name:
				batch_size = 256
			else:
				batch_size = 64

			if args.quantize:
				evaluation.run(
					model,
					output_folder=f"{save_path}/{output_folder_name}-quantized",
					encode_kwargs={"batch_size": batch_size, "precision": "binary"},
				)
			else:
				evaluation.run(
					model,
					output_folder=f"{save_path}/{output_folder_name}",
					encode_kwargs={"batch_size": batch_size},
				)
	except Exception as ex:
		print(ex)
		traceback.print_exc()

if __name__ == "__main__":
	torch.multiprocessing.set_start_method('spawn')
	
	for model_name in model_names:
		print(f"Starting evaluation for model: {model_name}")
		processes = []
		
		for gpu_id, tasks in TASK_LIST_RETRIEVAL_GPU_MAPPING.items():
			p = Process(target=evaluate_model, args=(model_name, gpu_id, tasks))
			p.start()
			processes.append(p)
		
		for p in processes:
			p.join()
		
		print(f"Completed evaluation for model: {model_name}")