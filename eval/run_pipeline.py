#!/usr/bin/env python3
"""
Multi-model Pipeline Evaluation

각 GPU가 독립적으로 (모델, 태스크) 큐를 처리합니다.
- GPU 0이 모델1 태스크를 끝내면 바로 모델2 태스크 시작
- OOM 발생 시 배치 사이즈를 절반으로 줄여 재시도
- 최대 3번 재시도 (batch_size / 2 / 4 / 8)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import traceback
from dataclasses import dataclass
from multiprocessing import Process, Queue, Manager
from pathlib import Path
from typing import Any

import torch
from setproctitle import setproctitle

from mteb import MTEB

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.models.loader import load_model, get_output_folder
from eval.models.config import get_model_config
from eval.tasks.registry import get_all_tasks, MTEB_TASKS, NANOBEIR_KO_TASK_NAMES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("pipeline")

# Suppress noisy logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

# Import shared GPU task distribution from evaluate.py
from eval.evaluate import distribute_tasks


@dataclass
class EvalJob:
    """Evaluation job for a single (model, task) pair."""
    model_name: str
    task_name: str
    truncate_dim: int | None = None


def check_task_completed(output_dir: str, output_folder: str, task_name: str) -> bool:
    """Check if a task has already been completed."""
    result_path = Path(output_dir) / output_folder
    possible_files = [
        result_path / f"{task_name}.json",
        result_path / task_name / "results.json",
    ]
    for file_path in possible_files:
        if file_path.exists():
            try:
                with open(file_path) as f:
                    data = json.load(f)
                if data and ("scores" in data or "test" in data):
                    return True
            except (json.JSONDecodeError, KeyError):
                pass
    return False


def is_oom_error(error: Exception) -> bool:
    """Check if the error is an OOM error."""
    error_str = str(error).lower()
    return any(x in error_str for x in [
        "out of memory",
        "cuda out of memory",
        "oom",
        "cudamalloc failed",
        "failed to allocate",
    ])


def evaluate_single_task_with_retry(
    model: Any,
    task_name: str,
    output_dir: str,
    output_folder: str,
    initial_batch_size: int,
    gpu_id: int,
    max_retries: int = 10,
) -> tuple[bool, int]:
    """
    Evaluate a single task with OOM retry logic.

    Returns:
        Tuple of (success, final_batch_size)
    """
    batch_size = initial_batch_size

    for attempt in range(max_retries + 1):
        try:
            # Clear CUDA cache before each attempt
            torch.cuda.empty_cache()

            # Get task instance
            task_instances = get_all_tasks([task_name])
            if not task_instances:
                logger.warning(f"GPU {gpu_id}: Task '{task_name}' not found")
                return False, batch_size

            task = task_instances[0]

            # Run evaluation
            evaluation = MTEB(tasks=[task])
            evaluation.run(
                model,
                output_folder=os.path.join(output_dir, output_folder),
                encode_kwargs={"batch_size": batch_size},
            )

            return True, batch_size

        except Exception as e:
            if is_oom_error(e) and attempt < max_retries:
                old_batch = batch_size
                batch_size = max(1, batch_size // 2)
                logger.warning(
                    f"GPU {gpu_id}: OOM on '{task_name}' with batch_size={old_batch}, "
                    f"retrying with batch_size={batch_size} (attempt {attempt + 2}/{max_retries + 1})"
                )
                torch.cuda.empty_cache()
                continue
            else:
                logger.error(f"GPU {gpu_id}: Failed '{task_name}': {e}")
                traceback.print_exc()
                return False, batch_size

    return False, batch_size


def gpu_worker(
    gpu_id: int,
    job_queue: Queue,
    result_queue: Queue,
    output_dir: str,
    use_bf16: bool,
    use_flash_attn: bool,
    skip_completed: bool,
):
    """
    GPU worker that processes jobs from the queue.

    Each worker:
    1. Pulls (model, task) jobs from the shared queue
    2. Loads model if different from current
    3. Evaluates task with OOM retry
    4. Reports results
    """
    try:
        device = torch.device(f"cuda:{gpu_id}")
        torch.cuda.set_device(device)

        logger.info(f"GPU {gpu_id}: Worker started")

        current_model_name = None
        current_model = None
        current_batch_size = None
        current_dim = None

        while True:
            # Get next job from queue
            job = job_queue.get()

            # Poison pill to stop worker
            if job is None:
                logger.info(f"GPU {gpu_id}: Worker shutting down")
                break

            model_name = job.model_name
            task_name = job.task_name
            truncate_dim = job.truncate_dim

            output_folder = get_output_folder(model_name, truncate_dim)

            # Skip if already completed
            if skip_completed and check_task_completed(output_dir, output_folder, task_name):
                logger.info(f"GPU {gpu_id}: Skipping '{task_name}' for {model_name} (completed)")
                result_queue.put({
                    "gpu_id": gpu_id,
                    "model": model_name,
                    "task": task_name,
                    "status": "skipped",
                })
                continue

            # Load model if changed
            model_key = (model_name, truncate_dim)
            if current_model_name != model_key:
                logger.info(f"GPU {gpu_id}: Loading model {model_name}" +
                           (f" (dim={truncate_dim})" if truncate_dim else ""))

                # Clear old model
                if current_model is not None:
                    del current_model
                    torch.cuda.empty_cache()

                current_model, current_batch_size = load_model(
                    model_name,
                    device=device,
                    use_bf16=use_bf16,
                    use_flash_attn=use_flash_attn,
                    truncate_dim=truncate_dim,
                )
                current_model_name = model_key
                current_dim = truncate_dim

                logger.info(f"GPU {gpu_id}: Model loaded, batch_size={current_batch_size}")

            # Evaluate task with retry
            short_model = model_name.split("/")[-1]
            setproctitle(f"{short_model}--{task_name}")
            logger.info(f"GPU {gpu_id}: Processing '{task_name}' for {model_name}")
            start_time = time.time()

            success, final_batch = evaluate_single_task_with_retry(
                model=current_model,
                task_name=task_name,
                output_dir=output_dir,
                output_folder=output_folder,
                initial_batch_size=current_batch_size,
                gpu_id=gpu_id,
            )

            elapsed = time.time() - start_time

            # Update batch size if it was reduced
            if final_batch < current_batch_size:
                current_batch_size = final_batch
                logger.info(f"GPU {gpu_id}: Reduced batch_size to {current_batch_size} for future tasks")

            result_queue.put({
                "gpu_id": gpu_id,
                "model": model_name,
                "task": task_name,
                "status": "success" if success else "failed",
                "elapsed": elapsed,
                "batch_size": final_batch,
            })

            status = "done" if success else "FAILED"
            logger.info(f"GPU {gpu_id}: '{task_name}' {status} in {elapsed:.1f}s")

    except Exception as e:
        logger.error(f"GPU {gpu_id}: Worker crashed: {e}")
        traceback.print_exc()


def run_pipeline(
    models: list[str],
    gpu_ids: list[int] | None = None,
    output_dir: str = "eval/results",
    use_bf16: bool = True,
    use_flash_attn: bool = True,
    truncate_dims: dict[str, int] | None = None,
    skip_completed: bool = True,
):
    """
    Run multi-model pipeline evaluation.

    Args:
        models: List of model names to evaluate
        gpu_ids: List of GPU IDs to use
        output_dir: Output directory for results
        use_bf16: Whether to use bfloat16
        use_flash_attn: Whether to use flash attention
        truncate_dims: Dict mapping model names to truncate dimensions
        skip_completed: Whether to skip completed tasks
    """
    if gpu_ids is None:
        num_gpus = torch.cuda.device_count()
        gpu_ids = list(range(num_gpus))
        logger.info(f"Auto-detected {num_gpus} GPUs: {gpu_ids}")

    truncate_dims = truncate_dims or {}

    # Distribute tasks across GPUs
    task_distribution = distribute_tasks(gpu_ids)

    logger.info(f"Pipeline evaluation starting")
    logger.info(f"Models: {models}")
    logger.info(f"GPUs: {gpu_ids}")
    for gid, gtasks in task_distribution.items():
        logger.info(f"GPU {gid}: {len(gtasks)} tasks assigned")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Create job queues for each GPU
    manager = Manager()
    gpu_queues = {gpu_id: manager.Queue() for gpu_id in gpu_ids}
    result_queue = manager.Queue()

    # Distribute jobs to GPU queues
    # Each GPU gets its assigned tasks for ALL models
    total_jobs = 0
    for model_name in models:
        truncate_dim = truncate_dims.get(model_name)
        for gpu_id in gpu_ids:
            tasks = task_distribution.get(gpu_id, [])
            for task_name in tasks:
                job = EvalJob(
                    model_name=model_name,
                    task_name=task_name,
                    truncate_dim=truncate_dim,
                )
                gpu_queues[gpu_id].put(job)
                total_jobs += 1

    logger.info(f"Total jobs queued: {total_jobs}")

    # Add poison pills to stop workers
    for gpu_id in gpu_ids:
        gpu_queues[gpu_id].put(None)

    # Start workers
    workers = []
    for gpu_id in gpu_ids:
        p = Process(
            target=gpu_worker,
            args=(
                gpu_id,
                gpu_queues[gpu_id],
                result_queue,
                output_dir,
                use_bf16,
                use_flash_attn,
                skip_completed,
            ),
        )
        p.start()
        workers.append((gpu_id, p))
        logger.info(f"Started worker for GPU {gpu_id}")

    # Wait for all workers to complete
    for gpu_id, p in workers:
        p.join()
        logger.info(f"GPU {gpu_id} worker finished")

    # Collect results
    results = {"success": 0, "failed": 0, "skipped": 0}
    while not result_queue.empty():
        r = result_queue.get()
        results[r["status"]] = results.get(r["status"], 0) + 1

    logger.info(f"Pipeline completed: {results}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Multi-model pipeline evaluation with GPU parallelism and OOM retry",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate multiple models (GPU 0 starts model 2 as soon as it finishes model 1)
  python eval/run_pipeline.py --models BAAI/bge-m3 nlpai-lab/KoE5

  # With specific GPUs
  python eval/run_pipeline.py --models BAAI/bge-m3 --gpus 0 1

  # With MRL dimensions
  python eval/run_pipeline.py --models jinaai/jina-embeddings-v3:256 jinaai/jina-embeddings-v3:512
        """,
    )

    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        required=True,
        help="Models to evaluate. Use model:dim format for MRL (e.g., jinaai/jina-embeddings-v3:256)",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        nargs="+",
        default=None,
        help="GPU IDs to use (default: 0, 1, 2)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="eval/results",
        help="Output directory (default: eval/results)",
    )
    parser.add_argument(
        "--no_bf16",
        action="store_true",
        help="Disable bfloat16",
    )
    parser.add_argument(
        "--no_flash_attn",
        action="store_true",
        help="Disable flash attention",
    )
    parser.add_argument(
        "--no_skip",
        action="store_true",
        help="Don't skip completed tasks",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Parse models and dimensions
    models = []
    truncate_dims = {}

    for model_spec in args.models:
        if ":" in model_spec:
            model_name, dim_str = model_spec.rsplit(":", 1)
            truncate_dims[model_name] = int(dim_str)
        else:
            model_name = model_spec
        models.append(model_name)

    # Remove duplicates while preserving order
    models = list(dict.fromkeys(models))

    # Set multiprocessing start method
    torch.multiprocessing.set_start_method("spawn", force=True)

    run_pipeline(
        models=models,
        gpu_ids=args.gpus,
        output_dir=args.output_dir,
        use_bf16=not args.no_bf16,
        use_flash_attn=not args.no_flash_attn,
        truncate_dims=truncate_dims,
        skip_completed=not args.no_skip,
    )


if __name__ == "__main__":
    main()
