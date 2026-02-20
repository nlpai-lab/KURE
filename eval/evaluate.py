"""
KURE Evaluation System

Benchmarking embedding models on Korean retrieval tasks including:
- MTEB Korean Retrieval tasks (9 tasks)
- NanoBEIR-ko tasks (13 subsets)

Features:
- bf16 + flash-attention support for optimized inference
- MRL (Matryoshka) dimension truncation support
- MTEB automatic prompt handling
- Multi-GPU parallel evaluation with task queue
- Sequential task processing per GPU (queue-based)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import traceback
from multiprocessing import Process, Queue, current_process
from pathlib import Path
from typing import Any

import torch
from setproctitle import setproctitle

from mteb import MTEB

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.models.loader import load_model, get_output_folder
from eval.models.config import get_model_config
from eval.tasks.registry import get_all_tasks, get_task_names, ALL_TASK_NAMES, MTEB_TASKS, NANOBEIR_KO_TASK_NAMES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("evaluate")

# Suppress noisy HTTP request logs from huggingface_hub
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

# Task weight estimates (higher = slower/larger corpus)
# Tasks not listed here default to weight 1
TASK_WEIGHTS: dict[str, int] = {
    "MIRACLRetrieval": 5,
    "MrTidyRetrieval": 5,
    "SQuADKorV1Retrieval": 4,
    "MultiLongDocRetrieval": 3,
    "Ko-StrategyQA": 2,
    "AutoRAGRetrieval": 2,
    "LawIRKo": 2,
}


def distribute_tasks(
    gpu_ids: list[int],
    task_names: list[str] | None = None,
) -> dict[int, list[str]]:
    """
    Distribute tasks across GPUs using greedy load-balancing.

    Sorts tasks by weight descending (heaviest first), then assigns each task
    to the GPU with the lowest current total weight.

    Args:
        gpu_ids: List of GPU IDs to distribute across.
        task_names: Specific task names to distribute. If None, uses all 22 tasks.

    Returns:
        Dict mapping gpu_id → list of task names.
    """
    tasks = task_names or list(ALL_TASK_NAMES)

    # Sort tasks by weight descending (heaviest first for better balancing)
    sorted_tasks = sorted(tasks, key=lambda t: TASK_WEIGHTS.get(t, 1), reverse=True)

    # Initialize GPU buckets
    assignment: dict[int, list[str]] = {gpu_id: [] for gpu_id in gpu_ids}
    gpu_load: dict[int, int] = {gpu_id: 0 for gpu_id in gpu_ids}

    # Greedy: assign each task to the GPU with the lowest current load
    for task in sorted_tasks:
        lightest_gpu = min(gpu_ids, key=lambda g: gpu_load[g])
        assignment[lightest_gpu].append(task)
        gpu_load[lightest_gpu] += TASK_WEIGHTS.get(task, 1)

    return assignment


def check_task_completed(output_dir: str, output_folder: str, task_name: str) -> bool:
    """Check if a task has already been completed by looking for result file."""
    # MTEB saves results as {task_name}.json
    result_path = Path(output_dir) / output_folder

    # Check for various possible result file patterns
    possible_files = [
        result_path / f"{task_name}.json",
        result_path / task_name / "results.json",
    ]

    for file_path in possible_files:
        if file_path.exists():
            try:
                with open(file_path) as f:
                    data = json.load(f)
                # Verify it has actual results
                if data and ("scores" in data or "test" in data):
                    return True
            except (json.JSONDecodeError, KeyError):
                pass

    return False


def evaluate_single_task(
    model: Any,
    task_name: str,
    output_dir: str,
    output_folder: str,
    batch_size: int,
    gpu_id: int,
) -> bool:
    """
    Evaluate a single task.

    Returns:
        True if successful, False otherwise
    """
    try:
        # Get task instance
        task_instances = get_all_tasks([task_name])

        if not task_instances:
            logger.warning(f"GPU {gpu_id}: Task '{task_name}' not found, skipping")
            return False

        task = task_instances[0]
        logger.info(f"GPU {gpu_id}: Starting task '{task_name}'")

        # Run evaluation for this single task
        evaluation = MTEB(tasks=[task])
        evaluation.run(
            model,
            output_folder=os.path.join(output_dir, output_folder),
            encode_kwargs={"batch_size": batch_size},
        )

        logger.info(f"GPU {gpu_id}: Completed task '{task_name}'")
        return True

    except Exception as e:
        logger.error(f"GPU {gpu_id}: Error in task '{task_name}': {e}")
        traceback.print_exc()
        return False


def evaluate_model_on_gpu_queue(
    model_name: str,
    gpu_id: int,
    task_queue: list[str],
    output_dir: str,
    use_bf16: bool = True,
    use_flash_attn: bool = True,
    truncate_dim: int | None = None,
    batch_size_override: int | None = None,
    skip_completed: bool = True,
) -> None:
    """
    Evaluate a model on tasks from a queue, one task at a time.

    Tasks are processed sequentially - when one finishes, the next starts.
    Model is loaded once and reused for all tasks.

    Args:
        model_name: HuggingFace model name or local path
        gpu_id: GPU device ID
        task_queue: List of task names to evaluate (processed sequentially)
        output_dir: Base output directory for results
        use_bf16: Whether to use bfloat16 precision
        use_flash_attn: Whether to use flash attention 2
        truncate_dim: MRL truncation dimension (None = full dimension)
        batch_size_override: Override batch size from config
        skip_completed: Skip tasks that already have results
    """
    try:
        # Set GPU device
        device = torch.device(f"cuda:{gpu_id}")
        torch.cuda.set_device(device)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        # Generate output folder name
        output_folder = get_output_folder(model_name, truncate_dim)

        # Set process title for monitoring
        setproctitle(f"eval-{output_folder[:30]}-gpu{gpu_id}")

        logger.info(f"GPU {gpu_id}: Starting queue-based evaluation of {model_name}")
        logger.info(f"GPU {gpu_id}: Task queue ({len(task_queue)} tasks): {task_queue}")

        # Load model once (reused for all tasks)
        logger.info(f"GPU {gpu_id}: Loading model...")
        model, default_batch_size = load_model(
            model_name,
            device=device,
            use_bf16=use_bf16,
            use_flash_attn=use_flash_attn,
            truncate_dim=truncate_dim,
        )

        batch_size = batch_size_override or default_batch_size
        logger.info(f"GPU {gpu_id}: Model loaded, using batch_size={batch_size}")

        # Process tasks one by one from the queue
        completed = 0
        skipped = 0
        failed = 0

        for idx, task_name in enumerate(task_queue, 1):
            # Check if already completed
            if skip_completed and check_task_completed(output_dir, output_folder, task_name):
                logger.info(f"GPU {gpu_id}: [{idx}/{len(task_queue)}] Skipping '{task_name}' (already completed)")
                skipped += 1
                continue

            logger.info(f"GPU {gpu_id}: [{idx}/{len(task_queue)}] Processing '{task_name}'...")
            start_time = time.time()

            success = evaluate_single_task(
                model=model,
                task_name=task_name,
                output_dir=output_dir,
                output_folder=output_folder,
                batch_size=batch_size,
                gpu_id=gpu_id,
            )

            elapsed = time.time() - start_time

            if success:
                completed += 1
                logger.info(f"GPU {gpu_id}: [{idx}/{len(task_queue)}] '{task_name}' done in {elapsed:.1f}s")
            else:
                failed += 1
                logger.error(f"GPU {gpu_id}: [{idx}/{len(task_queue)}] '{task_name}' failed after {elapsed:.1f}s")

        # Summary
        logger.info(f"GPU {gpu_id}: Queue completed - {completed} done, {skipped} skipped, {failed} failed")

    except Exception as e:
        logger.error(f"GPU {gpu_id}: Fatal error: {e}")
        traceback.print_exc()


def evaluate_model(
    model_name: str,
    tasks: list[str] | None = None,
    gpu_ids: list[int] | None = None,
    output_dir: str = "eval/results",
    use_bf16: bool = True,
    use_flash_attn: bool = True,
    truncate_dim: int | None = None,
    batch_size: int | None = None,
    skip_completed: bool = True,
) -> None:
    """
    Evaluate a model on Korean retrieval tasks using multiple GPUs.

    Each GPU processes its task queue sequentially (one task at a time).
    Multiple GPUs work in parallel on their respective queues.

    Args:
        model_name: HuggingFace model name or local path
        tasks: List of task names (None = all tasks)
        gpu_ids: List of GPU IDs to use (None = use all mapped GPUs)
        output_dir: Base output directory for results
        use_bf16: Whether to use bfloat16 precision
        use_flash_attn: Whether to use flash attention 2
        truncate_dim: MRL truncation dimension (None = full dimension)
        batch_size: Override batch size from config
        skip_completed: Skip tasks that already have results
    """
    logger.info(f"Starting evaluation for model: {model_name}")
    logger.info(f"Mode: Queue-based (sequential tasks per GPU, parallel across GPUs)")

    if gpu_ids is None:
        num_gpus = torch.cuda.device_count()
        gpu_ids = list(range(num_gpus))
        logger.info(f"Auto-detected {num_gpus} GPUs: {gpu_ids}")

    # Distribute tasks across GPUs
    task_distribution = distribute_tasks(gpu_ids, tasks)
    for gid, gtasks in task_distribution.items():
        logger.info(f"GPU {gid}: {len(gtasks)} tasks assigned")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    processes = []

    for gpu_id in gpu_ids:
        gpu_tasks = task_distribution.get(gpu_id, [])

        if not gpu_tasks:
            logger.info(f"GPU {gpu_id}: No tasks in queue, skipping")
            continue

        p = Process(
            target=evaluate_model_on_gpu_queue,
            args=(
                model_name,
                gpu_id,
                gpu_tasks,
                output_dir,
                use_bf16,
                use_flash_attn,
                truncate_dim,
                batch_size,
                skip_completed,
            ),
        )
        p.start()
        processes.append((gpu_id, p))
        logger.info(f"Started GPU {gpu_id} process with queue of {len(gpu_tasks)} tasks")

    # Wait for all processes to complete
    for gpu_id, p in processes:
        p.join()
        logger.info(f"GPU {gpu_id} process finished")

    logger.info(f"Completed evaluation for model: {model_name}")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate embedding models on Korean retrieval benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate a model on all tasks using all GPUs (queue-based)
  python eval/evaluate.py --model BAAI/bge-m3

  # Evaluate with specific tasks
  python eval/evaluate.py --model BAAI/bge-m3 --tasks BelebeleRetrieval NanoFEVERKo

  # Evaluate with MRL truncation
  python eval/evaluate.py --model jinaai/jina-embeddings-v3 --dim 256

  # Evaluate on specific GPUs
  python eval/evaluate.py --model BAAI/bge-m3 --gpus 0 1

  # Force re-evaluation (don't skip completed tasks)
  python eval/evaluate.py --model BAAI/bge-m3 --no_skip

  # Disable optimizations
  python eval/evaluate.py --model BAAI/bge-m3 --no_bf16 --no_flash_attn

  # List available tasks
  python eval/evaluate.py --list_tasks
        """,
    )

    parser.add_argument(
        "--model",
        type=str,
        help="HuggingFace model name or local path",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        default=None,
        help="Task names to evaluate (default: all tasks)",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        nargs="+",
        default=None,
        help="GPU IDs to use (default: auto-detect via CUDA_VISIBLE_DEVICES)",
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=None,
        help="MRL truncate dimension (default: full dimension)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Override batch size from model config",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="eval/results",
        help="Output directory for results (default: eval/results)",
    )
    parser.add_argument(
        "--no_bf16",
        action="store_true",
        help="Disable bfloat16 precision",
    )
    parser.add_argument(
        "--no_flash_attn",
        action="store_true",
        help="Disable flash attention 2",
    )
    parser.add_argument(
        "--no_skip",
        action="store_true",
        help="Don't skip completed tasks (force re-evaluation)",
    )
    parser.add_argument(
        "--list_tasks",
        action="store_true",
        help="List all available tasks and exit",
    )
    parser.add_argument(
        "--list_models",
        action="store_true",
        help="List all configured models and exit",
    )

    return parser.parse_args()


def list_tasks() -> None:
    """Print all available tasks."""
    print("\n" + "=" * 60)
    print("Available Evaluation Tasks")
    print("=" * 60)

    print("\nMTEB Korean Retrieval Tasks:")
    for i, task in enumerate(MTEB_TASKS, 1):
        print(f"  {i:2d}. {task}")

    print(f"\nNanoBEIR-ko Tasks ({len(NANOBEIR_KO_TASK_NAMES)} subsets):")
    for i, task in enumerate(NANOBEIR_KO_TASK_NAMES, 1):
        print(f"  {i:2d}. {task}")

    print(f"\nTotal: {len(MTEB_TASKS) + len(NANOBEIR_KO_TASK_NAMES)} tasks")

    # Show example distribution for 2 and 3 GPUs
    for n_gpus in [2, 3]:
        example_dist = distribute_tasks(list(range(n_gpus)))
        print(f"\nExample distribution ({n_gpus} GPUs):")
        for gpu_id, gpu_tasks in example_dist.items():
            total_weight = sum(TASK_WEIGHTS.get(t, 1) for t in gpu_tasks)
            print(f"  GPU {gpu_id}: {len(gpu_tasks)} tasks (weight={total_weight})")
            for idx, task in enumerate(gpu_tasks, 1):
                w = TASK_WEIGHTS.get(task, 1)
                print(f"    {idx}. {task}" + (f" (w={w})" if w > 1 else ""))

    print("=" * 60 + "\n")


def list_models() -> None:
    """Print all configured models."""
    from eval.models.config import MODEL_CONFIGS

    print("\n" + "=" * 60)
    print("Configured Models")
    print("=" * 60)

    for name, config in MODEL_CONFIGS.items():
        mrl_info = ""
        if config.supports_mrl:
            mrl_info = f" [MRL: {config.mrl_dims}]"

        flash_info = "flash-attn" if config.supports_flash_attn else "SDPA"

        print(f"\n  {name}")
        print(f"    batch_size: {config.batch_size}, {flash_info}{mrl_info}")

    print("\n" + "=" * 60 + "\n")


def main() -> None:
    """Main entry point."""
    args = parse_args()

    if args.list_tasks:
        list_tasks()
        return

    if args.list_models:
        list_models()
        return

    if not args.model:
        print("Error: --model is required")
        print("Use --list_tasks to see available tasks")
        print("Use --list_models to see configured models")
        sys.exit(1)

    # Set multiprocessing start method
    torch.multiprocessing.set_start_method("spawn", force=True)

    # Run evaluation
    evaluate_model(
        model_name=args.model,
        tasks=args.tasks,
        gpu_ids=args.gpus,
        output_dir=args.output_dir,
        use_bf16=not args.no_bf16,
        use_flash_attn=not args.no_flash_attn,
        truncate_dim=args.dim,
        batch_size=args.batch_size,
        skip_completed=not args.no_skip,
    )


if __name__ == "__main__":
    main()
