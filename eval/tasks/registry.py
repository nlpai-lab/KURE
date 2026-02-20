"""Task registry aggregating MTEB Korean tasks and NanoBEIR-ko custom tasks."""

from __future__ import annotations

import logging
from typing import Sequence

import mteb
from mteb.abstasks import AbsTask

from eval.tasks.nanobeir_ko import create_nanobeir_ko_tasks, NANOBEIR_KO_SUBSETS

logger = logging.getLogger(__name__)

# MTEB built-in Korean Retrieval tasks
MTEB_TASKS = [
    "BelebeleRetrieval",
    "MultiLongDocRetrieval",
    "Ko-StrategyQA",
    "AutoRAGRetrieval",
    "PublicHealthQA",
    "MIRACLRetrieval",
    "MrTidyRetrieval",
    "LawIRKo",
    "SQuADKorV1Retrieval",
]

# NanoBEIR-ko task names (for reference)
NANOBEIR_KO_TASK_NAMES = [f"{subset}Ko" for subset in NANOBEIR_KO_SUBSETS]

# All task names
ALL_TASK_NAMES = MTEB_TASKS + NANOBEIR_KO_TASK_NAMES


def get_mteb_tasks(task_names: list[str] | None = None) -> list[AbsTask]:
    """
    Get MTEB built-in Korean retrieval tasks.

    Args:
        task_names: Specific task names to get. If None, returns all MTEB_TASKS.

    Returns:
        List of MTEB task instances
    """
    if task_names is None:
        task_names = MTEB_TASKS

    # Filter to only MTEB tasks
    mteb_task_names = [t for t in task_names if t in MTEB_TASKS]

    if not mteb_task_names:
        return []

    try:
        tasks = mteb.get_tasks(
            tasks=mteb_task_names,
            languages=["kor-Kore", "kor-Hang"],
        )
        logger.info(f"Loaded {len(tasks)} MTEB tasks: {[t.metadata.name for t in tasks]}")
        return list(tasks)
    except Exception as e:
        logger.error(f"Error loading MTEB tasks: {e}")
        return []


def get_nanobeir_ko_tasks(task_names: list[str] | None = None) -> list[AbsTask]:
    """
    Get NanoBEIR-ko custom retrieval tasks.

    Args:
        task_names: Specific task names to get. If None, returns all NanoBEIR-ko tasks.

    Returns:
        List of NanoBEIR-ko task instances
    """
    all_tasks = create_nanobeir_ko_tasks()

    if task_names is None:
        return all_tasks

    # Filter to requested tasks
    filtered_tasks = [t for t in all_tasks if t.metadata.name in task_names]
    logger.info(f"Loaded {len(filtered_tasks)} NanoBEIR-ko tasks")
    return filtered_tasks


def get_all_tasks(task_names: list[str] | None = None) -> list[AbsTask]:
    """
    Get all evaluation tasks (MTEB + NanoBEIR-ko).

    Args:
        task_names: Specific task names to get. If None, returns all tasks.

    Returns:
        Combined list of all task instances
    """
    if task_names is None:
        # Return all tasks
        mteb_tasks = get_mteb_tasks()
        nanobeir_tasks = get_nanobeir_ko_tasks()
        all_tasks = mteb_tasks + nanobeir_tasks
        logger.info(f"Loaded total {len(all_tasks)} tasks ({len(mteb_tasks)} MTEB + {len(nanobeir_tasks)} NanoBEIR-ko)")
        return all_tasks

    # Separate task names by type
    mteb_names = [t for t in task_names if t in MTEB_TASKS]
    nanobeir_names = [t for t in task_names if t in NANOBEIR_KO_TASK_NAMES]

    # Check for unknown tasks
    unknown = [t for t in task_names if t not in MTEB_TASKS and t not in NANOBEIR_KO_TASK_NAMES]
    if unknown:
        logger.warning(f"Unknown task names (will be skipped): {unknown}")

    tasks = []

    if mteb_names:
        tasks.extend(get_mteb_tasks(mteb_names))

    if nanobeir_names:
        tasks.extend(get_nanobeir_ko_tasks(nanobeir_names))

    return tasks


def get_task_names() -> list[str]:
    """Get all available task names."""
    return ALL_TASK_NAMES


def print_available_tasks() -> None:
    """Print all available tasks for reference."""
    print("=" * 60)
    print("Available Evaluation Tasks")
    print("=" * 60)

    print("\nMTEB Korean Retrieval Tasks:")
    for i, task in enumerate(MTEB_TASKS, 1):
        print(f"  {i:2d}. {task}")

    print(f"\nNanoBEIR-ko Tasks ({len(NANOBEIR_KO_TASK_NAMES)} subsets):")
    for i, task in enumerate(NANOBEIR_KO_TASK_NAMES, 1):
        print(f"  {i:2d}. {task}")

    print(f"\nTotal: {len(ALL_TASK_NAMES)} tasks")
    print("=" * 60)


if __name__ == "__main__":
    print_available_tasks()
