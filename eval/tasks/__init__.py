"""Task definitions and registry."""

from eval.tasks.registry import get_all_tasks, MTEB_TASKS, NANOBEIR_KO_SUBSETS
from eval.tasks.nanobeir_ko import create_nanobeir_ko_tasks

__all__ = ["get_all_tasks", "MTEB_TASKS", "NANOBEIR_KO_SUBSETS", "create_nanobeir_ko_tasks"]
