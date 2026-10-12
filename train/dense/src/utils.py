import logging
import os
import sys

# Loggers that configure themselves and so ignore the root level set by basicConfig.
_THIRD_PARTY = ("transformers", "datasets", "sentence_transformers", "mteb")


def setup_logging():
    """INFO on the main process, warnings only on the replicas.

    Every rank runs the same code, so at INFO an 8-GPU run prints each line eight times and the
    output worth reading -- losses, eval scores -- is buried in the duplicates. Replicas stay at
    WARNING so a real problem on rank 5 still surfaces.
    """
    rank = int(os.environ.get("RANK", "0"))
    main = rank == 0
    logging.basicConfig(
        level=logging.INFO if main else logging.WARNING,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    if not main:
        for name in _THIRD_PARTY:
            logging.getLogger(name).setLevel(logging.WARNING)
