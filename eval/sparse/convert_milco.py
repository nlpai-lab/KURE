"""Convert a MILCO checkpoint (omai-research/milco-*) into a SparseEncoder directory, so evaluate.py
scores it like any other sparse model.

MILCO ships as its own AutoModel (remote code, milco.py) with no modules.json, so SparseEncoder cannot load it
directly. Wrapping it in train/sparse/src/milco_encoder.py::MilcoDualView and saving gives the dense
[pivot | LexEcho source] vector MILCO scores with, and the same checkpoint layout a MILCO fine-tune writes.
The weights are untouched: evaluated this way milco-650m scored 0.7824 vs 0.7817 through MILCO's own encode_text.

    uv run python eval/sparse/convert_milco.py omai-research/milco-650m output/milco-650m
    uv run python eval/sparse/evaluate.py output/milco-650m
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "train" / "sparse" / "src"))

from milco_encoder import MilcoDualView  # noqa: E402
from sentence_transformers import SparseEncoder  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", help="MILCO hub id or local directory")
    parser.add_argument("output_dir")
    parser.add_argument("--max-seq-length", type=int, default=8192,
                        help="MILCO was trained at 512; its bge-m3 body takes 8194 positions")
    parser.add_argument("--no-source-view", action="store_true", help="Drop the LexEcho source view")
    args = parser.parse_args()

    module = MilcoDualView(args.model, max_seq_length=args.max_seq_length, source_view=not args.no_source_view)
    SparseEncoder(modules=[module]).save_pretrained(args.output_dir)
    print(f"saved {args.model} -> {args.output_dir} "
          f"({module.get_sentence_embedding_dimension()} dims, max_seq_length {args.max_seq_length})")


if __name__ == "__main__":
    main()
