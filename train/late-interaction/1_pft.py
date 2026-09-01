"""Stage 1 of two: unsupervised contrastive pre-finetuning (PFT) for KURE-v2.

Trains a ColBERT from a bilingual encoder on weakly related (anchor, positive) pairs
with a very large in-batch negative pool. Stage 2 (2_sft.py) then adds supervision.

What defines this stage:

    loss        pylate's CachedContrastive. There is no teacher and no mined
                negatives -- the only negatives are the other pairs in the batch,
                which is why the batch is enormous.
    batch       16,384 by default. `accelerator_config.split_batches` makes
                `--batch-size` the *global* pool, so the number of GPUs changes
                the wall clock and not the loss.
    lengths     Asymmetric on purpose. Query expansion MASK-pads every anchor to
                `--query-length`, so that number is paid on every row; documents
                only get truncated at `--document-length`.

Batches stay language-pure: passing a dict of datasets makes the trainer draw each
batch from a single dataset, so a 16k in-batch negative pool is never separable by
language alone -- with this many negatives, "not Korean" would otherwise be an easy
and useless discriminator.

Data layout: <data-root>/pft_<lang> is a datasets.save_to_disk directory whose rows
carry `anchor` and `positive` columns.

    uv run torchrun --nproc_per_node 8 train/late-interaction/1_pft.py \\
        --data-root /path/to/pft --head multi --run-name kure-v2-pft
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default="skt/A.X-Encoder-base")
    parser.add_argument("--embedding-size", type=int, default=128)
    # `single` is pylate's default head: one Dense, hidden -> embedding_size.
    # `multi` is the KURE-v2 head: hidden -> 2*hidden -> hidden, both residual, then
    # hidden -> embedding_size; every layer bias-free with an identity activation.
    parser.add_argument("--head", choices=("single", "multi"), default="multi")
    parser.add_argument("--trust-remote-code", action="store_true")

    parser.add_argument("--data-root", required=True,
                        help="Directory containing pft_<lang> datasets (anchor, positive).")
    parser.add_argument("--languages", nargs="+", default=["ko", "en"],
                        help="Loads <data-root>/pft_<lang> for each.")
    parser.add_argument("--max-samples", type=int, default=0, help="Cap rows per split (debug).")

    parser.add_argument("--query-length", type=int, default=64)
    parser.add_argument("--document-length", type=int, default=256)
    parser.add_argument("--query-expansion", choices=("on", "off"), default="on")

    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=16384,
                        help="Global in-batch negative pool, split across devices.")
    parser.add_argument("--mini-batch-size", type=int, default=512,
                        help="GradCache chunk. Memory only; does not change the loss.")
    # The score matrix is batch x batch, and for late interaction each entry is a MaxSim
    # over query_length x document_length token pairs. At 16,384 that matrix cannot be
    # materialised at once, so pylate chunks it. None lets pylate pick.
    parser.add_argument("--score-mini-batch-size", type=int, default=None)
    parser.add_argument("--accumulation", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.02)
    parser.add_argument("--precision", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--save-steps", type=float, default=0.1)
    parser.add_argument("--save-total-limit", type=int, default=None)
    parser.add_argument("--eval-steps", type=float, default=0.05)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--no-dev-eval", action="store_true")
    parser.add_argument("--no-eval-on-start", action="store_true")
    parser.add_argument("--stop-at-step", type=int, default=0,
                        help="Stop after N optimizer steps. For smoke runs.")

    parser.add_argument("--num-proc", type=int, default=16)
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--run-name", default="kure-v2-pft")
    parser.add_argument("--report-to", default="wandb")
    parser.add_argument("--wandb-project", default="kure-v2")
    parser.add_argument("--resume-from", default=None)
    return parser.parse_args()


def build_colbert(args, torch, models):
    """A ColBERT with either head. Shared with 2_sft.py.

    Zero-valued sizes mean "keep whatever the checkpoint carries": embedding_size=0
    skips requesting a projection (the checkpoint's own Dense stack is loaded as-is),
    and query/document_length=0 keep the checkpoint's configured lengths.
    """
    kwargs = {
        "do_query_expansion": args.query_expansion == "on",
        "model_kwargs": {"dtype": torch.float32, "attn_implementation": "sdpa"},
        "trust_remote_code": args.trust_remote_code,
    }
    if args.query_length > 0:
        kwargs["query_length"] = args.query_length
    if args.document_length > 0:
        kwargs["document_length"] = args.document_length
    if args.head == "single":
        if args.embedding_size > 0:
            kwargs["embedding_size"] = args.embedding_size
        return models.ColBERT(args.model_name, **kwargs)

    from sentence_transformers.models import Transformer

    # Built from modules rather than a name, so pylate takes the stack as given.
    model_args = dict(kwargs.pop("model_kwargs"))
    trusted = {"trust_remote_code": True} if kwargs.pop("trust_remote_code", False) else {}
    base = Transformer(args.model_name,
                       model_args={**model_args, **trusted},
                       tokenizer_args=dict(trusted) or None,
                       config_args=dict(trusted) or None)
    hidden = base.get_word_embedding_dimension()
    width = args.embedding_size or 128
    identity = torch.nn.Identity()
    stack = [
        models.Dense(hidden, 2 * hidden, bias=False, activation_function=identity,
                     use_residual=True),
        models.Dense(2 * hidden, hidden, bias=False, activation_function=identity,
                     use_residual=True),
        models.Dense(hidden, width, bias=False, activation_function=identity,
                     use_residual=False),
    ]
    logger.info("head=multi %d -> %d -> %d -> %d (residual on the first two)",
                hidden, 2 * hidden, hidden, width)
    return models.ColBERT(modules=[base, *stack], **kwargs)


def load_pairs(path: str, max_samples: int):
    """One (anchor, positive) dataset, with every other column dropped.

    Column *order* is what the collator reads, not column names, so a stray third column
    would silently become a second positive.
    """
    from datasets import load_from_disk

    dataset = load_from_disk(path)
    missing = {"anchor", "positive"} - set(dataset.column_names)
    if missing:
        raise SystemExit(f"{path}: missing column(s) {sorted(missing)}")
    dataset = dataset.select_columns(["anchor", "positive"])
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    logger.info("%s: %d pairs", path, len(dataset))
    return dataset


def main() -> None:
    args = parse_args()
    logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)
    if args.report_to == "wandb":
        os.environ.setdefault("WANDB_PROJECT", args.wandb_project)

    import torch
    from sentence_transformers import (
        SentenceTransformerTrainer,
        SentenceTransformerTrainingArguments,
    )
    from sentence_transformers.training_args import MultiDatasetBatchSamplers
    from pylate import losses, models, utils

    model = build_colbert(args, torch, models)
    logger.info("model=%s | query_length=%s document_length=%s expansion=%s head=%s",
                args.model_name, model.query_length, model.document_length,
                model.do_query_expansion, args.head)

    train_dataset = {
        lang: load_pairs(f"{args.data_root}/pft_{lang}", args.max_samples)
        for lang in args.languages
    }

    loss_kwargs = {}
    if args.score_mini_batch_size:
        loss_kwargs["score_mini_batch_size"] = args.score_mini_batch_size
    train_loss = losses.CachedContrastive(
        model=model,
        mini_batch_size=args.mini_batch_size,
        gather_across_devices=True,
        temperature=args.temperature,
        **loss_kwargs,
    )

    output_dir = f"{args.output_dir}/{args.run_name}"
    training_args = SentenceTransformerTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.accumulation,
        # split_batches: --batch-size is the global contrastive batch (the in-batch
        # negative pool), split across devices rather than multiplied by them.
        accelerator_config={"split_batches": True},
        learning_rate=args.learning_rate,
        # Transformers v5 deprecated warmup_ratio; a float warmup_steps is the ratio.
        warmup_steps=args.warmup_ratio,
        lr_scheduler_type="linear",
        bf16=args.precision == "bf16",
        fp16=args.precision == "fp16",
        seed=args.seed,
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        eval_strategy="no" if args.no_dev_eval else "steps",
        eval_steps=args.eval_steps,
        eval_on_start=not (args.no_dev_eval or args.no_eval_on_start),
        per_device_eval_batch_size=args.eval_batch_size,
        run_name=args.run_name,
        report_to=[] if args.report_to == "none" else args.report_to,
        multi_dataset_batch_sampler=MultiDatasetBatchSamplers.PROPORTIONAL,
    )

    sys.path.insert(0, str(Path(__file__).resolve().parent))

    callbacks = []
    if args.stop_at_step > 0:
        from modules import StopAtStepCallback

        callbacks.append(StopAtStepCallback(stop_at_step=args.stop_at_step))

    dev_evaluator = None
    if not args.no_dev_eval:
        from dev_evaluator import build_dev_evaluator

        dev_evaluator = build_dev_evaluator(batch_size=args.eval_batch_size)

    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        loss=train_loss,
        evaluator=dev_evaluator,
        data_collator=utils.ColBERTCollator(tokenize_fn=model.tokenize),
        callbacks=callbacks,
    )
    trainer.train(resume_from_checkpoint=args.resume_from)
    model.save_pretrained(f"{output_dir}/final")
    logger.info("saved %s/final", output_dir)


if __name__ == "__main__":
    main()
