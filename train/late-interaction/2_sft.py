"""Stage 2 of two: supervised fine-tuning (SFT) for KURE-v2.

Continues from the stage-1 (1_pft.py) checkpoint with contrastive learning plus
KL distillation from a reranker teacher, on (query, positive, negative_0..9) triplets
whose `label` column carries the teacher scores aligned to [positive, negative_0, ...].

Batches are language-pure: passing a dict of datasets makes SentenceTransformerTrainer
draw each batch from one dataset, so in-batch negatives are never separable by language
alone.

The loss (CachedContrastiveKLDiv) and collator (ColBERTCollatorSampleNeg) live in
modules.py, extracted from lightonai's mdenseon-mlateon training code (Apache-2.0).

Data layout: <data-root>/sft_<lang> is a datasets.save_to_disk directory whose rows
carry `query`, `positive`, `negative_0`..`negative_9`, and `label` columns.

    uv run torchrun --nproc_per_node 8 train/late-interaction/2_sft.py \\
        --model-name output/kure-v2-pft/final --data-root /path/to/sft \\
        --run-name kure-v2-sft
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# The collator tokenizes every column that is not a negative and not named
# "dataset_name" as a document, so only these columns may survive into training.
TEXT_COLUMNS = ["query", "positive"] + [f"negative_{i}" for i in range(10)]
KEEP_COLUMNS = TEXT_COLUMNS + ["label"]


def load_split(sft_root: str, name: str, margin: float, max_samples: int | None,
               num_proc: int, cache_dir: Path):
    """Load one sft split, drop false negatives, and keep only the columns the collator needs.

    label is [positive, negative_0, ..., negative_9] of teacher logits. A row whose
    hardest negative scores within `margin` of the positive is more likely a mislabelled
    positive than a negative, so it is dropped rather than distilled from. The threshold is
    a margin, not a ratio: these are logits and can be negative, so a ratio is not meaningful.
    """
    from datasets import load_from_disk

    dataset = load_from_disk(f"{sft_root}/{name}")
    before = len(dataset)

    dataset = dataset.filter(
        lambda batch: [max(label[1:]) < label[0] - margin for label in batch["label"]],
        batched=True,
        num_proc=num_proc,
        desc=f"{name}: false-negative filter",
        cache_file_name=str(cache_dir / f"{name}-filter-{margin}.arrow"),
    )
    logger.info(
        "%s: %d -> %d rows (dropped %d, %.2f%%) with margin=%.2f",
        name, before, len(dataset), before - len(dataset),
        100 * (before - len(dataset)) / before, margin,
    )

    dataset = dataset.select_columns(KEEP_COLUMNS)
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    return dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", required=True,
                        help="Stage-1 checkpoint (e.g. output/kure-v2-pft/final).")
    # 0 means "do not request a projection": use when the checkpoint already carries
    # its own Dense stack, so pylate does not append another.
    parser.add_argument("--embedding-size", type=int, default=0)
    # `single` with --embedding-size 0 loads the checkpoint's module stack as-is --
    # the right mode when --model-name is the stage-1 output, which already carries
    # the multi-layer head. `multi` BUILDS A FRESH head over the checkpoint's
    # transformer (discarding any trained head); use it only when starting stage 2
    # from a raw encoder.
    parser.add_argument("--head", choices=("single", "multi"), default="single")
    parser.add_argument("--query-expansion", choices=("on", "off"), default="on")
    parser.add_argument("--trust-remote-code", action="store_true")

    parser.add_argument("--data-root", required=True,
                        help="Directory containing sft_<lang> datasets "
                             "(query, positive, negative_0..9, label).")
    parser.add_argument("--languages", nargs="+", default=["ko", "en"],
                        help="sft_<lang> splits to train on.")

    # query_length is a fixed cost because query expansion MASK-pads every query to it;
    # document_length is only a truncation ceiling, since the collator pads to the batch
    # max. 0 means "keep whatever the checkpoint was configured with".
    parser.add_argument("--query-length", type=int, default=64)
    parser.add_argument("--document-length", type=int, default=1024)

    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Global batch (split across devices).")
    parser.add_argument("--accumulation", type=int, default=1,
                        help="Optimizer steps see --batch-size times this.")
    parser.add_argument("--mini-batch-size", type=int, default=8,
                        help="GradCache chunk; memory only, does not change the loss.")
    parser.add_argument("--num-negatives", type=int, default=7,
                        help="Negatives sampled per step from the stored 10.")

    parser.add_argument("--contrastive-weight", type=float, default=1.0, help="0 trains on KL alone.")
    parser.add_argument("--kldiv-weight", type=float, default=1.0)
    parser.add_argument("--contrastive-temperature", type=float, default=0.02)
    # Sharp student vs teacher temperature distills the teacher's ranking structure
    # rather than its absolute margins.
    parser.add_argument("--student-temperature", type=float, default=0.001)
    parser.add_argument("--teacher-temperature", type=float, default=0.1)

    # margin=0 drops only outright inversions (a negative outscoring the positive).
    # Raising it cuts much deeper for little gain, since KL distillation learns the
    # teacher's ranking and tolerates near-ties.
    parser.add_argument("--false-negative-margin", type=float, default=0.0)
    parser.add_argument("--max-samples", type=int, default=None, help="Cap rows per split (debug).")
    parser.add_argument("--num-proc", type=int, default=16)
    parser.add_argument("--cache-dir", default="cache")

    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--run-name", default="kure-v2-sft")
    parser.add_argument("--wandb-project", default="kure-v2")
    parser.add_argument("--report-to", default="wandb")
    parser.add_argument("--logging-steps", type=int, default=10)
    # Floats below 1 are ratios of total training steps.
    parser.add_argument("--save-steps", type=float, default=0.1)
    parser.add_argument("--eval-steps", type=float, default=0.05)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--no-dev-eval", action="store_true")
    parser.add_argument("--save-total-limit", type=int, default=None)
    parser.add_argument("--no-eval-on-start", action="store_true", help="Skip the step-0 baseline eval.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stop-at-step", type=int, default=0, help="Stop early for a smoke run.")
    parser.add_argument("--resume-from", default=None,
                        help='Checkpoint directory, or "auto" for the latest.')
    # bf16, not fp16: student_temperature=0.001 divides MaxSim scores (bounded by
    # query_length) by 1000, which lands at 97% of the fp16 range at query_length=64.
    parser.add_argument("--precision", choices=("bf16", "fp16", "fp32"), default="bf16")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)

    if args.report_to == "wandb":
        os.environ.setdefault("WANDB_PROJECT", args.wandb_project)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    import torch
    from sentence_transformers import (
        SentenceTransformerTrainer,
        SentenceTransformerTrainingArguments,
    )
    from sentence_transformers.training_args import MultiDatasetBatchSamplers
    from pylate import models

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from modules import (
        CachedContrastiveKLDiv,
        ColBERTCollatorSampleNeg,
        StopAtStepCallback,
    )
    # "1_pft" starts with a digit, so it cannot appear in an import statement
    import importlib
    build_colbert = importlib.import_module("1_pft").build_colbert

    model = build_colbert(args, torch, models)
    logger.info(
        "model=%s | query_length=%s document_length=%s do_query_expansion=%s",
        args.model_name, model.query_length, model.document_length, model.do_query_expansion,
    )

    # A dict of datasets makes the trainer draw each batch from a single dataset,
    # which is what keeps batches language-pure.
    train_dataset = {
        lang: load_split(args.data_root, f"sft_{lang}", args.false_negative_margin,
                         args.max_samples, args.num_proc, cache_dir)
        for lang in args.languages
    }
    for lang, dataset in train_dataset.items():
        logger.info("%s: %d rows, columns=%s", lang, len(dataset), dataset.column_names)

    train_loss = CachedContrastiveKLDiv(
        model=model,
        contrastive_temperature=args.contrastive_temperature,
        student_temperature=args.student_temperature,
        teacher_temperature=args.teacher_temperature,
        contrastive_weight=args.contrastive_weight,
        kldiv_weight=args.kldiv_weight,
        mini_batch_size=args.mini_batch_size,
    )

    output_dir = str(Path(args.output_dir) / args.run_name)
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

    callbacks = []
    if args.stop_at_step > 0:
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
        data_collator=ColBERTCollatorSampleNeg(
            model.tokenize, num_negatives=args.num_negatives
        ),
        callbacks=callbacks,
    )

    resume = args.resume_from
    if resume == "auto":
        found = sorted(Path(output_dir).glob("checkpoint-*"),
                       key=lambda d: int(d.name.split("-")[1]))
        resume = str(found[-1]) if found else None
        logger.info("resuming from %s", resume or "scratch")
    trainer.train(resume_from_checkpoint=resume)
    model.save_pretrained(f"{output_dir}/final")
    logger.info("saved to %s/final", output_dir)


if __name__ == "__main__":
    main()
