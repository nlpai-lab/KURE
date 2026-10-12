"""KURE-v3: dense retriever distillation.

KL-divergence distillation of a teacher's scores over each query's candidates (the positive and
its hard negatives), wrapped in Matryoshka Representation Learning, optionally with a Global
Orthogonal Regularization term. `--distill_loss kldiv_minmax` min-max normalizes both score
vectors per row before the softmax, so the KL matches the shape of the teacher's ranking rather
than its scale -- see src/losses.py.
"""

import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.sentence_transformer.data_collator import SentenceTransformerDataCollator
from sentence_transformers.sentence_transformer.losses import DistillKLDivLoss
from sentence_transformers.sentence_transformer.model_card import SentenceTransformerModelCardData
from sentence_transformers.sentence_transformer.trainer import SentenceTransformerTrainer
from transformers import HfArgumentParser

from arguments import DataArguments, DenseTrainingArguments, ModelArguments
from data import collator_prompts, load_train_datasets
from evaluators import load_evaluator
from length_grouping import LengthGroupedTrainer, ProbeMemoryCallback
from losses import ComponentLoggingMatryoshkaLoss, KLDivGORLoss, MinMaxDistillKLDivLoss
from utils import setup_logging


def build_loss(model, training_args):
    if training_args.distill_loss == "kldiv":
        base_loss = DistillKLDivLoss(model, temperature=training_args.kldiv_temperature)
    elif training_args.distill_loss == "kldiv_minmax":
        base_loss = MinMaxDistillKLDivLoss(model, temperature=training_args.kldiv_temperature)
    else:
        raise ValueError(f"Unknown distill_loss: {training_args.distill_loss!r}")
    if training_args.gor_weight:
        # Inside Matryoshka on purpose: one forward per dimension serves both terms.
        base_loss = KLDivGORLoss(
            model, base_loss, gor_weight=training_args.gor_weight,
            mean_weight=training_args.gor_mean_weight,
            second_moment_weight=training_args.gor_second_moment_weight,
            aggregation=training_args.gor_aggregation,
        )

    if not training_args.matryoshka_dims:
        return base_loss
    # Fail here rather than several minutes into training on a shape mismatch.
    full_dim = model.get_sentence_embedding_dimension()
    if max(training_args.matryoshka_dims) > full_dim:
        raise ValueError(f"matryoshka_dims {training_args.matryoshka_dims} exceed the model's {full_dim}")
    # ComponentLogging...: same total, but each dimension (and the inner kl/gor split) is
    # logged next to `loss` instead of vanishing into it.
    return ComponentLoggingMatryoshkaLoss(
        model, base_loss, matryoshka_dims=training_args.matryoshka_dims
    )


def main(model_args, data_args, training_args):
    setup_logging()
    logging.info(f"GPU count available: {torch.cuda.device_count()}")
    logging.info(f"Model args: {model_args}")
    logging.info(f"Data args: {data_args}")
    logging.info(f"Training args: {training_args}")

    model = SentenceTransformer(
        model_args.model_name_or_path,
        model_card_data=SentenceTransformerModelCardData(
            language=["ko", "en"],
            license="apache-2.0",
            model_name="KURE-v3",
        ),
        model_kwargs={
            "attn_implementation": model_args.attn_implementation,
            "torch_dtype": model_args.dtype,
        },
        trust_remote_code=model_args.trust_remote_code,
    )
    model.max_seq_length = model_args.max_seq_length

    train_dataset = load_train_datasets(
        data_args.train_dataset_path,
        num_negatives=data_args.num_negatives,
        mix_languages=training_args.mix_languages,
        normalize_scores=training_args.normalize_scores,
        drop_false_negatives=training_args.drop_false_negatives,
        false_negative_margin=training_args.false_negative_margin,
        max_train_samples=data_args.max_train_samples,
    )
    logging.info(f"Train dataset loaded: {train_dataset}")

    loss = build_loss(model, training_args)
    evaluator = load_evaluator(data_args, training_args)

    columns = (
        list(next(iter(train_dataset.values())).column_names)
        if isinstance(train_dataset, dict)
        else list(train_dataset.column_names)
    )
    prompts = collator_prompts(columns, training_args.query_prompt, training_args.document_prompt)
    logging.info(f"collator_prompts = {prompts}")

    trainer_cls, extra_kwargs = SentenceTransformerTrainer, {}
    if training_args.extended_length:
        trainer_cls = LengthGroupedTrainer
        extra_kwargs = {
            "chunk_tokens": training_args.exlen_chunk_tokens,
            "probe_blocks": training_args.exlen_probe_blocks,
        }
        logging.info(
            f"extended-length mode at max_seq_length {model_args.max_seq_length}: batches over "
            f"{training_args.exlen_chunk_tokens} tokens run in row chunks under that budget"
        )

    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        loss=loss,
        evaluator=evaluator,
        data_collator=SentenceTransformerDataCollator(
            preprocess_fn=model.preprocess, prompts=prompts
        ),
        **extra_kwargs,
    )
    if training_args.extended_length and training_args.exlen_probe_blocks:
        # One probe block per dataset is one micro-step each; report once they are all done.
        n_datasets = len(train_dataset) if isinstance(train_dataset, dict) else 1
        probe_micro_steps = n_datasets * training_args.exlen_probe_blocks
        probe_steps = -(-probe_micro_steps // training_args.gradient_accumulation_steps)
        trainer.add_callback(ProbeMemoryCallback(probe_steps))
    trainer.train(training_args.resume_from_checkpoint)

    evaluator(model)
    model.save_pretrained(os.path.join(training_args.output_dir, "final"))


if __name__ == "__main__":
    parser = HfArgumentParser((ModelArguments, DataArguments, DenseTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    main(model_args, data_args, training_args)
