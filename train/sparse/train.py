"""KURE-Sparse: learned sparse (SPLADE) retriever training.

One entrypoint for both stages, selected by --distill_loss:

    contrastive         pre-finetuning on (anchor, positive) pairs with in-batch negatives (pft.sh)
    contrastive_kldiv   SFT: in-batch contrastive + KL distillation of a reranker teacher's scores
                        over each query's 11 candidates (train.sh)

with an L1 or FLOPS sparsity regularizer on queries and documents. The student is an MLM backbone
pooled by src/pooling.py (a SparseEncoder checkpoint keeps its own pooling), or MILCO through
src/milco_encoder.py.
"""

import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import torch
from sentence_transformers.base.modules import Transformer
from sentence_transformers.sparse_encoder import (
    SparseEncoder,
    SparseEncoderDataCollator,
    SparseEncoderModelCardData,
    SparseEncoderTrainer,
)
from sentence_transformers.sparse_encoder.losses import (
    SparseDistillKLDivLoss,
    SparseMultipleNegativesRankingLoss,
    SpladeLoss,
)
from sentence_transformers.sparse_encoder.modules import SpladePooling
from transformers import HfArgumentParser

from arguments import DataArguments, ModelArguments, SparseTrainingArguments
from data import collator_prompts, load_pft_datasets, load_train_datasets
from evaluators import load_evaluator
from length_grouping import LengthGroupedTrainer, ProbeMemoryCallback
from losses import (
    ComponentLoggingCachedSpladeLoss,
    L1Regularizer,
    MinMaxSparseDistillKLDivLoss,
    SparseContrastiveKLDivLoss,
    TokenBudgetSpladeLoss,
)
from milco_encoder import MilcoDualView, is_milco
from pooling import ShiftedTopKSpladePooling, canonical_ids_for
from utils import setup_logging


def is_sparse_encoder(model_name_or_path: str) -> bool:
    """True when the checkpoint already carries SparseEncoder modules.

    Without this, resuming a SPLADE checkpoint would silently rebuild it as
    MLM + SpladePooling and drop whatever module config it was saved with.
    """
    if os.path.isdir(model_name_or_path):
        return os.path.exists(os.path.join(model_name_or_path, "modules.json"))

    from huggingface_hub import file_exists

    return file_exists(model_name_or_path, "modules.json")


def build_model(model_args, training_args) -> SparseEncoder:
    model_card_data = SparseEncoderModelCardData(
        language=["ko", "en"],
        license="apache-2.0",
        model_name="Korean/English SPLADE retriever distilled from a reranker teacher",
    )
    model_kwargs = {
        "attn_implementation": model_args.attn_implementation,
        "torch_dtype": model_args.dtype,
    }
    if is_sparse_encoder(model_args.model_name_or_path):
        model = SparseEncoder(
            model_args.model_name_or_path,
            model_kwargs=model_kwargs,
            model_card_data=model_card_data,
        )
        model.max_seq_length = model_args.max_seq_length
        return model
    if is_milco(model_args.model_name_or_path):
        # Its own pivot/LexEcho heads replace the MLM + pooling stack, so none of the knobs below apply.
        milco = MilcoDualView(
            model_args.model_name_or_path, max_seq_length=model_args.max_seq_length, model_kwargs=model_kwargs
        )
        return SparseEncoder(modules=[milco], model_card_data=model_card_data)

    mlm = Transformer(
        model_args.model_name_or_path,
        transformer_task="fill-mask",
        max_seq_length=model_args.max_seq_length,
        model_kwargs=model_kwargs,
    )
    if model_args.mlm_head_rescale != 1.0:
        # W_head <- W_head / k (arXiv:2606.18811, eq. 6). For a tied head this is the shared
        # embedding matrix, which keeps the tying; the embedding LayerNorm undoes the shrink on the
        # input side, so only the vocab projection -- and with it every SPLADE score -- gets 1/k.
        # The bias is left alone, as in the paper.
        head = mlm.auto_model.get_output_embeddings()
        with torch.no_grad():
            head.weight.div_(model_args.mlm_head_rescale)
        logging.info(
            f"MLM head rescaled by 1/{model_args.mlm_head_rescale}: "
            f"row-L2 mean now {head.weight.float().norm(dim=1).mean():.3f}"
        )
    pooling_kwargs = dict(
        pooling_strategy=training_args.splade_pooling,
        activation_function="relu",
        chunk_size=training_args.splade_chunk_size or None,
    )
    exclude = list(model_args.splade_exclude_token_ids)
    canonical = model_args.splade_canonical_min_chars
    if model_args.splade_logit_shift or model_args.splade_top_k or exclude or canonical:
        pooling = ShiftedTopKSpladePooling(
            logit_shift=model_args.splade_logit_shift,
            top_k=model_args.splade_top_k or None,
            prefix_token_ids=exclude,
            masked_vocab_ids=exclude,
            canonical_ids=canonical_ids_for(mlm.tokenizer, canonical) if canonical else None,
            **pooling_kwargs,
        )
    else:
        pooling = SpladePooling(**pooling_kwargs)
    return SparseEncoder(modules=[mlm, pooling], model_card_data=model_card_data)


def build_loss(model, training_args):
    if training_args.distill_loss == "kldiv":
        base_loss = SparseDistillKLDivLoss(model, temperature=training_args.kldiv_temperature)
    elif training_args.distill_loss == "kldiv_minmax":
        base_loss = MinMaxSparseDistillKLDivLoss(model, temperature=training_args.kldiv_temperature)
    elif training_args.distill_loss == "contrastive":
        # Pre-finetuning: no teacher, no mined negatives. Every other row in the batch is the
        # negative pool, which is why this stage is the one that needs a huge batch.
        base_loss = SparseMultipleNegativesRankingLoss(
            model,
            scale=training_args.contrastive_scale,
            gather_across_devices=training_args.gather_across_devices,
        )
    elif training_args.distill_loss == "contrastive_kldiv":
        base_loss = SparseContrastiveKLDivLoss(
            model,
            contrastive_weight=training_args.contrastive_weight,
            kldiv_weight=training_args.kldiv_weight,
            scale=training_args.contrastive_scale,
            student_temperature=training_args.student_temperature
            or training_args.kldiv_temperature,
            teacher_temperature=training_args.teacher_temperature
            or training_args.kldiv_temperature,
            normalize_student=training_args.normalize_student,
            normalize_teacher=training_args.normalize_teacher,
            gather_across_devices=training_args.gather_across_devices,
        )
    else:
        raise ValueError(f"Unknown distill_loss: {training_args.distill_loss!r}")

    # GradCache pays a second forward pass to make the batch bigger than memory allows. That
    # is worth nothing for pure KL, which only ever ranks a query's own 11 candidates, but it
    # is the whole point once a contrastive term is in play: the negative pool IS the batch.
    # Verified equivalent -- against plain SpladeLoss in float64 the loss matched exactly and
    # the largest relative gradient difference was 1.8e-15.
    kwargs = dict(
        loss=base_loss,
        document_regularizer_weight=training_args.document_regularizer_weight,
        query_regularizer_weight=training_args.query_regularizer_weight,
    )
    if training_args.splade_regularizer == "l1":
        kwargs.update(document_regularizer=L1Regularizer(), query_regularizer=L1Regularizer())
    elif training_args.splade_regularizer != "flops":
        raise ValueError(f"Unknown splade_regularizer: {training_args.splade_regularizer!r}")
    if training_args.extended_length:
        # Only the contrastive losses couple the ranks, and only when they gather across them.
        sync_paths = "contrastive" in training_args.distill_loss and training_args.gather_across_devices
        return TokenBudgetSpladeLoss(
            model, chunk_tokens=training_args.exlen_chunk_tokens, sync_paths=sync_paths, **kwargs
        )
    if training_args.use_cached_splade:
        return ComponentLoggingCachedSpladeLoss(
            model, mini_batch_size=training_args.mini_batch_size, **kwargs
        )
    return SpladeLoss(model, **kwargs)


def main(model_args, data_args, training_args):
    setup_logging()
    logging.info(f"GPU count available: {torch.cuda.device_count()}")
    logging.info(f"Model args: {model_args}")
    logging.info(f"Data args: {data_args}")
    logging.info(f"Training args: {training_args}")

    model = build_model(model_args, training_args)

    if training_args.distill_loss == "contrastive":
        train_dataset = load_pft_datasets(
            data_args.train_dataset_path,
            mix_languages=training_args.mix_languages,
            max_train_samples=data_args.max_train_samples,
        )
    else:
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
    evaluator = load_evaluator(data_args, training_args, sparse=True)

    columns = (
        list(next(iter(train_dataset.values())).column_names)
        if isinstance(train_dataset, dict)
        else list(train_dataset.column_names)
    )
    prompts = collator_prompts(columns, training_args.query_prompt, training_args.document_prompt)
    logging.info(f"collator_prompts = {prompts}")
    # The collator prompts only reach the training batches. A model built here from modules has no
    # prompts of its own, so a checkpoint would be saved without them and encode_query /
    # encode_document -- what the in-training evaluator and eval/sparse/evaluate.py call -- would
    # encode bare text: a "[Q] "-trained student evaluated without "[Q] ", silently.
    wanted = {
        name: value
        for name, value in (("query", training_args.query_prompt), ("document", training_args.document_prompt))
        if value
    }
    # A checkpoint that carries prompts was trained with them; continuing it with different (or no)
    # QUERY_PROMPT/DOCUMENT_PROMPT would train and save under one convention and encode under another.
    # A fresh SparseEncoder ships empty-string defaults ({'query': '', 'document': ''}); those are
    # "no prompts", not a mismatch, so compare only the non-empty ones the checkpoint actually set.
    existing = {name: value for name, value in dict(model.prompts).items() if value}
    if existing and existing != wanted:
        raise SystemExit(f"checkpoint prompts {existing} differ from QUERY_PROMPT/DOCUMENT_PROMPT {wanted}")
    model.prompts = wanted

    trainer_cls, extra_kwargs = SparseEncoderTrainer, {}
    if training_args.extended_length:
        trainer_cls = LengthGroupedTrainer
        extra_kwargs = {"probe_blocks": training_args.exlen_probe_blocks}
        # Not optional: DDP re-broadcasts the module buffers on every forward that follows a
        # synchronised one, so a rank on the plain path (12 forwards) and a rank on the GradCache
        # path issue different numbers of collectives and the step hangs -- reproduced on 2 GPUs.
        # The only buffers are ModernBERT's constant rotary inv_freq; there is nothing to sync.
        training_args.ddp_broadcast_buffers = False
        logging.info(
            f"extended-length mode at max_seq_length {model_args.max_seq_length}: batches over "
            f"{training_args.exlen_chunk_tokens} padded tokens go through GradCache in chunks under that budget"
        )

    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        loss=loss,
        evaluator=evaluator,
        data_collator=SparseEncoderDataCollator(preprocess_fn=model.preprocess, prompts=prompts),
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
    parser = HfArgumentParser((ModelArguments, DataArguments, SparseTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    main(model_args, data_args, training_args)
