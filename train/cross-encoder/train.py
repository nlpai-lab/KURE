"""KURE-Reranker: a Korean/English reranker (CrossEncoder) distilled from a Qwen3-Reranker teacher.

Pointwise MSE on the teacher's raw yes/no logit (the ettin-reranker recipe), one stage over the 11-way
SFT rows flattened to (query, doc, label). Two backbones:

    causal LM (Qwen/Qwen3-1.7B -> KURE-Reranker-base): scores with logit("yes") - logit("no") over the
        Qwen3-Reranker chat template, the input format the teacher labelled with (src/qwen_backbone.py)
    encoder (skt/A.X-Encoder-base -> KURE-Reranker-nano): a fresh single-logit classification head

--extended_length trains at max_seq_length 8192 with length-grouped batches (src/length_grouping.py).
"""

import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
from sentence_transformers.cross_encoder import (
    CrossEncoder,
    CrossEncoderModelCardData,
    CrossEncoderTrainer,
    CrossEncoderTrainingArguments,
)
from sentence_transformers.cross_encoder.losses import MSELoss
from transformers import AutoConfig, HfArgumentParser

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))
sys.path.insert(0, str(HERE.parents[1] / "eval" / "cross-encoder"))  # the in-training evaluator
from data import load_sft  # noqa: E402
from evaluate import RerankNDCGEvaluator  # noqa: E402
from length_grouping import LengthGroupedCrossEncoderTrainer, ProbeMemoryCallback  # noqa: E402
from qwen_backbone import DEFAULT_INSTRUCT, build_qwen_cross_encoder, is_causal  # noqa: E402
from utils import setup_logging  # noqa: E402


@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="skt/A.X-Encoder-base")
    dtype: str = field(default="bfloat16")
    attn_implementation: str = field(default="flash_attention_2")
    max_seq_length: int = field(default=8192)
    prompt: str = field(
        default="",
        metadata={"help": "instruction for a causal backbone; must match the teacher's <Instruct> so "
                          "student logits share the label's input space (default: the teacher's)"},
    )


@dataclass
class DataArguments:
    train_dataset_path: list[str] = field(default_factory=list, metadata={"nargs": "+"})
    max_train_samples: int = field(default=0, metadata={"help": "0 = all"})


@dataclass
class RerankerTrainingArguments(CrossEncoderTrainingArguments):
    # Extended-length training (src/length_grouping.py): length-homogeneous batches shuffled in
    # blocks of world_size, a batch over the token budget row-chunked through the step, and the
    # longest block first as a memory probe.
    extended_length: bool = field(default=False)
    exlen_chunk_tokens: int = field(
        default=16_000,
        metadata={"help": "A batch whose text columns exceed this many tokens total is run a few "
                          "rows at a time; pointwise MSE has no in-batch terms so it stays exact."},
    )
    exlen_probe_blocks: int = field(
        default=1,
        metadata={"help": "How many of the longest blocks (one batch per rank each) to run first, "
                          "as the opening micro-steps, to prove the config does not OOM."},
    )


def build_model(model_args, causal: bool) -> CrossEncoder:
    model_kwargs = {"dtype": model_args.dtype, "attn_implementation": model_args.attn_implementation}
    card = CrossEncoderModelCardData(language=["ko", "en"], license="apache-2.0")
    if causal:
        return build_qwen_cross_encoder(model_args.model_name_or_path, max_length=model_args.max_seq_length,
                                        model_kwargs=model_kwargs, model_card_data=card)
    return CrossEncoder(
        model_args.model_name_or_path,
        max_length=model_args.max_seq_length,
        model_kwargs=model_kwargs,
        model_card_data=card,
        num_labels=1,
        # Regressing the teacher's raw logit, so the score must stay un-squashed; ST's default for
        # num_labels=1 is Sigmoid and that default is what the checkpoint saves.
        activation_fn=nn.Identity(),
    )


def main(model_args, data_args, training_args):
    setup_logging()
    logging.info(f"GPUs: {torch.cuda.device_count()} | {model_args} | {data_args}")

    causal = is_causal(AutoConfig.from_pretrained(model_args.model_name_or_path))
    model = build_model(model_args, causal)
    if causal:
        # The prompt becomes the system message, which the template drops into `<Instruct>: `.
        training_args.prompts = model_args.prompt or DEFAULT_INSTRUCT
        logging.info(f"causal reranker: <Instruct> = {training_args.prompts!r}")

    train_dataset = load_sft(data_args.train_dataset_path, data_args.max_train_samples or None)
    logging.info(f"train dataset: {train_dataset}")
    evaluator = RerankNDCGEvaluator(batch_size=training_args.per_device_eval_batch_size)

    trainer_cls, extra = CrossEncoderTrainer, {}
    if training_args.extended_length:
        trainer_cls = LengthGroupedCrossEncoderTrainer
        extra = {"chunk_tokens": training_args.exlen_chunk_tokens,
                 "probe_blocks": training_args.exlen_probe_blocks}
    trainer = trainer_cls(model=model, args=training_args, train_dataset=train_dataset,
                          loss=MSELoss(model), evaluator=evaluator, **extra)
    if training_args.extended_length and training_args.exlen_probe_blocks:
        probe_steps = -(-training_args.exlen_probe_blocks // training_args.gradient_accumulation_steps)
        trainer.add_callback(ProbeMemoryCallback(probe_steps))
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)

    evaluator(model)
    model.save_pretrained(os.path.join(training_args.output_dir, "final"))


if __name__ == "__main__":
    parser = HfArgumentParser((ModelArguments, DataArguments, RerankerTrainingArguments))
    main(*parser.parse_args_into_dataclasses())
