"""Make a plain Qwen3 causal LM usable as a CrossEncoder reranker, in the Qwen3-Reranker format.

`Qwen/Qwen3-Reranker-*` works with sentence-transformers out of the box because those repos ship
the wiring: a `sentence_bert_config.json` declaring a **message** modality, a chat template that
lays the pair out as `<Instruct>/<Query>/<Document>`, and a `1_LogitScore/config.json` naming the
yes/no token ids. A base `Qwen/Qwen3-0.6B` or `Qwen/Qwen3-1.7B` ships none of that, and
`CrossEncoder("Qwen/Qwen3-1.7B")` therefore builds a model whose pair inputs tokenize to nothing --
`predict()` raises `cannot reshape tensor of 0 elements`.

Nothing is missing from the weights; only the wiring. This module supplies it, so a base Qwen3 is
fed exactly what the teacher saw when it labelled the data:

    <|im_start|>system
    Judge whether the Document meets the requirements based on the Query and the Instruct
    provided. Note that the answer can only be "yes" or "no".<|im_end|>
    <|im_start|>user
    <Instruct>: {instruction}
    <Query>: {query}
    <Document>: {document}<|im_end|>
    <|im_start|>assistant
    <think>

    </think>

That is byte-for-byte the template the teacher (Qwen3-Reranker) scored the `label` column with -- so student and teacher score in the same input space, which is the point of the
distillation.

The yes/no token ids are the same 9693/2152 in the base and reranker tokenizers, but they are
looked up rather than hardcoded so a different backbone fails loudly instead of scoring garbage.
"""

from __future__ import annotations

import logging

from sentence_transformers.base.modules import Transformer
from sentence_transformers.cross_encoder import CrossEncoder, CrossEncoderModelCardData
from sentence_transformers.cross_encoder.modules.logit_score import LogitScore
from torch import nn

# Copied verbatim (741 chars) from Qwen/Qwen3-Reranker-0.6B's tokenizer_config.json. This
# layout is Qwen's own -- hence this module's name; it is not a general causal-LM recipe. The base Qwen3 chat template is a
# general-purpose conversation one and produces none of this structure.
QWEN_RERANKER_CHAT_TEMPLATE = (
    '{%- set instruction = messages | selectattr("role", "eq", "system") | map(attribute="content")'
    ' | first | default("Given a web search query, retrieve relevant passages that answer the query")'
    " -%}\n"
    '{%- set query_text = messages | selectattr("role", "eq", "query") | map(attribute="content")'
    " | first -%}\n"
    '{%- set document_text = messages | selectattr("role", "eq", "document") | map(attribute="content")'
    " | first -%}\n"
    "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the"
    ' Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n'
    "<|im_start|>user\n<Instruct>: {{ instruction }}\n<Query>: {{ query_text }}\n"
    "<Document>: {{ document_text }}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n\n"
)

# From Qwen/Qwen3-Reranker-0.6B's sentence_bert_config.json. The "message" entry is what turns a
# (query, document) pair into chat messages; without it the pair never reaches the tokenizer.
MESSAGE_MODALITY_CONFIG = {
    "text": {"method": "forward", "method_output_name": "logits"},
    "message": {"method": "forward", "method_output_name": "logits", "format": "flat"},
}
MODULE_OUTPUT_NAME = "causal_logits"  # LogitScore reads this by default

# The instruction the teacher used when labelling. It lands in the {{ instruction }} slot.
DEFAULT_INSTRUCT = "Given a web search query, retrieve relevant passages that answer the query"


def is_causal(config) -> bool:
    """True when the checkpoint is a causal LM, which needs the reranker wiring below."""
    architectures = getattr(config, "architectures", None) or []
    return bool(architectures) and architectures[0].endswith("ForCausalLM")


def build_qwen_cross_encoder(
    model_name_or_path: str,
    *,
    max_length: int,
    model_kwargs: dict,
    model_card_data: CrossEncoderModelCardData | None = None,
) -> CrossEncoder:
    """CrossEncoder over a Qwen3 causal LM that scores with logit("yes") - logit("no").

    Built explicitly rather than letting CrossEncoder auto-detect, so a base Qwen3 gets the same
    treatment as a Qwen3-Reranker: the message modality, the reranker chat template, and the
    yes/no head. Applying it to a repo that already ships this wiring is a no-op in effect -- the
    values are the ones that repo declares.
    """
    transformer = Transformer(
        model_name_or_path,
        transformer_task="text-generation",
        model_kwargs=model_kwargs,
        modality_config=MESSAGE_MODALITY_CONFIG,
        module_output_name=MODULE_OUTPUT_NAME,
        max_seq_length=max_length,
    )

    tokenizer = transformer.tokenizer
    template = tokenizer.chat_template or ""
    if "<Document>" not in template:
        # A general-purpose chat template would put the pair in a plain user turn, which is not
        # what the teacher scored. Swap in the reranker layout.
        tokenizer.chat_template = QWEN_RERANKER_CHAT_TEMPLATE
        logging.info("causal reranker: installed the Qwen3-Reranker chat template")
    else:
        logging.info("causal reranker: backbone already ships a reranker chat template")

    true_id = tokenizer.convert_tokens_to_ids("yes")
    false_id = tokenizer.convert_tokens_to_ids("no")
    unk = getattr(tokenizer, "unk_token_id", None)
    if true_id is None or false_id is None or true_id == unk or false_id == unk:
        raise ValueError(
            f"{model_name_or_path}: tokenizer has no single 'yes'/'no' token "
            f"(got {true_id}/{false_id}); LogitScore cannot score this backbone."
        )
    logging.info(f"causal reranker: LogitScore yes={true_id} no={false_id}")

    return CrossEncoder(
        modules=[transformer, LogitScore(true_token_id=true_id, false_token_id=false_id)],
        # The score is a raw logit margin; MSE regresses onto the teacher's raw margin.
        activation_fn=nn.Identity(),
        model_card_data=model_card_data,
    )
