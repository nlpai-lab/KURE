"""MILCO (arXiv:2510.00671) as a sentence-transformers input module, so this pipeline trains,
saves and evaluates it like any SparseEncoder.

MILCO is a bge-m3 body (XLM-R large) -> projector -> splade-v3's MLM head over the English BERT
vocabulary (the pivot view, 30522 dims), plus LexEcho: one relu score per input token, placed on that
token's XLM-R id (the source view, 250002 dims). Its native API returns the two as one sparse
[N, 30522 + 250002] tensor scored by a dot product; this module emits the same vector dense, as
``sentence_embedding``, so every loss and regularizer here applies to both views unchanged.

The source view adds up the scores of a repeated token: MILCO builds its COO tensor with one entry per
position and coalescing sums duplicates, so this uses scatter_add. Special tokens (<s>, </s>) are
attended and scored exactly as MILCO does.
"""

from __future__ import annotations

import inspect
import shutil
from pathlib import Path

import torch
from sentence_transformers.base.modules.input_module import InputModule
from transformers import AutoModel, AutoTokenizer


def is_milco(model_name_or_path: str) -> bool:
    """True for a MILCO checkpoint, a local directory or a Hub id (e.g. omai-research/milco-650m)."""
    config = Path(model_name_or_path) / "config.json"
    if not Path(model_name_or_path).is_dir():
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import EntryNotFoundError

        try:
            config = Path(hf_hub_download(model_name_or_path, "config.json"))
        except EntryNotFoundError:
            return False
    return config.is_file() and '"model_type": "milco"' in config.read_text()


class MilcoDualView(InputModule):
    config_file_name = "milco_encoder_config.json"  # config.json is MILCO's own
    config_keys = ["max_seq_length", "source_view", "chunk_size"]

    def __init__(
        self,
        model_name_or_path: str,
        max_seq_length: int = 8192,
        source_view: bool = True,
        chunk_size: int = 512,
        model_kwargs: dict | None = None,
    ) -> None:
        super().__init__()
        model_kwargs = dict(model_kwargs or {})
        # MILCOModel itself declares no attention backends; only its XLM-R body has them.
        attn = model_kwargs.pop("attn_implementation", None)
        # A dtype would reach only MILCOModel's own layers: its sub-models are built with from_config and keep
        # their config's float32, so bf16 projector/echo would sit next to an fp32 body -- and at lr 1e-5 a bf16
        # weight of typical size rounds every AdamW step away. Keep fp32 master weights; training runs under
        # --bf16 autocast, and the evaluation runs it in bf16.
        model_kwargs.pop("torch_dtype", None)
        model_kwargs.pop("dtype", None)
        self.milco = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True, **model_kwargs).float()
        if attn:
            self.milco.m_model.set_attn_implementation(attn)
        # BertLMPredictionHead.bias is never read in the forward (the decoder carries its own bias), but
        # MILCO's checkpoint stores the two on one storage, which save_pretrained refuses as untied
        # shared tensors. Give the unused one its own copy.
        head = self.milco.mlm_head.predictions
        head.bias = torch.nn.Parameter(head.decoder.bias.detach().clone())
        # Never used in the forward: frozen, so they neither take optimizer state nor stall a DDP built
        # with find_unused_parameters=False.
        for param in [*self.milco.m_model.pooler.parameters(), self.milco.scale, head.bias]:
            param.requires_grad_(False)
        # The XLM-R body supports checkpointing; the wrapper class just does not say so.
        self.milco.supports_gradient_checkpointing = True
        has_tokenizer = (Path(model_name_or_path) / "tokenizer_config.json").is_file()
        # trust_remote_code: a saved checkpoint's config.json maps to MILCO's code, and AutoTokenizer reads it.
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path if has_tokenizer else self.milco.config.multilingual_encoder_checkpoint,
            trust_remote_code=True,
        )
        self.max_seq_length = max_seq_length
        self.source_view = source_view
        self.chunk_size = chunk_size

    def preprocess(self, inputs, prompt: str | None = None, **kwargs) -> dict[str, torch.Tensor]:
        texts = self._prepend_prompt(list(inputs), prompt) if prompt else list(inputs)
        return self.tokenizer(
            texts, padding=True, truncation=True, max_length=self.max_seq_length, return_tensors="pt"
        )

    def forward(self, features: dict[str, torch.Tensor], **kwargs) -> dict[str, torch.Tensor]:
        # MILCOModel._encode, with the MLM head run over position chunks: it materializes
        # [batch, seq, 30522] logits at once, ~16 GB for an evaluation batch of 32 at 8192 tokens.
        # Max over chunks of the max is the same max, and log1p(relu(.)) is monotone, so the
        # result is unchanged.
        milco = self.milco
        input_ids, attention_mask = features["input_ids"], features["attention_mask"]
        hidden = milco.m_model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        mask = attention_mask.unsqueeze(-1)
        pivot = None
        for i in range(0, hidden.shape[1], self.chunk_size):
            logits = milco.mlm_head(milco.activation(milco.projector(hidden[:, i : i + self.chunk_size])))
            chunk_max = (logits * mask[:, i : i + self.chunk_size]).max(dim=1).values
            pivot = chunk_max if pivot is None else torch.maximum(pivot, chunk_max)
        pivot = torch.log1p(torch.relu(pivot)).float()
        if not self.source_view:
            features["sentence_embedding"] = pivot
            return features
        weights = torch.relu(milco.echo(hidden).squeeze(-1)) * attention_mask
        source = pivot.new_zeros((input_ids.shape[0], self.milco.config.m_vocab_size), dtype=torch.float32)
        source = source.scatter_add(1, input_ids, weights.float())
        features["sentence_embedding"] = torch.cat([pivot, source], dim=1)
        return features

    def get_sentence_embedding_dimension(self) -> int:
        config = self.milco.config
        return config.en_vocab_size + (config.m_vocab_size if self.source_view else 0)

    def save(self, output_path: str, *args, safe_serialization: bool = True, **kwargs) -> None:
        # Keep the unused bias equal to the trained one, in case a later transformers re-ties the two on load.
        head = self.milco.mlm_head.predictions
        head.bias.data.copy_(head.decoder.bias.data)
        self.milco.save_pretrained(output_path)
        # MILCO's remote code, for AutoModel(trust_remote_code=True) on the saved directory.
        shutil.copyfile(inspect.getfile(type(self.milco)), Path(output_path) / "milco.py")
        self.tokenizer.save_pretrained(output_path)
        self.save_config(output_path)
        # modules.json names this class as milco_encoder.MilcoDualView; ship the source so the
        # checkpoint loads where train/sparse/src is not on sys.path (as pooling.py does).
        shutil.copyfile(Path(__file__), Path(output_path) / Path(__file__).name)

    @classmethod
    def load(cls, model_name_or_path: str, subfolder: str = "", model_kwargs: dict | None = None, **kwargs):
        config = cls.load_config(model_name_or_path, subfolder=subfolder)
        path = str(Path(model_name_or_path) / subfolder) if subfolder else model_name_or_path
        return cls(path, model_kwargs=model_kwargs, **config)
