"""Graft the A.X-Encoder-base MLM head onto the KURE-v2-unsupervised body: the KURE-Sparse-nano backbone.

The SparseUp recipe (Linkup 2026): start SPLADE from a late-interaction pre-finetuned body, which has no
MLM head any more, and branch the original backbone's head back on. KURE-v2-unsupervised is A.X-Encoder-base
pre-finetuned with PyLate (train/late-interaction/1_pft.py); its drift from A.X is tiny (tok_embeddings
row-cosine 0.9996, layers ~5% relative Frobenius), so the tied decoder stays the body's own embeddings and only
head.dense / head.norm / decoder.bias come from A.X. The body's vocabulary is 50002 (PyLate added "[Q] "=50000
and "[D] "=50001): the bias is zero-padded, and src/pooling.py mutes those two output dims at train time.
Fill-mask sanity after grafting: "서울은 대한민국의 <mask>이다" -> 수도 / 소재지 / 국 / 도시.

    uv run python train/sparse/graft_mlm_head.py output/ax-kurev2unsup-mlm
"""

import argparse

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import AutoConfig, AutoTokenizer, ModernBertForMaskedLM

BODY, HEAD = "nlpai-lab/KURE-v2-unsupervised", "skt/A.X-Encoder-base"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("output_dir")
    args = parser.parse_args()

    config = AutoConfig.from_pretrained(HEAD)
    config.vocab_size = 50002
    config.tie_word_embeddings = True
    model = ModernBertForMaskedLM(config)
    body = load_file(hf_hub_download(BODY, "model.safetensors"))
    head = load_file(hf_hub_download(HEAD, "model.safetensors"))
    state = {"model." + k: v.float() for k, v in body.items()}
    state["head.dense.weight"] = head["head.dense.weight"].float()
    state["head.norm.weight"] = head["head.norm.weight"].float()
    state["decoder.bias"] = torch.cat([head["decoder.bias"].float(), torch.zeros(2)])
    missing, unexpected = model.load_state_dict(state, strict=False)
    print("missing:", missing, "| unexpected:", unexpected)
    model.tie_weights()
    print("decoder tied to the body's embeddings:",
          model.model.embeddings.tok_embeddings.weight.data_ptr() == model.decoder.weight.data_ptr())
    model.save_pretrained(args.output_dir)
    AutoTokenizer.from_pretrained(BODY).save_pretrained(args.output_dir)
    print("saved ->", args.output_dir)


if __name__ == "__main__":
    main()
