"""Model configuration registry with batch_size, bf16/flash-attn compatibility, MRL support."""

from __future__ import annotations

from dataclasses import dataclass, field
from mteb.types import PromptType


@dataclass
class ModelConfig:
    """Configuration for a specific embedding model."""

    name: str
    batch_size: int = 64
    supports_bf16: bool = True
    supports_flash_attn: bool = True  # False for BERT/RoBERTa architectures
    supports_mrl: bool = False  # Matryoshka Representation Learning
    default_dim: int | None = None  # Full embedding dimension
    mrl_dims: list[int] | None = None  # Supported MRL truncation dimensions
    custom_prompts: dict[str, str] | None = None  # Only for non-MTEB registered models


# Flash-attn compatibility:
# - Yes: Qwen, Mistral, LLaMA, Gemma architectures (causal attention)
# - No (use SDPA): BERT, RoBERTa, XLM-R architectures (bidirectional attention)

MODEL_CONFIGS: dict[str, ModelConfig] = {
    # === Large Instruction-tuned Models (7B+) ===
    "Alibaba-NLP/gte-Qwen2-7B-instruct": ModelConfig(
        name="Alibaba-NLP/gte-Qwen2-7B-instruct",
        batch_size=4,
        supports_flash_attn=True,  # Qwen2 architecture
    ),
    "intfloat/e5-mistral-7b-instruct": ModelConfig(
        name="intfloat/e5-mistral-7b-instruct",
        batch_size=4,
        supports_flash_attn=True,  # Mistral architecture
    ),
    "Salesforce/SFR-Embedding-2_R": ModelConfig(
        name="Salesforce/SFR-Embedding-2_R",
        batch_size=8,
        supports_flash_attn=True,  # Mistral-based
    ),
    # === Qwen3 Embedding Series ===
    "Qwen/Qwen3-Embedding-0.6B": ModelConfig(
        name="Qwen/Qwen3-Embedding-0.6B",
        batch_size=64,
        supports_flash_attn=True,
        supports_mrl=True,
        default_dim=1024,
        mrl_dims=[128, 256, 512, 768, 1024],
    ),
    "Qwen/Qwen3-Embedding-4B": ModelConfig(
        name="Qwen/Qwen3-Embedding-4B",
        batch_size=16,
        supports_flash_attn=True,
        supports_mrl=True,
        default_dim=2560,
        mrl_dims=[128, 256, 512, 1024, 1536, 2048, 2560],
    ),
    "Qwen/Qwen3-Embedding-8B": ModelConfig(
        name="Qwen/Qwen3-Embedding-8B",
        batch_size=8,
        supports_flash_attn=True,
        supports_mrl=True,
        default_dim=4096,
        mrl_dims=[128, 256, 512, 1024, 2048, 3072, 4096],
    ),
    # === Gemma-based Models ===
    "BAAI/bge-multilingual-gemma2": ModelConfig(
        name="BAAI/bge-multilingual-gemma2",
        batch_size=256,
        supports_flash_attn=True,  # Gemma2 architecture
    ),
    "google/embeddinggemma-300m": ModelConfig(
        name="google/embeddinggemma-300m",
        batch_size=256,
        supports_flash_attn=True,  # Gemma architecture
    ),
    # === MRL-supporting Models ===
    "nomic-ai/nomic-embed-text-v1.5": ModelConfig(
        name="nomic-ai/nomic-embed-text-v1.5",
        batch_size=64,
        supports_flash_attn=True,
        supports_mrl=True,
        default_dim=768,
        mrl_dims=[64, 128, 256, 512, 768],
    ),
    "nomic-ai/nomic-embed-text-v2-moe": ModelConfig(
        name="nomic-ai/nomic-embed-text-v2-moe",
        batch_size=64,
        supports_flash_attn=True,
        supports_mrl=True,
        default_dim=768,
        mrl_dims=[64, 128, 256, 512, 768],
    ),
    "jinaai/jina-embeddings-v3": ModelConfig(
        name="jinaai/jina-embeddings-v3",
        batch_size=8,
        supports_flash_attn=False,  # XLMRobertaLoRA doesn't support flash attn
        supports_mrl=True,
        default_dim=1024,
        mrl_dims=[32, 64, 128, 256, 512, 768, 1024],
    ),
	"jinaai/jina-embeddings-v5-text-small": ModelConfig(
        name="jinaai/jina-embeddings-v5-text-small",
        batch_size=64,
        supports_flash_attn=True,  
        supports_mrl=True,
        default_dim=1024,
        mrl_dims=[32, 64, 128, 256, 512, 768, 1024],
    ),
	"jinaai/jina-embeddings-v5-text-nano": ModelConfig(
        name="jinaai/jina-embeddings-v5-text-nano",
        batch_size=64,
        supports_flash_attn=True,  
        supports_mrl=True,
        default_dim=768,
        mrl_dims=[32, 64, 128, 256, 512, 768],
    ),
    "Snowflake/snowflake-arctic-embed-l-v2.0": ModelConfig(
        name="Snowflake/snowflake-arctic-embed-l-v2.0",
        batch_size=64,
        supports_flash_attn=True,
        supports_mrl=True,
        default_dim=1024,
        mrl_dims=[256, 512, 768, 1024],
        custom_prompts={
            PromptType.query.value: "query: ",
        },
    ),
    "dragonkue/snowflake-arctic-embed-l-v2.0-ko": ModelConfig(
        name="dragonkue/snowflake-arctic-embed-l-v2.0-ko",
        batch_size=64,
        supports_flash_attn=True,
        supports_mrl=True,
        default_dim=1024,
        mrl_dims=[256, 512, 768, 1024],
        custom_prompts={
            PromptType.query.value: "query: ",
        },
    ),
    # === BERT/RoBERTa-based Models (No Flash-Attn) ===
    "intfloat/multilingual-e5-large-instruct": ModelConfig(
        name="intfloat/multilingual-e5-large-instruct",
        batch_size=512,
        supports_flash_attn=False,  # XLM-RoBERTa architecture
    ),
    "intfloat/multilingual-e5-large": ModelConfig(
        name="intfloat/multilingual-e5-large",
        batch_size=512,
        supports_flash_attn=False,  # XLM-RoBERTa architecture
    ),
    "nlpai-lab/KoE5": ModelConfig(
        name="nlpai-lab/KoE5",
        batch_size=512,
        supports_flash_attn=False,  # mE5-based (XLM-RoBERTa)
        custom_prompts={
            PromptType.query.value: "query: ",
            PromptType.document.value: "passage: ",
        },
    ),
    "nlpai-lab/KURE-v1": ModelConfig(
        name="nlpai-lab/KURE-v1",
        batch_size=64,
        supports_flash_attn=False,  # BGE-M3 based
    ),
    "BAAI/bge-m3": ModelConfig(
        name="BAAI/bge-m3",
        batch_size=64,
        supports_flash_attn=False,  # XLM-RoBERTa architecture
    ),
    "dragonkue/BGE-m3-ko": ModelConfig(
        name="dragonkue/BGE-m3-ko",
        batch_size=64,
        supports_flash_attn=False,  # BGE-M3 based
    ),
    "upskyy/bge-m3-korean": ModelConfig(
        name="upskyy/bge-m3-korean",
        batch_size=64,
        supports_flash_attn=False,  # BGE-M3 based
    ),
    "jhgan/ko-sroberta-multitask": ModelConfig(
        name="jhgan/ko-sroberta-multitask",
        batch_size=512,
        supports_flash_attn=False,  # RoBERTa architecture
    ),
    "Alibaba-NLP/gte-multilingual-base": ModelConfig(
        name="Alibaba-NLP/gte-multilingual-base",
        batch_size=256,
        supports_flash_attn=False,  # BERT-like architecture
    ),
    # === Korean-specific Models ===
    "FronyAI/frony-embed-medium-arctic-ko-v2.5": ModelConfig(
        name="FronyAI/frony-embed-medium-arctic-ko-v2.5",
        batch_size=64,
        supports_flash_attn=True,
        custom_prompts={
            PromptType.query.value: "<Q>",
            PromptType.document.value: "<P>",
        },
    ),
    # === PIXIE Models ===
    "telepix/PIXIE-Spell-Preview-0.6B": ModelConfig(
        name="telepix/PIXIE-Spell-Preview-0.6B",
        batch_size=64,
        supports_flash_attn=True,
        custom_prompts={
            PromptType.query.value: "query: ",
        },
    ),
    "telepix/PIXIE-Spell-Preview-1.7B": ModelConfig(
        name="telepix/PIXIE-Spell-Preview-1.7B",
        batch_size=32,
        supports_flash_attn=True,
        custom_prompts={
            PromptType.query.value: "query: ",
        },
    ),
    "telepix/PIXIE-Rune-Preview": ModelConfig(
        name="telepix/PIXIE-Rune-Preview",
        batch_size=64,
        supports_flash_attn=True,
        custom_prompts={
            PromptType.query.value: "query: ",
        },
    ),
    "telepix/PIXIE-Rune-v1.0": ModelConfig(
        name="telepix/PIXIE-Rune-v1.0",
        batch_size=64,
        supports_flash_attn=False,  # Flash attention not supported
        # prompts: HuggingFace config에서 자동 로드 (query: )
    ),
    # === NVIDIA Models ===
    "nvidia/llama-embed-nemotron-8b": ModelConfig(
        name="nvidia/llama-embed-nemotron-8b",
        batch_size=4,
        supports_flash_attn=True,  # LLaMA architecture
        # prompts: HuggingFace config에서 자동 로드 (instruction)
    ),
    # === Other Models ===
    "SamilPwC-AXNode-GenAI/PwC-Embedding_expr": ModelConfig(
        name="SamilPwC-AXNode-GenAI/PwC-Embedding_expr",
        batch_size=256,
        supports_flash_attn=False,
    ),
}


def get_model_config(model_name: str) -> ModelConfig:
    """Get configuration for a model, with defaults for unknown models."""
    if model_name in MODEL_CONFIGS:
        return MODEL_CONFIGS[model_name]

    # Infer configuration based on model name patterns
    config = ModelConfig(name=model_name)

    # Batch size inference
    name_lower = model_name.lower()
    if "7b" in name_lower or "8b" in name_lower:
        config.batch_size = 4
    elif "4b" in name_lower:
        config.batch_size = 16
    elif "jina" in name_lower:
        config.batch_size = 8
    elif any(x in name_lower for x in ["e5-large", "koe5", "multilingual-e5"]):
        config.batch_size = 512
    elif "gemma" in name_lower:
        config.batch_size = 256

    # Flash attention inference based on architecture patterns
    # BERT/RoBERTa/XLM-R based models don't support flash-attn well
    if any(x in name_lower for x in ["bert", "roberta", "xlm", "e5-large", "bge-m3", "koe5"]):
        config.supports_flash_attn = False
    # Qwen, Mistral, Gemma, LLaMA based models support flash-attn
    elif any(x in name_lower for x in ["qwen", "mistral", "gemma", "llama", "snowflake", "arctic"]):
        config.supports_flash_attn = True

    # MRL inference
    if any(x in name_lower for x in ["nomic", "jina", "snowflake", "arctic", "qwen3-embedding"]):
        config.supports_mrl = True

    # Custom prompts for known patterns
    if "koe5" in name_lower or "ku-hiai-ontheit" in name_lower:
        config.custom_prompts = {
            PromptType.query.value: "query: ",
            PromptType.document.value: "passage: ",
        }
    elif any(x in name_lower for x in ["snowflake", "pixie", "arctic"]):
        config.custom_prompts = {
            PromptType.query.value: "query: ",
        }
    elif "frony" in name_lower:
        config.custom_prompts = {
            PromptType.query.value: "<Q>",
            PromptType.document.value: "<P>",
        }

    return config
