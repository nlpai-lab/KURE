"""Unified model loader with bf16 + flash-attn + MRL support."""

from __future__ import annotations

import logging
import os

import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.models import StaticEmbedding

import mteb
from mteb import SentenceTransformerEncoderWrapper

from eval.models.config import ModelConfig, get_model_config

logger = logging.getLogger(__name__)


def _is_flash_attn_error(error: Exception) -> bool:
	"""Check if error is related to flash attention not being supported."""
	error_str = str(error).lower()
	return any(x in error_str for x in [
		"flash attention",
		"flash_attention",
		"flash-attention",
		"does not support flash attention",
		"flashattention",
		"is not installed",
	])


def _get_hf_model(model: mteb.EncoderProtocol):
	"""Navigate wrapper layers to find the actual HuggingFace model."""
	# MTEB wrapper -> .model (SentenceTransformer)
	inner = getattr(model, "model", model)
	# SentenceTransformer -> first module -> .auto_model
	if hasattr(inner, "_first_module"):
		transformer = inner._first_module()
		return getattr(transformer, "auto_model", None)
	if hasattr(inner, "auto_model"):
		return inner.auto_model
	return None


def _disable_use_cache(model: mteb.EncoderProtocol, model_name: str) -> None:
	"""Disable KV cache to avoid DynamicCache compatibility issues."""
	hf_model = _get_hf_model(model)
	if hf_model is not None and hasattr(hf_model, "config"):
		hf_model.config.use_cache = False
		logger.info(f"Disabled use_cache for {model_name}")
	else:
		logger.warning(f"Could not disable use_cache for {model_name}: HF model not found")


def _apply_bf16(model: mteb.EncoderProtocol, model_name: str) -> None:
	"""Apply bfloat16 to model loaded via mteb.get_model()."""
	if hasattr(model, "model") and hasattr(model.model, "to"):
		model.model.to(dtype=torch.bfloat16)
		logger.info(f"Applied bfloat16 to {model_name}")
	else:
		logger.warning(f"Could not apply bf16 to {model_name}: no .model attribute")


def _load_sentence_transformer(
	model_name: str,
	device: str | torch.device,
	model_kwargs: dict,
	truncate_dim: int | None,
	trust_remote_code: bool = True,
) -> SentenceTransformer:
	"""Load SentenceTransformer with given kwargs."""
	st_kwargs = {
		"model_name_or_path": model_name,
		"device": device,
		"trust_remote_code": trust_remote_code,
	}

	if model_kwargs:
		st_kwargs["model_kwargs"] = model_kwargs

	if truncate_dim is not None:
		st_kwargs["truncate_dim"] = truncate_dim

	return SentenceTransformer(**st_kwargs)


def load_model(
	model_name: str,
	device: str | torch.device = "cuda",
	use_bf16: bool = True,
	use_flash_attn: bool = True,
	truncate_dim: int | None = None,
) -> tuple[mteb.EncoderProtocol, int]:
	"""
	Load an embedding model with optimized settings.

	Automatically falls back from flash attention to SDPA if flash attention
	is not supported by the model.

	Args:
		model_name: HuggingFace model name or local path
		device: Device to load model on
		use_bf16: Whether to use bfloat16 precision (default: True)
		use_flash_attn: Whether to try flash attention 2 (default: True)
		truncate_dim: MRL truncation dimension (None = full dimension)

	Returns:
		Tuple of (MTEB-compatible model, batch_size)
	"""
	config = get_model_config(model_name)

	# Validate MRL dimension
	if truncate_dim is not None:
		if not config.supports_mrl:
			logger.warning(
				f"{model_name} does not support MRL (Matryoshka), ignoring truncate_dim={truncate_dim}"
			)
			truncate_dim = None
		elif config.mrl_dims and truncate_dim not in config.mrl_dims:
			logger.warning(
				f"truncate_dim={truncate_dim} not in supported dims {config.mrl_dims} for {model_name}"
			)

	# Handle model2vec models
	if "m2v" in model_name.lower():
		static_embedding = StaticEmbedding.from_model2vec(model_name)
		model = SentenceTransformer(modules=[static_embedding], device=device)
		return SentenceTransformerEncoderWrapper(model), config.batch_size

	# Check if local model path
	is_local = os.path.isdir(model_name)

	# Try to load with MTEB's registered model first (handles prompts automatically)
	try:
		if not is_local and config.custom_prompts is None:
			mteb_kwargs = {}
			if use_bf16 and config.supports_bf16:
				mteb_kwargs["model_kwargs"] = {"torch_dtype": torch.bfloat16}
			model = mteb.get_model(model_name, device=device, **mteb_kwargs)
			_disable_use_cache(model, model_name)
			if use_bf16 and config.supports_bf16:
				_apply_bf16(model, model_name)
			logger.info(f"Loaded {model_name} via MTEB (prompts auto-configured)")
			return model, config.batch_size
	except Exception as e:
		logger.debug(f"MTEB get_model failed for {model_name}: {e}, falling back to manual loading")

	# Build model_kwargs for optimizations
	def _build_model_kwargs(try_flash_attn: bool) -> dict:
		kwargs = {}
		if use_bf16 and config.supports_bf16:
			kwargs["torch_dtype"] = torch.bfloat16
		if try_flash_attn:
			kwargs["attn_implementation"] = "flash_attention_2"
		return kwargs

	# Try loading with flash attention first, fallback to SDPA if it fails
	if use_flash_attn:
		model_kwargs = _build_model_kwargs(try_flash_attn=True)
		try:
			logger.info(f"Trying flash_attention_2 for {model_name}")
			model = _load_sentence_transformer(
				model_name, device, model_kwargs, truncate_dim
			)
			logger.info(f"Loaded {model_name} with flash_attention_2")
		except Exception as e:
			if _is_flash_attn_error(e):
				logger.warning(
					f"Flash attention not supported for {model_name}, falling back to SDPA"
				)
				# Retry without flash attention
				model_kwargs = _build_model_kwargs(try_flash_attn=False)
				model = _load_sentence_transformer(
					model_name, device, model_kwargs, truncate_dim
				)
				logger.info(f"Loaded {model_name} with SDPA (fallback)")
			else:
				raise
	else:
		# Load without flash attention
		model_kwargs = _build_model_kwargs(try_flash_attn=False)
		model = _load_sentence_transformer(
			model_name, device, model_kwargs, truncate_dim
		)
		logger.info(f"Loaded {model_name} with SDPA")

	# Disable KV cache to avoid DynamicCache compatibility issues
	_disable_use_cache(model, model_name)

	if use_bf16 and config.supports_bf16:
		logger.info(f"Using bfloat16 for {model_name}")

	if truncate_dim is not None:
		logger.info(f"Using MRL truncation to {truncate_dim} dimensions")

	# Wrap with prompts if needed
	if config.custom_prompts:
		logger.info(f"Applying custom prompts for {model_name}: {config.custom_prompts}")
		wrapped_model = SentenceTransformerEncoderWrapper(
			model=model,
			model_prompts=config.custom_prompts,
		)
		return wrapped_model, config.batch_size

	# Return MTEB-compatible wrapper
	return SentenceTransformerEncoderWrapper(model), config.batch_size


def get_output_folder(model_name: str, truncate_dim: int | None = None) -> str:
	"""
	Generate output folder name for results.

	Args:
		model_name: Model name or path
		truncate_dim: MRL truncation dimension

	Returns:
		Sanitized folder name for results
	"""
	# Sanitize model name for filesystem
	folder_name = model_name.replace("/", "_")

	# Add MRL dimension suffix if truncated
	if truncate_dim is not None:
		folder_name = f"{folder_name}_dim{truncate_dim}"

	return folder_name
