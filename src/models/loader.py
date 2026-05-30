"""Model loading and quantization setup for vLLM and SGLang."""

import logging
from dataclasses import dataclass
from typing import Any, Optional

from src.models.registry import get_model_spec, ModelSpec

logger = logging.getLogger(__name__)


@dataclass
class LoadedModel:
    """Container for a loaded model and its metadata."""
    name: str
    framework: str
    quantization: str
    spec: ModelSpec
    engine: Any  # vLLM LLM or SGLang Engine
    backend: str = ""


class ModelLoader:
    """Load LLMs for inference on AMD GPUs via ROCm.

    Supports vLLM and SGLang backends with FP16, GPTQ, and AWQ quantization.
    """

    def __init__(self, device: str = "cuda", trust_remote_code: bool = True):
        self.device = device
        self.trust_remote_code = trust_remote_code
        self._loaded: dict[str, LoadedModel] = {}

    def load(self, model_name: str, framework: str = "vllm",
             quantization: str = "fp16", **kwargs) -> LoadedModel:
        """Load a model with the specified framework and quantization.

        Args:
            model_name: HuggingFace model name or local path.
            framework: Backend to use ('vllm' or 'sglang').
            quantization: Quantization method ('fp16', 'gptq', 'awq').
            **kwargs: Additional framework-specific arguments.

        Returns:
            LoadedModel instance with the loaded engine.

        Raises:
            ImportError: If the framework is not installed.
            RuntimeError: If model loading fails.
        """
        cache_key = f"{model_name}:{framework}:{quantization}"
        if cache_key in self._loaded:
            logger.info(f"Returning cached model: {cache_key}")
            return self._loaded[cache_key]

        spec = get_model_spec(model_name)
        logger.info(f"Loading {model_name} with {framework} ({quantization})")

        if framework == "vllm":
            engine = self._load_vllm(model_name, quantization, spec, **kwargs)
        elif framework == "sglang":
            engine = self._load_sglang(model_name, quantization, spec, **kwargs)
        else:
            raise ValueError(f"Unsupported framework: {framework}")

        loaded = LoadedModel(
            name=model_name,
            framework=framework,
            quantization=quantization,
            spec=spec,
            engine=engine,
            backend=framework,
        )
        self._loaded[cache_key] = loaded
        logger.info(f"Model loaded successfully: {model_name}")
        return loaded

    def _load_vllm(self, model_name: str, quantization: str,
                   spec: ModelSpec, **kwargs) -> Any:
        """Load model via vLLM."""
        try:
            from vllm import LLM
        except ImportError:
            raise ImportError("vLLM not installed. Install with: pip install vllm")

        llm_kwargs = {
            "model": model_name,
            "trust_remote_code": self.trust_remote_code,
            "max_model_len": min(spec.context_length, 4096),
            "dtype": "auto",
            **kwargs,
        }

        if quantization == "gptq":
            llm_kwargs["quantization"] = "gptq"
        elif quantization == "awq":
            llm_kwargs["quantization"] = "awq"

        return LLM(**llm_kwargs)

    def _load_sglang(self, model_name: str, quantization: str,
                     spec: ModelSpec, **kwargs) -> Any:
        """Load model via SGLang."""
        try:
            import sglang as sgl
        except ImportError:
            raise ImportError("SGLang not installed. Install with: pip install sglang")

        runtime_kwargs = {
            "model_path": model_name,
            "trust_remote_code": self.trust_remote_code,
            **kwargs,
        }

        if quantization == "gptq":
            runtime_kwargs["quantization"] = "gptq"
        elif quantization == "awq":
            runtime_kwargs["quantization"] = "awq"

        return sgl.Engine(**runtime_kwargs)

    def unload(self, model_name: str, framework: str = "vllm",
               quantization: str = "fp16"):
        """Unload a model from memory."""
        cache_key = f"{model_name}:{framework}:{quantization}"
        if cache_key in self._loaded:
            del self._loaded[cache_key]
            logger.info(f"Unloaded: {cache_key}")

    def unload_all(self):
        """Unload all cached models."""
        self._loaded.clear()
        logger.info("All models unloaded")
