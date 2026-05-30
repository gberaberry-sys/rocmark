"""Unit tests for ModelLoader."""

import pytest
from unittest.mock import patch, MagicMock

from src.models.loader import ModelLoader, LoadedModel
from src.models.registry import get_model_spec, MODEL_REGISTRY, ModelSpec


class TestModelRegistry:
    """Tests for model registry."""

    def test_registry_not_empty(self):
        assert len(MODEL_REGISTRY) > 0

    def test_get_known_model(self):
        spec = get_model_spec("meta-llama/Meta-Llama-3-8B")
        assert spec.name == "Meta Llama 3 8B"
        assert spec.params_billions == 8.0
        assert "fp16" in spec.quantizations

    def test_get_unknown_model_returns_default(self):
        spec = get_model_spec("unknown/model-xyz")
        assert spec.name == "model-xyz"
        assert "fp16" in spec.quantizations

    def test_partial_match(self):
        spec = get_model_spec("some-org/Meta-Llama-3-8B")
        assert "Llama" in spec.name

    def test_list_models(self):
        models = list_models()
        assert len(models) > 0
        assert all(isinstance(m, str) for m in models)

    def test_model_spec_dataclass(self):
        spec = ModelSpec(
            name="Test", params_billions=1.0, context_length=2048
        )
        assert spec.quantizations == ["fp16"]
        assert spec.vram_fp16_gb == 0.0


class TestModelLoader:
    """Tests for ModelLoader."""

    def test_init(self):
        loader = ModelLoader()
        assert loader.device == "cuda"
        assert loader.trust_remote_code is True
        assert loader._loaded == {}

    def test_init_custom(self):
        loader = ModelLoader(device="cpu", trust_remote_code=False)
        assert loader.device == "cpu"
        assert loader.trust_remote_code is False

    def test_cache_returns_same_model(self):
        loader = ModelLoader()
        mock_model = MagicMock()
        loader._loaded["test:vllm:fp16"] = mock_model
        result = loader.load("test", framework="vllm", quantization="fp16")
        assert result is mock_model

    def test_unload(self):
        loader = ModelLoader()
        loader._loaded["test:vllm:fp16"] = MagicMock()
        loader.unload("test", "vllm", "fp16")
        assert "test:vllm:fp16" not in loader._loaded

    def test_unload_all(self):
        loader = ModelLoader()
        loader._loaded["a:vllm:fp16"] = MagicMock()
        loader._loaded["b:sglang:gptq"] = MagicMock()
        loader.unload_all()
        assert len(loader._loaded) == 0

    def test_unsupported_framework_raises(self):
        loader = ModelLoader()
        with pytest.raises(ValueError, match="Unsupported framework"):
            loader.load("test", framework="tensorflow")
