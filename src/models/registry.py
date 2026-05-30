"""Model registry with metadata for supported models."""

from dataclasses import dataclass, field


@dataclass
class ModelSpec:
    """Specification for a supported model."""
    name: str
    params_billions: float
    context_length: int
    quantizations: list[str] = field(default_factory=lambda: ["fp16"])
    vram_fp16_gb: float = 0.0
    vram_gptq_gb: float = 0.0
    vram_awq_gb: float = 0.0
    architectures: list[str] = field(default_factory=lambda: ["llama"])
    license: str = "apache-2.0"
    description: str = ""


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "meta-llama/Meta-Llama-3-8B": ModelSpec(
        name="Meta Llama 3 8B",
        params_billions=8.0,
        context_length=8192,
        quantizations=["fp16", "gptq", "awq"],
        vram_fp16_gb=16.0,
        vram_gptq_gb=5.8,
        vram_awq_gb=5.6,
        architectures=["llama"],
        license="llama3",
        description="Meta's latest 8B parameter model with improved reasoning.",
    ),
    "meta-llama/Meta-Llama-3-70B": ModelSpec(
        name="Meta Llama 3 70B",
        params_billions=70.0,
        context_length=8192,
        quantizations=["fp16", "gptq", "awq"],
        vram_fp16_gb=140.0,
        vram_gptq_gb=38.0,
        vram_awq_gb=36.0,
        architectures=["llama"],
        license="llama3",
        description="Meta's flagship 70B model, best-in-class for its size.",
    ),
    "mistralai/Mistral-7B-v0.3": ModelSpec(
        name="Mistral 7B v0.3",
        params_billions=7.0,
        context_length=32768,
        quantizations=["fp16", "gptq", "awq"],
        vram_fp16_gb=14.0,
        vram_gptq_gb=5.2,
        vram_awq_gb=5.0,
        architectures=["mistral"],
        license="apache-2.0",
        description="Mistral AI's efficient 7B model with sliding window attention.",
    ),
    "Qwen/Qwen2-7B": ModelSpec(
        name="Qwen2 7B",
        params_billions=7.0,
        context_length=32768,
        quantizations=["fp16", "gptq", "awq"],
        vram_fp16_gb=14.0,
        vram_gptq_gb=5.0,
        vram_awq_gb=4.8,
        architectures=["qwen2"],
        license="apache-2.0",
        description="Alibaba's Qwen2 7B with strong multilingual capabilities.",
    ),
    "deepseek-ai/DeepSeek-V2": ModelSpec(
        name="DeepSeek V2",
        params_billions=236.0,
        context_length=128000,
        quantizations=["fp16", "awq"],
        vram_fp16_gb=400.0,
        vram_awq_gb=80.0,
        architectures=["deepseek_v2"],
        license="deepseek",
        description="DeepSeek V2 MoE model with 236B parameters.",
    ),
}


def get_model_spec(model_name: str) -> ModelSpec:
    """Get model specification by name.

    Args:
        model_name: HuggingFace model name or path.

    Returns:
        ModelSpec for the model.

    Raises:
        KeyError: If model is not in registry.
    """
    if model_name in MODEL_REGISTRY:
        return MODEL_REGISTRY[model_name]

    # Try partial match
    for key, spec in MODEL_REGISTRY.items():
        if model_name.split("/")[-1].lower() in key.lower():
            return spec

    # Return default for unknown models
    return ModelSpec(
        name=model_name.split("/")[-1],
        params_billions=0.0,
        context_length=4096,
        quantizations=["fp16"],
        description="Unregistered model — using default config.",
    )


def list_models() -> list[str]:
    """List all registered model names."""
    return list(MODEL_REGISTRY.keys())
