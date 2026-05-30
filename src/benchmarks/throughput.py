"""Throughput benchmark — tokens/sec and time-to-first-token."""

import time
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ThroughputResult:
    """Throughput benchmark results."""
    tokens_per_sec: float
    ttft_ms: float
    total_tokens: int
    total_time_sec: float
    num_requests: int
    avg_tokens_per_request: float = 0.0
    framework: str = ""
    model: str = ""
    quantization: str = ""

    def __post_init__(self):
        if self.num_requests > 0:
            self.avg_tokens_per_request = self.total_tokens / self.num_requests


class ThroughputBenchmark:
    """Measure LLM inference throughput.

    Benchmarks tokens/sec and time-to-first-token (TTFT) using
    either vLLM or SGLang batch inference.
    """

    def __init__(self, model_info, framework: str = "vllm"):
        self.model = model_info
        self.framework = framework

    def run(self, prompts: list[str], max_tokens: int = 512,
            temperature: float = 0.0, warmup: int = 5) -> ThroughputResult:
        """Run throughput benchmark.

        Args:
            prompts: List of input prompts.
            max_tokens: Maximum tokens to generate per prompt.
            temperature: Sampling temperature (0 = greedy).
            warmup: Number of warmup requests before measurement.

        Returns:
            ThroughputResult with benchmark metrics.
        """
        logger.info(f"Throughput benchmark: {len(prompts)} prompts, warmup={warmup}")

        # Warmup
        if warmup > 0:
            logger.info(f"Warming up with {warmup} requests...")
            self._generate(prompts[:warmup], max_tokens, temperature)

        # Actual benchmark
        start = time.perf_counter()
        ttft_start = time.perf_counter()
        total_tokens = 0
        first_token_time = None

        outputs = self._generate(prompts, max_tokens, temperature)

        for output in outputs:
            tokens_generated = len(output.get("token_ids", []))
            total_tokens += tokens_generated
            if first_token_time is None:
                first_token_time = time.perf_counter()

        elapsed = time.perf_counter() - start
        ttft = ((first_token_time or start) - ttft_start) * 1000

        tps = total_tokens / elapsed if elapsed > 0 else 0

        return ThroughputResult(
            tokens_per_sec=tps,
            ttft_ms=ttft,
            total_tokens=total_tokens,
            total_time_sec=elapsed,
            num_requests=len(prompts),
            framework=self.framework,
            model=self.model.name,
            quantization=self.model.quantization,
        )

    def _generate(self, prompts: list[str], max_tokens: int,
                  temperature: float) -> list[dict]:
        """Generate outputs using the loaded model engine."""
        if self.framework == "vllm":
            return self._generate_vllm(prompts, max_tokens, temperature)
        elif self.framework == "sglang":
            return self._generate_sglang(prompts, max_tokens, temperature)
        else:
            raise ValueError(f"Unknown framework: {self.framework}")

    def _generate_vllm(self, prompts, max_tokens, temperature) -> list[dict]:
        """Generate via vLLM."""
        from vllm import SamplingParams
        params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
        )
        outputs = self.model.engine.generate(prompts, params)
        return [{"token_ids": o.outputs[0].token_ids, "text": o.outputs[0].text}
                for o in outputs]

    def _generate_sglang(self, prompts, max_tokens, temperature) -> list[dict]:
        """Generate via SGLang."""
        requests = [{"prompt": p, "max_tokens": max_tokens,
                     "temperature": temperature} for p in prompts]
        outputs = self.model.engine.generate(requests)
        return [{"token_ids": o.get("token_ids", []), "text": o.get("text", "")}
                for o in outputs]
