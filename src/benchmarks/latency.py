"""Latency benchmark — P50/P95/P99 distribution."""

import time
import logging
import statistics
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class LatencyResult:
    """Latency benchmark results."""
    p50_ms: float
    p95_ms: float
    p99_ms: float
    mean_ms: float
    min_ms: float
    max_ms: float
    stdev_ms: float
    num_requests: int
    all_latencies_ms: list[float]


class LatencyBenchmark:
    """Measure per-request inference latency distribution.

    Runs individual requests sequentially and collects
    latency measurements for percentile analysis.
    """

    def __init__(self, model_info, framework: str = "vllm"):
        self.model = model_info
        self.framework = framework

    def run(self, prompts: list[str], max_tokens: int = 512,
            temperature: float = 0.0) -> LatencyResult:
        """Run latency benchmark.

        Args:
            prompts: List of input prompts.
            max_tokens: Maximum tokens per generation.
            temperature: Sampling temperature.

        Returns:
            LatencyResult with percentile metrics.
        """
        logger.info(f"Latency benchmark: {len(prompts)} requests")
        latencies = []

        for i, prompt in enumerate(prompts):
            start = time.perf_counter()
            self._generate_single(prompt, max_tokens, temperature)
            latency_ms = (time.perf_counter() - start) * 1000
            latencies.append(latency_ms)

            if (i + 1) % 10 == 0:
                logger.info(f"  Completed {i + 1}/{len(prompts)} requests")

        sorted_lat = sorted(latencies)
        n = len(sorted_lat)

        return LatencyResult(
            p50_ms=sorted_lat[int(n * 0.50)] if n > 0 else 0,
            p95_ms=sorted_lat[int(n * 0.95)] if n > 0 else 0,
            p99_ms=sorted_lat[int(n * 0.99)] if n > 0 else 0,
            mean_ms=statistics.mean(latencies) if latencies else 0,
            min_ms=min(latencies) if latencies else 0,
            max_ms=max(latencies) if latencies else 0,
            stdev_ms=statistics.stdev(latencies) if len(latencies) > 1 else 0,
            num_requests=len(prompts),
            all_latencies_ms=latencies,
        )

    def _generate_single(self, prompt: str, max_tokens: int, temperature: float):
        """Generate a single response."""
        if self.framework == "vllm":
            from vllm import SamplingParams
            params = SamplingParams(temperature=temperature, max_tokens=max_tokens)
            self.model.engine.generate([prompt], params)
        elif self.framework == "sglang":
            self.model.engine.generate({
                "prompt": prompt, "max_tokens": max_tokens,
                "temperature": temperature
            })
