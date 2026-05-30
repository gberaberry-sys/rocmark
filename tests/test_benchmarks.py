"""Unit tests for benchmark classes."""

import pytest
from unittest.mock import MagicMock

from src.benchmarks.throughput import ThroughputBenchmark, ThroughputResult
from src.benchmarks.latency import LatencyBenchmark, LatencyResult
from src.benchmarks.memory import MemoryProfiler, MemoryResult


class TestThroughputResult:
    """Tests for ThroughputResult dataclass."""

    def test_basic_creation(self):
        result = ThroughputResult(
            tokens_per_sec=42.3, ttft_ms=89.0, total_tokens=1000,
            total_time_sec=23.6, num_requests=10
        )
        assert result.tokens_per_sec == 42.3
        assert result.avg_tokens_per_request == 100.0

    def test_zero_requests(self):
        result = ThroughputResult(
            tokens_per_sec=0, ttft_ms=0, total_tokens=0,
            total_time_sec=0, num_requests=0
        )
        assert result.avg_tokens_per_request == 0


class TestLatencyResult:
    """Tests for LatencyResult dataclass."""

    def test_percentiles(self):
        latencies = [float(i) for i in range(100)]
        result = LatencyResult(
            p50_ms=50.0, p95_ms=95.0, p99_ms=99.0,
            mean_ms=49.5, min_ms=0, max_ms=99, stdev_ms=28.87,
            num_requests=100, all_latencies_ms=latencies
        )
        assert result.p50_ms < result.p95_ms < result.p99_ms


class TestMemoryProfiler:
    """Tests for MemoryProfiler."""

    def test_init(self):
        profiler = MemoryProfiler(interval=0.5, gpu_ids=[0, 1])
        assert profiler.interval == 0.5
        assert profiler.gpu_ids == [0, 1]

    def test_empty_results(self):
        profiler = MemoryProfiler()
        result = profiler.get_results()
        assert result.peak_gb == 0
        assert result.avg_gb == 0

    def test_manual_snapshots(self):
        profiler = MemoryProfiler()
        profiler._snapshots = [
            {"gpu_id": 0, "used_gb": 8.0, "total_gb": 16.0, "timestamp": 1.0, "utilization_pct": 50.0},
            {"gpu_id": 0, "used_gb": 12.0, "total_gb": 16.0, "timestamp": 2.0, "utilization_pct": 75.0},
        ]
        result = profiler.get_results()
        assert result.peak_gb == 12.0
        assert result.avg_gb == 10.0
        assert result.min_gb == 8.0
