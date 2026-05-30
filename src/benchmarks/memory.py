"""Memory profiling — VRAM tracking via rocm-smi."""

import subprocess
import threading
import time
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class MemoryResult:
    """Memory profiling results."""
    peak_gb: float
    avg_gb: float
    min_gb: float
    snapshots: list[dict] = field(default_factory=list)
    num_gpus: int = 0


class MemoryProfiler:
    """Track GPU VRAM usage during benchmark execution.

    Runs rocm-smi in a background thread to capture VRAM
    snapshots at regular intervals.
    """

    def __init__(self, interval: float = 1.0, gpu_ids: list[int] | None = None):
        self.interval = interval
        self.gpu_ids = gpu_ids or [0]
        self._snapshots: list[dict] = []
        self._running = False
        self._thread: threading.Thread | None = None

    def start(self):
        """Start memory profiling in background thread."""
        self._snapshots = []
        self._running = True
        self._thread = threading.Thread(target=self._profile_loop, daemon=True)
        self._thread.start()
        logger.info("Memory profiler started")

    def stop(self):
        """Stop memory profiling."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        logger.info(f"Memory profiler stopped ({len(self._snapshots)} snapshots)")

    def get_results(self) -> MemoryResult:
        """Get profiling results.

        Returns:
            MemoryResult with peak, average, and snapshot data.
        """
        if not self._snapshots:
            return MemoryResult(peak_gb=0, avg_gb=0, min_gb=0, num_gpus=len(self.gpu_ids))

        for gpu_id in self.gpu_ids:
            gpu_values = [s["used_gb"] for s in self._snapshots if s["gpu_id"] == gpu_id]
            if not gpu_values:
                continue
            return MemoryResult(
                peak_gb=max(gpu_values),
                avg_gb=sum(gpu_values) / len(gpu_values),
                min_gb=min(gpu_values),
                snapshots=self._snapshots,
                num_gpus=len(self.gpu_ids),
            )

        return MemoryResult(peak_gb=0, avg_gb=0, min_gb=0)

    def _profile_loop(self):
        """Background profiling loop."""
        while self._running:
            try:
                self._snapshot()
            except Exception as e:
                logger.debug(f"Snapshot failed: {e}")
            time.sleep(self.interval)

    def _snapshot(self):
        """Take a VRAM snapshot via rocm-smi."""
        try:
            result = subprocess.run(
                ["rocm-smi", "--showmeminfo", "vram", "--json"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode != 0:
                return

            import json
            data = json.loads(result.stdout)
            timestamp = time.time()

            for card_id, card_data in data.items():
                if not card_id.startswith("card"):
                    continue
                gpu_id = int(card_id.replace("card", ""))
                if gpu_id not in self.gpu_ids:
                    continue

                used = int(card_data.get("VRAM Total Used Memory (B)", 0))
                total = int(card_data.get("VRAM Total Memory (B)", 0))

                self._snapshots.append({
                    "timestamp": timestamp,
                    "gpu_id": gpu_id,
                    "used_gb": round(used / (1024**3), 2),
                    "total_gb": round(total / (1024**3), 2),
                    "utilization_pct": round(used / total * 100, 1) if total > 0 else 0,
                })
        except (FileNotFoundError, subprocess.TimeoutExpired, json.JSONDecodeError):
            pass
