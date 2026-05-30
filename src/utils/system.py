"""System detection utilities for AMD GPUs and ROCm."""

import subprocess
import platform
import re
from typing import Optional
from dataclasses import dataclass


@dataclass
class GPUInfo:
    """AMD GPU information."""
    id: int
    name: str
    vram_gb: float
    uuid: str
    temperature: Optional[float] = None
    power_draw: Optional[float] = None


def detect_gpus() -> list[dict]:
    """Detect AMD GPUs via rocm-smi.

    Returns:
        List of GPU info dicts with id, name, vram_gb, uuid.
    """
    try:
        result = subprocess.run(
            ["rocm-smi", "--showproductname", "--showmeminfo", "vram", "--json"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode != 0:
            return _detect_gpus_fallback()

        import json
        data = json.loads(result.stdout)
        gpus = []
        for card_id, card_data in data.items():
            if not card_id.startswith("card"):
                continue
            gpu_id = int(card_id.replace("card", ""))
            name = card_data.get("Card Series", "Unknown AMD GPU")
            vram_total = card_data.get("VRAM Total Memory (B)", 0)
            vram_gb = round(int(vram_total) / (1024**3), 1) if vram_total else 0
            gpus.append({
                "id": gpu_id,
                "name": name,
                "vram_gb": vram_gb,
                "uuid": card_data.get("Card UUID", f"gpu-{gpu_id}"),
            })
        return gpus
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return _detect_gpus_fallback()


def _detect_gpus_fallback() -> list[dict]:
    """Fallback GPU detection using /sys or lspci."""
    try:
        result = subprocess.run(
            ["lspci", "-nn"], capture_output=True, text=True, timeout=5
        )
        gpus = []
        gpu_id = 0
        for line in result.stdout.splitlines():
            if "AMD" in line and ("Display" in line or "VGA" in line):
                name_match = re.search(r"AMD.*?\[(.+?)\]", line)
                name = name_match.group(1) if name_match else "AMD GPU"
                gpus.append({
                    "id": gpu_id,
                    "name": name,
                    "vram_gb": 0,
                    "uuid": f"gpu-{gpu_id}",
                })
                gpu_id += 1
        return gpus
    except Exception:
        return []


def get_rocm_version() -> str:
    """Get installed ROCm version string."""
    for cmd in [
        ["rocm-smi", "--version"],
        ["cat", "/opt/rocm/.info/version"],
    ]:
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                version_match = re.search(r"(\d+\.\d+(?:\.\d+)?)", result.stdout)
                if version_match:
                    return version_match.group(1)
        except Exception:
            continue

    try:
        result = subprocess.run(
            ["dpkg", "-l"], capture_output=True, text=True, timeout=5
        )
        for line in result.stdout.splitlines():
            if "rocm-dev" in line or "rocm-core" in line:
                parts = line.split()
                for p in parts:
                    if re.match(r"\d+\.\d+", p):
                        return p
    except Exception:
        pass

    return "not found"


def get_system_info() -> dict:
    """Collect system information for benchmark metadata."""
    info = {
        "OS": f"{platform.system()} {platform.release()}",
        "Architecture": platform.machine(),
        "Python": platform.python_version(),
        "ROCm": get_rocm_version(),
    }

    try:
        with open("/proc/cpuinfo") as f:
            cpu_lines = f.readlines()
        for line in cpu_lines:
            if "model name" in line:
                info["CPU"] = line.split(":")[1].strip()
                break
    except Exception:
        info["CPU"] = "unknown"

    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if "MemTotal" in line:
                    kb = int(line.split()[1])
                    info["RAM"] = f"{kb // (1024**1)} GB"
                    break
    except Exception:
        info["RAM"] = "unknown"

    return info
