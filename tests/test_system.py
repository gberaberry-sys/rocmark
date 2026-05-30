"""Unit tests for system detection utilities."""

import pytest
from unittest.mock import patch, MagicMock

from src.utils.system import (
    detect_gpus, get_rocm_version, get_system_info, GPUInfo
)


class TestDetectGPUs:
    """Tests for GPU detection."""

    def test_returns_list(self):
        result = detect_gpus()
        assert isinstance(result, list)

    @patch("subprocess.run")
    def test_parses_rocm_smi_output(self, mock_run):
        import json
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=json.dumps({
                "card0": {
                    "Card Series": "AMD Instinct MI210",
                    "VRAM Total Memory (B)": "34359738368",
                    "Card UUID": "gpu-0-uuid"
                }
            })
        )
        gpus = detect_gpus()
        assert len(gpus) == 1
        assert gpus[0]["name"] == "AMD Instinct MI210"
        assert gpus[0]["vram_gb"] == 32.0

    @patch("subprocess.run")
    def test_handles_failure(self, mock_run):
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="error")
        gpus = detect_gpus()
        assert isinstance(gpus, list)


class TestGetRocmVersion:
    """Tests for ROCm version detection."""

    def test_returns_string(self):
        result = get_rocm_version()
        assert isinstance(result, str)


class TestGetSystemInfo:
    """Tests for system info collection."""

    def test_returns_dict(self):
        info = get_system_info()
        assert isinstance(info, dict)
        assert "OS" in info
        assert "Python" in info

    def test_has_required_keys(self):
        info = get_system_info()
        for key in ["OS", "Architecture", "Python", "ROCm"]:
            assert key in info, f"Missing key: {key}"
