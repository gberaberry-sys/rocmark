# ROCmark — ROCm GPU Benchmark Suite

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![ROCm 6.x](https://img.shields.io/badge/ROCm-6.x-orange.svg)](https://rocm.docs.amd.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**ROCmark** is a lightweight benchmarking toolkit for evaluating LLM inference performance on AMD GPUs via ROCm. It measures throughput, latency, and memory efficiency across multiple serving frameworks (vLLM, SGLang) and quantization backends (GPTQ, AWQ, FP16).

Built for researchers and developers who need reproducible GPU benchmarks without vendor lock-in.

---

## Why ROCmark?

Most LLM benchmarks target NVIDIA exclusively. ROCmark fills the gap for AMD MI-series and Radeon GPUs by providing:

- **Framework parity** — identical test harness for vLLM and SGLang
- **Quantization comparison** — FP16 vs GPTQ vs AWQ on the same hardware
- **Memory profiling** — VRAM usage snapshots at each inference stage
- **Reproducible reports** — JSON + Markdown output with full system metadata

---

## Architecture

```
rocmark/
├── src/
│   ├── benchmarks/
│   │   ├── throughput.py      # Tokens/sec, time-to-first-token
│   │   ├── latency.py         # P50/P95/P99 latency distribution
│   │   └── memory.py          # VRAM tracking via rocm-smi
│   ├── models/
│   │   ├── loader.py          # Model download + quantization setup
│   │   └── registry.py        # Supported model catalog
│   ├── utils/
│   │   ├── system.py          # GPU detection, ROCm version check
│   │   ├── prompts.py         # Standardized test prompt sets
│   │   └── logging.py         # Structured benchmark logging
│   └── reports/
│       ├── generator.py       # JSON + Markdown report builder
│       └── templates/
│           └── report.md.j2   # Jinja2 report template
├── main.py                    # CLI entrypoint
├── config.yaml                # Default benchmark config
├── requirements.txt
├── .env.example
├── tests/
│   ├── test_loader.py
│   └── test_benchmarks.py
└── LICENSE
```

---

## Quick Start

### Prerequisites

- AMD GPU with ROCm 6.x installed ([install guide](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/))
- Python 3.10+
- vLLM or SGLang installed

### Installation

```bash
git clone https://github.com/gberaberry-sys/rocmark.git
cd rocmark
pip install -r requirements.txt
cp .env.example .env
```

### Run a Benchmark

```bash
# Benchmark Llama 3 8B on vLLM with FP16
python main.py benchmark \
  --model meta-llama/Meta-Llama-3-8B \
  --framework vllm \
  --quantization fp16 \
  --num-prompts 100 \
  --output reports/

# Compare frameworks
python main.py compare \
  --model meta-llama/Meta-Llama-3-8B \
  --frameworks vllm sglang \
  --quantizations fp16 gptq \
  --output reports/comparison/
```

### Generate Report

```bash
python main.py report --input reports/ --format markdown
```

---

## Supported Models

| Model | Parameters | FP16 | GPTQ | AWQ |
|-------|-----------|------|------|-----|
| Llama 3 8B | 8B | ✅ | ✅ | ✅ |
| Llama 3 70B | 70B | ✅ | ✅ | ✅ |
| Mistral 7B | 7B | ✅ | ✅ | ✅ |
| Qwen2 7B | 7B | ✅ | ✅ | ✅ |
| DeepSeek V2 | 236B | ✅ | — | ✅ |

---

## Sample Results

Tested on AMD Instinct MI210 (32GB VRAM), ROCm 6.2, Ubuntu 22.04:

| Model | Framework | Quant | Throughput (tok/s) | TTFT (ms) | VRAM (GB) |
|-------|-----------|-------|-------------------|-----------|-----------|
| Llama 3 8B | vLLM | FP16 | 42.3 | 89 | 16.2 |
| Llama 3 8B | vLLM | GPTQ-4bit | 67.8 | 52 | 5.8 |
| Llama 3 8B | SGLang | FP16 | 45.1 | 76 | 15.9 |
| Llama 3 8B | SGLang | AWQ-4bit | 71.2 | 48 | 5.6 |
| Mistral 7B | vLLM | FP16 | 48.7 | 71 | 14.8 |

---

## Configuration

Edit `config.yaml` to customize:

```yaml
benchmark:
  num_prompts: 100
  max_tokens: 512
  temperature: 0.0
  warmup_requests: 5

models:
  - name: meta-llama/Meta-Llama-3-8B
    frameworks: [vllm, sglang]
    quantizations: [fp16, gptq, awq]

system:
  rocm_smi_interval: 1.0  # seconds between VRAM snapshots
  output_dir: ./reports
```

---

## Contributing

1. Fork the repo
2. Create a feature branch (`git checkout -b feat/new-benchmark`)
3. Commit your changes (`git commit -m 'add new benchmark'`)
4. Push to the branch (`git push origin feat/new-benchmark`)
5. Open a Pull Request

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Acknowledgments

- [ROCm](https://rocm.docs.amd.com/) — AMD's open-source GPU computing platform
- [vLLM](https://github.com/vllm-project/vllm) — High-throughput LLM serving
- [SGLang](https://github.com/sgl-project/sglang) — Fast serving framework for LLMs
- AMD AI Developer Program — Cloud credits for benchmarking
