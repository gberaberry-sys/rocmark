#!/usr/bin/env python3
"""ROCmark — ROCm GPU Benchmark Suite for LLM Inference."""

import argparse
import sys
import yaml
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

from src.utils.system import detect_gpus, get_rocm_version, get_system_info
from src.models.registry import MODEL_REGISTRY
from src.benchmarks.throughput import ThroughputBenchmark
from src.benchmarks.latency import LatencyBenchmark
from src.benchmarks.memory import MemoryProfiler
from src.models.loader import ModelLoader
from src.reports.generator import ReportGenerator
from src.utils.logging import setup_logging

console = Console()

BANNER = """[bold cyan]
 ██████╗  ██████╗  ██████╗███╗   ███╗ █████╗ ██████╗ ██╗  ██╗
 ██╔══██╗██╔═══██╗██╔════╝████╗ ████║██╔══██╗██╔══██╗██║ ██╔╝
 ██████╔╝██║   ██║██║     ██╔████╔██║███████║██████╔╝█████╔╝
 ██╔══██╗██║   ██║██║     ██║╚██╔╝██║██╔══██║██╔══██╗██╔═██╗
 ██║  ██║╚██████╔╝╚██████╗██║ ╚═╝ ██║██║  ██║██║  ██║██║  ██╗
 ╚═╝  ╚═╝ ╚═════╝  ╚═════╝╚═╝     ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝[/bold cyan]
[dim]ROCm GPU Benchmark Suite for LLM Inference v1.0.0[/dim]
"""


def load_config(config_path: str = "config.yaml") -> dict:
    """Load benchmark configuration from YAML file."""
    path = Path(config_path)
    if not path.exists():
        console.print(f"[yellow]Config not found at {config_path}, using defaults[/yellow]")
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def cmd_detect(args):
    """Detect AMD GPUs and ROCm installation."""
    console.print(BANNER)
    console.print(Panel("[bold]System Detection[/bold]", style="blue"))

    info = get_system_info()
    table = Table(title="System Information", box=box.ROUNDED)
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")
    for key, val in info.items():
        table.add_row(key, str(val))
    console.print(table)

    gpus = detect_gpus()
    if not gpus:
        console.print("[red]No AMD GPUs detected. Ensure ROCm is installed.[/red]")
        sys.exit(1)

    gpu_table = Table(title="Detected GPUs", box=box.ROUNDED)
    gpu_table.add_column("ID", style="cyan")
    gpu_table.add_column("Name", style="green")
    gpu_table.add_column("VRAM", style="yellow")
    gpu_table.add_column("Status", style="bold")
    for gpu in gpus:
        gpu_table.add_row(str(gpu["id"]), gpu["name"], f'{gpu["vram_gb"]} GB', "[green]OK[/green]")
    console.print(gpu_table)

    rocm = get_rocm_version()
    console.print(f"
[bold]ROCm Version:[/bold] {rocm}")


def cmd_benchmark(args):
    """Run a single benchmark."""
    console.print(BANNER)
    config = load_config(args.config)

    console.print(Panel(f"[bold]Benchmarking {args.model}[/bold]\n"
                        f"Framework: {args.framework} | Quantization: {args.quantization}",
                        style="blue"))

    loader = ModelLoader()
    console.print("[cyan]Loading model...[/cyan]")
    model_info = loader.load(args.model, framework=args.framework, quantization=args.quantization)

    prompts_config = config.get("benchmark", {})
    num_prompts = args.num_prompts or prompts_config.get("num_prompts", 100)
    max_tokens = args.max_tokens or prompts_config.get("max_tokens", 512)

    from src.utils.prompts import get_test_prompts
    prompts = get_test_prompts(num_prompts)

    # Memory profiling
    profiler = MemoryProfiler()
    profiler.start()

    # Throughput benchmark
    tp_bench = ThroughputBenchmark(model_info, framework=args.framework)
    console.print(f"[cyan]Running {num_prompts} prompts...[/cyan]")
    tp_result = tp_bench.run(prompts, max_tokens=max_tokens, warmup=5)

    # Latency benchmark
    lat_bench = LatencyBenchmark(model_info, framework=args.framework)
    lat_result = lat_bench.run(prompts, max_tokens=max_tokens)

    profiler.stop()
    mem_result = profiler.get_results()

    # Display results
    result_table = Table(title="Benchmark Results", box=box.ROUNDED)
    result_table.add_column("Metric", style="cyan")
    result_table.add_column("Value", style="green")
    result_table.add_row("Throughput", f"{tp_result.tokens_per_sec:.1f} tok/s")
    result_table.add_row("TTFT (avg)", f"{tp_result.ttft_ms:.1f} ms")
    result_table.add_row("Latency P50", f"{lat_result.p50_ms:.1f} ms")
    result_table.add_row("Latency P95", f"{lat_result.p95_ms:.1f} ms")
    result_table.add_row("Latency P99", f"{lat_result.p99_ms:.1f} ms")
    result_table.add_row("Peak VRAM", f"{mem_result.peak_gb:.1f} GB")
    console.print(result_table)

    # Generate report
    output_dir = Path(args.output or config.get("system", {}).get("output_dir", "./reports"))
    reporter = ReportGenerator(output_dir)
    report_path = reporter.generate_json(args.model, args.framework, args.quantization,
                                         tp_result, lat_result, mem_result)
    console.print(f"
[green]Report saved to {report_path}[/green]")


def cmd_compare(args):
    """Compare frameworks and quantizations."""
    console.print(BANNER)
    config = load_config(args.config)

    frameworks = args.frameworks.split(",") if args.frameworks else ["vllm", "sglang"]
    quants = args.quantizations.split(",") if args.quantizations else ["fp16", "gptq"]

    console.print(Panel(f"[bold]Comparison: {args.model}[/bold]\n"
                        f"Frameworks: {frameworks} | Quantizations: {quants}",
                        style="blue"))

    loader = ModelLoader()
    from src.utils.prompts import get_test_prompts
    prompts = get_test_prompts(50)

    results = []
    for fw in frameworks:
        for quant in quants:
            console.print(f"
[cyan]Testing {fw} + {quant}...[/cyan]")
            try:
                model_info = loader.load(args.model, framework=fw, quantization=quant)
                profiler = MemoryProfiler()
                profiler.start()
                tp = ThroughputBenchmark(model_info, framework=fw)
                tp_result = tp.run(prompts, max_tokens=256, warmup=3)
                lat = LatencyBenchmark(model_info, framework=fw)
                lat_result = lat.run(prompts, max_tokens=256)
                profiler.stop()
                mem = profiler.get_results()
                results.append({
                    "framework": fw, "quantization": quant,
                    "throughput": tp_result, "latency": lat_result, "memory": mem
                })
                console.print(f"  [green]✓ {tp_result.tokens_per_sec:.1f} tok/s, "
                              f"P95 {lat_result.p95_ms:.1f}ms, "
                              f"VRAM {mem.peak_gb:.1f}GB[/green]")
            except Exception as e:
                console.print(f"  [red]✗ Failed: {e}[/red]")

    # Comparison table
    table = Table(title="Framework Comparison", box=box.ROUNDED)
    table.add_column("Framework", style="cyan")
    table.add_column("Quant", style="yellow")
    table.add_column("tok/s", style="green")
    table.add_column("TTFT (ms)", style="green")
    table.add_column("P95 (ms)", style="green")
    table.add_column("VRAM (GB)", style="yellow")
    for r in results:
        table.add_row(r["framework"], r["quantization"],
                      f"{r['throughput'].tokens_per_sec:.1f}",
                      f"{r['throughput'].ttft_ms:.1f}",
                      f"{r['latency'].p95_ms:.1f}",
                      f"{r['memory'].peak_gb:.1f}")
    console.print(table)

    output_dir = Path(args.output or "./reports/comparison")
    reporter = ReportGenerator(output_dir)
    reporter.generate_comparison(args.model, results)
    console.print(f"
[green]Comparison report saved to {output_dir}[/green]")


def cmd_report(args):
    """Generate report from existing benchmark data."""
    console.print(BANNER)
    input_dir = Path(args.input)
    if not input_dir.exists():
        console.print(f"[red]Directory not found: {input_dir}[/red]")
        sys.exit(1)

    reporter = ReportGenerator(input_dir)
    output = reporter.generate_from_directory(args.format or "markdown")
    console.print(f"[green]Report generated: {output}[/green]")


def main():
    parser = argparse.ArgumentParser(
        prog="rocmark",
        description="ROCmark — ROCm GPU Benchmark Suite for LLM Inference",
    )
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # detect
    p_detect = subparsers.add_parser("detect", help="Detect AMD GPUs and ROCm")
    p_detect.set_defaults(func=cmd_detect)

    # benchmark
    p_bench = subparsers.add_parser("benchmark", help="Run a single benchmark")
    p_bench.add_argument("--model", required=True, help="Model name or path")
    p_bench.add_argument("--framework", default="vllm", choices=["vllm", "sglang"])
    p_bench.add_argument("--quantization", default="fp16", choices=["fp16", "gptq", "awq"])
    p_bench.add_argument("--num-prompts", type=int, help="Number of test prompts")
    p_bench.add_argument("--max-tokens", type=int, help="Max tokens per generation")
    p_bench.add_argument("--output", help="Output directory for reports")
    p_bench.set_defaults(func=cmd_benchmark)

    # compare
    p_compare = subparsers.add_parser("compare", help="Compare frameworks/quantizations")
    p_compare.add_argument("--model", required=True, help="Model name or path")
    p_compare.add_argument("--frameworks", default="vllm,sglang", help="Comma-separated frameworks")
    p_compare.add_argument("--quantizations", default="fp16,gptq", help="Comma-separated quantizations")
    p_compare.add_argument("--output", help="Output directory")
    p_compare.set_defaults(func=cmd_compare)

    # report
    p_report = subparsers.add_parser("report", help="Generate report from data")
    p_report.add_argument("--input", required=True, help="Input data directory")
    p_report.add_argument("--format", choices=["markdown", "json"], default="markdown")
    p_report.set_defaults(func=cmd_report)

    args = parser.parse_args()

    if args.verbose:
        setup_logging("DEBUG")
    else:
        setup_logging("INFO")

    if not args.command:
        console.print(BANNER)
        parser.print_help()
        sys.exit(0)

    args.func(args)


if __name__ == "__main__":
    main()
