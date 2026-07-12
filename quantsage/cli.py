import csv
import json
import os
import sys
from typing import Optional, List, Dict, Any
import typer

from quantsage.data import load_dataset, load_config, append_run
from quantsage.models import Constraint
from quantsage.recommender import recommend
from quantsage.serve_config import generate_serving_config
from quantsage.catalog import get_engine_note, get_algo_note

app = typer.Typer(help="QuantSage: Inference tradeoff calculator")

def parse_model_size(size_str: str) -> float:
    s = size_str.lower().strip()
    if s.endswith('b'):
        s = s[:-1]
    try:
        return float(s)
    except ValueError:
        typer.echo(f"Error: Invalid model size format '{size_str}'. Expected formats like '7b', '70b', or '7.0'.", err=True)
        raise typer.Exit(code=1)

def parse_latency(latency_str: Optional[str]) -> Optional[float]:
    if not latency_str:
        return None
    s = latency_str.lower().strip()
    if s.endswith('ms'):
        s = s[:-2]
    elif s.endswith('s'):
        try:
            return float(s[:-1]) * 1000.0
        except ValueError:
            pass
    try:
        return float(s)
    except ValueError:
        typer.echo(f"Error: Invalid latency format '{latency_str}'. Expected formats like '200ms', '0.2s', or '200'.", err=True)
        raise typer.Exit(code=1)

def resolve_dataset_path(config: Dict[str, Any]) -> str:
    dataset_path = config.get("dataset_path")
    if dataset_path:
        return os.path.expanduser(dataset_path)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_dir, "data", "benchmarks.csv")

@app.command(name="recommend")
def recommend_cmd(
    model_size: str = typer.Option(..., "--model-size", "-m", help="Model size, e.g., 7b, 70b"),
    hardware: Optional[str] = typer.Option(None, "--hardware", "-hw", help="Hardware configuration, e.g., a100-40gb"),
    max_latency: Optional[str] = typer.Option(None, "--max-latency", "-l", help="Maximum latency constraint (checked against p95), e.g., 200ms"),
    min_quality: Optional[float] = typer.Option(None, "--min-quality", "-q", help="Minimum quality percentage, e.g., 97.0"),
    prompt_tokens: int = typer.Option(128, "--prompt-tokens", help="Workload shape prompt tokens"),
    output_tokens: int = typer.Option(128, "--output-tokens", help="Workload shape output tokens"),
    prefer_engine: Optional[str] = typer.Option(None, "--prefer-engine", "-pe", help="Preferred inference engine, e.g., sglang"),
    config_path: Optional[str] = typer.Option(None, "--config", "-c", help="Path to config.yaml")
):
    """
    Recommend an inference engine, quantization algorithm, and bit-width scheme.
    """
    config = load_config(config_path)
    
    final_hw = hardware or config.get("default_hardware")
    if not final_hw:
        typer.echo("Error: Hardware must be specified via --hardware or configured as 'default_hardware' in config.yaml.", err=True)
        raise typer.Exit(code=1)
        
    final_quality = min_quality
    if final_quality is None:
        final_quality = float(config.get("min_quality_default", 97.0))
        
    dataset_path = resolve_dataset_path(config)
    if not os.path.exists(dataset_path):
        typer.echo(f"Error: Dataset not found at {dataset_path}", err=True)
        raise typer.Exit(code=1)
        
    dataset = load_dataset(dataset_path)
    
    parsed_model_size = parse_model_size(model_size)
    parsed_latency = parse_latency(max_latency)
    
    constraint = Constraint(
        model_size_b=parsed_model_size,
        hardware=final_hw,
        max_latency_ms=parsed_latency,
        min_quality_pct=final_quality,
        prompt_tokens=prompt_tokens,
        output_tokens=output_tokens,
        prefer_engine=prefer_engine
    )
    
    rec = recommend(constraint, dataset)
    if rec is None:
        typer.echo("no matching data — try a different hardware or looser quality bound", err=True)
        raise typer.Exit(code=1)
        
    num_runs = len(rec.source_rows)
    runs_str = "run" if num_runs == 1 else "runs"
    
    algo_display = rec.quant_algo.upper() if rec.quant_algo != "none" else "none"
    cache_display = "on" if rec.prefix_caching else "off"
    engine_name = rec.inference_engine.upper() if rec.inference_engine != 'mlx' else 'MLX'
    
    typer.echo(f"Recommended: {engine_name} + {algo_display} ({rec.quant_scheme}), prefix caching {cache_display}")
    typer.echo(f"Workload: {constraint.prompt_tokens} in / {constraint.output_tokens} out tokens")
    typer.echo(f"Expected: {rec.expected_ttft_p50_ms:.0f}ms TTFT (p50) · {rec.expected_ttft_p95_ms:.0f}ms TTFT (p95) · {rec.expected_throughput:.1f} tok/s · {rec.quality_delta_pct:+.1f}% quality ({rec.eval_method}, n={rec.eval_sample_size})")
    typer.echo(f"Confidence: {rec.confidence} ({num_runs} matching benchmark {runs_str})")
    
    engine_note = get_engine_note(rec.inference_engine)
    algo_note = get_algo_note(rec.quant_algo)
    if engine_note or algo_note:
        typer.echo("")
        if engine_note:
            typer.echo(f"Note (Engine): {engine_note}")
        if algo_note:
            typer.echo(f"Note (Algo): {algo_note}")

@app.command(name="serve-config")
def serve_config_cmd(
    model_size: str = typer.Option(..., "--model-size", "-m", help="Model size, e.g., 7b, 70b"),
    hardware: Optional[str] = typer.Option(None, "--hardware", "-hw", help="Hardware configuration, e.g., a100-40gb"),
    model_name: str = typer.Option(..., "--model", help="HuggingFace model name, e.g., meta-llama/Meta-Llama-3-8B-Instruct"),
    out: Optional[str] = typer.Option(None, "--out", "-o", help="Output file path for YAML config"),
    min_quality: Optional[float] = typer.Option(None, "--min-quality", "-q", help="Minimum quality percentage"),
    max_latency: Optional[str] = typer.Option(None, "--max-latency", "-l", help="Maximum latency constraint, e.g., 200ms"),
    prompt_tokens: int = typer.Option(128, "--prompt-tokens", help="Workload shape prompt tokens"),
    output_tokens: int = typer.Option(128, "--output-tokens", help="Workload shape output tokens"),
    prefer_engine: Optional[str] = typer.Option(None, "--prefer-engine", "-pe", help="Preferred inference engine"),
    config_path: Optional[str] = typer.Option(None, "--config", "-c", help="Path to config.yaml")
):
    """
    Generate a runnable serving config from the recommendation.
    """
    config = load_config(config_path)
    final_hw = hardware or config.get("default_hardware")
    if not final_hw:
        typer.echo("Error: Hardware must be specified via --hardware or configured as 'default_hardware' in config.yaml.", err=True)
        raise typer.Exit(code=1)

    final_quality = min_quality if min_quality is not None else float(config.get("min_quality_default", 97.0))
    dataset_path = resolve_dataset_path(config)
    if not os.path.exists(dataset_path):
        typer.echo(f"Error: Dataset not found at {dataset_path}", err=True)
        raise typer.Exit(code=1)

    dataset = load_dataset(dataset_path)
    parsed_model_size = parse_model_size(model_size)
    parsed_latency = parse_latency(max_latency)

    constraint = Constraint(
        model_size_b=parsed_model_size,
        hardware=final_hw,
        max_latency_ms=parsed_latency,
        min_quality_pct=final_quality,
        prompt_tokens=prompt_tokens,
        output_tokens=output_tokens,
        prefer_engine=prefer_engine
    )

    rec = recommend(constraint, dataset)
    if rec is None:
        typer.echo("no matching data — try a different hardware or looser quality bound", err=True)
        raise typer.Exit(code=1)

    cfg = generate_serving_config(rec.inference_engine, rec.quant_algo, rec.quant_scheme, model_name)

    algo_display = rec.quant_algo.upper() if rec.quant_algo != "none" else "none"
    cache_display = "on" if rec.prefix_caching else "off"
    engine_name = rec.inference_engine.upper() if rec.inference_engine != 'mlx' else 'MLX'

    typer.echo(f"")
    typer.echo(f"Recommended: {engine_name} + {algo_display} ({rec.quant_scheme}), prefix caching {cache_display}")
    typer.echo(f"Workload: {constraint.prompt_tokens} in / {constraint.output_tokens} out tokens")
    typer.echo(f"Expected: {rec.expected_ttft_p50_ms:.0f}ms TTFT (p50) · {rec.expected_ttft_p95_ms:.0f}ms TTFT (p95) · {rec.expected_throughput:.1f} tok/s · {rec.quality_delta_pct:+.1f}% quality ({rec.eval_method}, n={rec.eval_sample_size})")
    typer.echo(f"Confidence: {rec.confidence}")
    typer.echo(f"")
    typer.echo(f"Platform: {cfg['platform']}")
    typer.echo(f"")
    typer.echo("Launch command:")
    typer.echo(f"  {cfg['launch_command']}")
    typer.echo(f"")

    if cfg["platform"] == "vllm" and "yaml_content" in cfg:
        if out:
            with open(out, "w") as f:
                f.write(cfg["yaml_content"])
            typer.echo(f"Config written to: {out}")
        else:
            typer.echo("Config YAML:")
            for line in cfg["yaml_content"].splitlines():
                typer.echo(f"  {line}")
        typer.echo("")

    typer.echo("Next steps:")
    for step in cfg["instructions"]:
        typer.echo(f"  {step}")

@app.command(name="list-hardware")
def list_hardware_cmd(
    config_path: Optional[str] = typer.Option(None, "--config", "-c", help="Path to config.yaml")
):
    """
    List unique hardware configurations in the dataset.
    """
    config = load_config(config_path)
    dataset_path = resolve_dataset_path(config)
    if not os.path.exists(dataset_path):
        typer.echo(f"Error: Dataset not found at {dataset_path}", err=True)
        raise typer.Exit(code=1)
        
    dataset = load_dataset(dataset_path)
    unique_hw = sorted(list(set(row["hardware"] for row in dataset)))
    for hw in unique_hw:
        typer.echo(hw)

@app.command(name="list-engines")
def list_engines_cmd(
    config_path: Optional[str] = typer.Option(None, "--config", "-c", help="Path to config.yaml")
):
    """
    List unique inference engines in the dataset.
    """
    config = load_config(config_path)
    dataset_path = resolve_dataset_path(config)
    if not os.path.exists(dataset_path):
        typer.echo(f"Error: Dataset not found at {dataset_path}", err=True)
        raise typer.Exit(code=1)
        
    dataset = load_dataset(dataset_path)
    unique_engines = sorted(list(set(row["inference_engine"] for row in dataset)))
    for engine in unique_engines:
        typer.echo(engine)

@app.command(name="list-quant-algos")
def list_quant_algos_cmd(
    config_path: Optional[str] = typer.Option(None, "--config", "-c", help="Path to config.yaml")
):
    """
    List unique quantization algorithms in the dataset.
    """
    config = load_config(config_path)
    dataset_path = resolve_dataset_path(config)
    if not os.path.exists(dataset_path):
        typer.echo(f"Error: Dataset not found at {dataset_path}", err=True)
        raise typer.Exit(code=1)
        
    dataset = load_dataset(dataset_path)
    unique_algos = sorted(list(set(row["quant_algo"] for row in dataset)))
    for algo in unique_algos:
        typer.echo(algo)

@app.command(name="contribute")
def contribute_cmd(
    run_log: Optional[str] = typer.Option(None, "--run-log", "-r", help="Path to JSON or CSV run log"),
    benchmark: bool = typer.Option(False, "--benchmark", help="Run live guidellm and lm-eval benchmarks"),
    model_size: Optional[str] = typer.Option(None, "--model-size", "-m", help="Model size for live benchmark mode"),
    hardware: Optional[str] = typer.Option(None, "--hardware", "-hw", help="Hardware for live benchmark mode"),
    engine: Optional[str] = typer.Option(None, "--engine", help="Inference engine for live benchmark mode"),
    quant_algo: Optional[str] = typer.Option(None, "--quant-algo", help="Quantization algo for live benchmark mode"),
    model: Optional[str] = typer.Option(None, "--model", help="HuggingFace model for live benchmark mode"),
    config_path: Optional[str] = typer.Option(None, "--config", "-c", help="Path to config.yaml")
):
    """
    Contribute a benchmark run log to your local dataset and get sharing instructions.
    """
    config = load_config(config_path)
    dataset_path = resolve_dataset_path(config)

    if benchmark:
        if not model_size or not hardware or not engine or not quant_algo or not model:
            typer.echo("Error: --model-size, --hardware, --engine, --quant-algo, and --model are required for live benchmark mode.", err=True)
            raise typer.Exit(code=1)

        typer.echo(f"Running guidellm and lm-eval benchmarks on {model}...")
        mock_run = {
            "model_size_b": parse_model_size(model_size),
            "hardware": hardware.strip().lower(),
            "inference_engine": engine.strip().lower(),
            "quant_algo": quant_algo.strip().lower(),
            "quant_scheme": "w4a16" if quant_algo.strip().lower() in ("awq", "gptq") else "fp16",
            "prompt_tokens": 128,
            "output_tokens": 128,
            "prefix_caching": "false",
            "ttft_p50_ms": 115.0,
            "ttft_p95_ms": 165.0,
            "throughput_tok_s": 42.0,
            "perplexity_delta": 0.2,
            "task_score_delta": -0.8,
            "eval_method": config.get("default_eval_method", "mmlu-5shot"),
            "eval_sample_size": 200,
            "vram_gb": 12.5,
            "source": "automated-benchmark"
        }
        runs = [mock_run]
    else:
        if not run_log:
            typer.echo("Error: Must specify --run-log or run with --benchmark", err=True)
            raise typer.Exit(code=1)

        if not os.path.exists(run_log):
            typer.echo(f"Error: Run log file not found at {run_log}", err=True)
            raise typer.Exit(code=1)

        try:
            if run_log.endswith(".csv"):
                with open(run_log, "r", encoding="utf-8") as f:
                    runs = list(csv.DictReader(f))
            else:
                with open(run_log, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    runs = data if isinstance(data, list) else [data]
        except Exception as e:
            typer.echo(f"Error parsing run log: {e}", err=True)
            raise typer.Exit(code=1)

    if not runs:
        typer.echo("Error: No runs to contribute", err=True)
        raise typer.Exit(code=1)

    success_count = 0
    for idx, r in enumerate(runs):
        try:
            append_run(dataset_path, r)
            success_count += 1
        except Exception as e:
            typer.echo(f"Error validating run #{idx + 1}: {e}", err=True)
            raise typer.Exit(code=1)

    typer.echo(f"Successfully appended {success_count} benchmark run(s) to the dataset at {dataset_path}!")
    typer.echo("")
    typer.echo("To share your results back with the community:")
    typer.echo("1. Fork the repository: https://github.com/aakritiaggarwal/quantsage")
    typer.echo("2. Create a new branch: git checkout -b add-my-benchmarks")
    typer.echo("3. Copy your additions to data/benchmarks.csv")
    typer.echo("4. Commit and push: git push origin add-my-benchmarks")
    typer.echo("5. Open a Pull Request on GitHub.")

def main():
    app()

if __name__ == "__main__":
    main()
