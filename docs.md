# QuantSage User Guide

This guide covers all commands, configuration options, dataset layout, and how to use QuantSage.

---

## What it Decides

QuantSage evaluates and recommends configuration options across three distinct decisions:

1. **Inference Engine**: `vllm` | `sglang` | `tensorrt-llm` | `mlx`
2. **Quantization Algorithm**: `gptq` | `awq` | `smoothquant` | `fp8` | `mlx-quant` | `none`
3. **Bit-width Scheme**: `w4a16` | `w8a8` | `q4` | `q8` | `fp16`

---

## Installation

```bash
git clone https://github.com/aakriti1318/quantsage
cd quantsage
pip install -e .
```

### Extras
- `pip install -e ".[advanced]"`: Enables regression-based interpolation (pandas, scikit-learn).
- `pip install -e ".[benchmark]"`: Enables live benchmarking (guidellm, lm-eval).

---

## Command Reference

### 1. `recommend`
Find the best engine, algorithm, and scheme matching your hardware, size, and workload.

```bash
quantsage recommend --model-size 7b --hardware a100-40gb \
  --prompt-tokens 512 --output-tokens 256 \
  --max-latency 200ms --min-quality 98 \
  --prefer-engine sglang
```

- `--max-latency`: Validated against tail latency (**p95 TTFT**), not the mean.
- Workload shapes (`--prompt-tokens` / `--output-tokens`): Automatically finds the closest matching shape.
- Confidence levels:
  - `exact`: Match found. Downgraded to `exact (low sample)` if the evaluation sample count is less than 50.
  - `interpolated`: Estimated via linear regression.
  - `no_data`: No matches or similar runs found.

### 2. `serve-config`
Generate copy-pasteable serving commands and configuration files.

```bash
quantsage serve-config --model-size 7b --hardware a100-40gb \
  --model meta-llama/Meta-Llama-3-8B-Instruct --prefer-engine sglang
```

- Supports outputting YAML configs (vLLM only) with `--out config.yaml`.
- Handles platform-specific execution commands for SGLang, vLLM, TensorRT-LLM, and MLX.

### 3. `contribute`
Add your own benchmark runs to your local database.

```bash
# Add from a JSON file
quantsage contribute --run-log my_run.json

# Run live benchmark measurements (requires [benchmark] extra)
quantsage contribute --benchmark --model-size 8b --hardware a100-40gb \
  --engine vllm --quant-algo awq --model meta-llama/Meta-Llama-3-8B-Instruct
```

### 4. Database Exploration

```bash
quantsage list-hardware
quantsage list-engines
quantsage list-quant-algos
```

---

## Configuration File (`~/.quantsage/config.yaml`)

Define default settings for your workspace:

```yaml
default_hardware: a100-40gb
min_quality_default: 97
default_eval_method: mmlu-5shot
dataset_path: ~/.quantsage/benchmarks.csv
```

---

## Dataset Schema (`data/benchmarks.csv`)

| Column | Description |
|---|---|
| `model_size_b` | Model size in billions |
| `hardware` | Target hardware name (e.g. `rtx-4090`) |
| `inference_engine` | Serving stack (`vllm`, `sglang`, `mlx`) |
| `quant_algo` | Method (`awq`, `gptq`, `fp8`, `none`) |
| `quant_scheme` | Width format (`w4a16`, `fp16`) |
| `prompt_tokens` / `output_tokens` | Workload shape during benchmark |
| `prefix_caching` | Boolean string (`true`/`false`) |
| `ttft_p50_ms` / `ttft_p95_ms` | Median and tail latencies |
| `throughput_tok_s` | Tokens per second |
| `task_score_delta` | Quality change percentage vs FP16 baseline |
| `eval_method` | Grounding benchmark (e.g. `mmlu-5shot`) |
| `eval_sample_size` | Evaluation sample count |
| `vram_gb` | VRAM utilization |
| `source` | Measurement source |
