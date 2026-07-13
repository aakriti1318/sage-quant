# SageQuant — User Guide

## What is it?

SageQuant is a CLI that answers: **given my model, my hardware, my workload shape, and my budget for tail latency and quality loss — which inference engine, quantization algorithm, and bit-width scheme should I use, and how do I run it?**

It looks up real benchmark data and returns a recommendation with a confidence label.

---

## What it supports

| Feature | Notes |
|---------|-------|
| Workload shape matching | Prioritizes runs matching your prompt/output tokens |
| Tail Latency budgets | Checks `--max-latency` against p95, not p50 or mean |
| Confidence Downgrading | Marks as `(low sample)` when backing eval sample size < 50 |
| serving configs | Generates launch commands for vLLM, SGLang, TensorRT-LLM, and MLX |
| Local config file | `~/.sage-quant/config.yaml` for defaults |
| Contributing runs | Append your own benchmark runs via JSON or CSV file |

---

## User Journey

### 1. Install

```bash
pip install -e .
```

---

### 2. Check coverage

```bash
sage-quant list-hardware
sage-quant list-engines
sage-quant list-quant-algos
```

---

### 3. Get a recommendation

```bash
sage-quant recommend --model-size 7b --hardware a100-40gb --prompt-tokens 512 --output-tokens 256
```

Output:
```
Recommended: VLLM + none (fp16), prefix caching off
Workload: 512 in / 256 out tokens
Expected: 85ms TTFT (p50) · 140ms TTFT (p95) · 45.0 tok/s · +0.0% quality (mmlu-5shot, n=200)
Confidence: exact (5 matching benchmark runs)
```

---

## Example Scenarios to Try

Here are common real-world serving constraints you can solve with SageQuant:

### Scenario A: Zero Quality Loss Allowed (FP16 Baseline)
Find the best engine that guarantees 100% quality retention (no quantization artifacts):
```bash
sage-quant recommend --model-size 7b --hardware rtx-4090 --min-quality 100
```

### Scenario B: Restrict serving to SGLang
Optimize the quantization scheme while enforcing SGLang as the serving engine:
```bash
sage-quant recommend --model-size 8b --hardware a100-40gb --prefer-engine sglang
```

### Scenario C: High-Throughput Edge Serving on Mac (Apple Silicon)
Find what fits on Apple M1 Pro within a tail-latency budget of 350ms:
```bash
sage-quant recommend --model-size 7b --hardware m1-pro --max-latency 350ms
```

### Scenario D: Generate MLX Config for Apple Silicon
Generate MLX serving instructions directly:
```bash
sage-quant serve-config --model-size 7b --hardware m1-pro --model mlx-community/Llama-3-8B-Instruct-4bit
```

---

### 4. Generate serving configs

```bash
sage-quant serve-config \
  --model-size 7b \
  --hardware a100-40gb \
  --model meta-llama/Meta-Llama-3-8B-Instruct \
  --prefer-engine sglang
```

Output:
```
Recommended: SGLANG + AWQ (w4a16), prefix caching on
Workload: 512 in / 256 out tokens
Expected: 98ms TTFT (p50) · 165ms TTFT (p95) · 52.0 tok/s · -1.4% quality (mmlu-5shot, n=200)
Confidence: exact

Platform: sglang

Launch command:
  python -m sglang.launch_server --model-path meta-llama/Meta-Llama-3-8B-Instruct --quantization awq --host 0.0.0.0 --port 30000
```

---

### 5. Contribute your own benchmark data

Append a JSON log file:
```bash
sage-quant contribute --run-log my_run.json
```

---

## CLI Reference

```
sage-quant recommend    --model-size SIZE --hardware HW [--max-latency MS] [--min-quality PCT] [--prompt-tokens INT] [--output-tokens INT] [--prefer-engine ENG]
sage-quant serve-config --model-size SIZE --hardware HW --model NAME [--out FILE] [--min-quality PCT] [--max-latency MS] [--prompt-tokens INT] [--output-tokens INT] [--prefer-engine ENG]
sage-quant list-hardware
sage-quant list-engines
sage-quant list-quant-algos
sage-quant contribute   --run-log FILE
```

---

## Behavior Notes

### What happens when constraints can't be satisfied?

If no benchmark combination passes both `--max-latency` and `--min-quality` (e.g. you request 1ms latency and 100% quality), the CLI exits with:

```
no matching data — try a different hardware or looser quality bound
```

Exit code is `1`. No silent fallback, no fudged result.

### Input format and case sensitivity

- `--hardware` is **case-insensitive**: `A100-40GB`, `a100-40gb`, `a100-40GB` are all treated identically.
- `--model-size` accepts both `7b` and `7B` (case-insensitive) as well as bare floats like `7.0`.
- `--max-latency` accepts `200ms`, `0.2s`, or bare `200` (assumed ms).

### Constraint hierarchy when no exact match exists

If no row in the dataset exactly matches your hardware + model size, the recommender interpolates from the closest available data using linear regression on bit-width and model size. The result is labeled `confidence: interpolated`. If the dataset contains no rows for your hardware family at all, it falls back to the full dataset and labels the result accordingly.

### Interpolation without `[advanced]` extra

Interpolation uses a **pure-Python linear regression** fallback and does not require `scikit-learn`. Installing `pip install sage-quant[advanced]` enables numpy-accelerated paths and the optional Streamlit explorer, but is not required for any CLI command.

### Tie-breaking

When two configurations score identically (same quality delta, same throughput/latency ratio), the one with **lower p95 latency** wins. If still tied, insertion order from the dataset is used.
