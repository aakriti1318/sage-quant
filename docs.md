# QuantSage — User Guide

## What is it?

QuantSage is a CLI that answers: **given my model, my hardware, my workload shape, and my budget for tail latency and quality loss — which inference engine, quantization algorithm, and bit-width scheme should I use, and how do I run it?**

It looks up real benchmark data and returns a recommendation with a confidence label.

---

## What it supports

| Feature | Notes |
|---------|-------|
| Workload shape matching | Prioritizes runs matching your prompt/output tokens |
| Tail Latency budgets | Checks `--max-latency` against p95, not p50 or mean |
| Confidence Downgrading | Marks as `(low sample)` when backing eval sample size < 50 |
| serving configs | Generates launch commands for vLLM, SGLang, TensorRT-LLM, and MLX |
| Local config file | `~/.quantsage/config.yaml` for defaults |
| Contributing runs | Append logs or run live benchmarks |

---

## User Journey

### 1. Install

```bash
pip install -e .
```

---

### 2. Check coverage

```bash
quantsage list-hardware
quantsage list-engines
quantsage list-quant-algos
```

---

### 3. Get a recommendation

```bash
quantsage recommend --model-size 7b --hardware a100-40gb --prompt-tokens 512 --output-tokens 256
```

Output:
```
Recommended: VLLM + none (fp16), prefix caching off
Workload: 512 in / 256 out tokens
Expected: 85ms TTFT (p50) · 140ms TTFT (p95) · 45.0 tok/s · +0.0% quality (mmlu-5shot, n=200)
Confidence: exact (5 matching benchmark runs)

Note (Engine): vLLM is highly optimized for throughput, features PagedAttention, and has broad hardware/model support.
Note (Algo): FP16/BF16 unquantized baseline.
```

Filter by engine:
```bash
quantsage recommend --model-size 7b --hardware a100-40gb --prefer-engine sglang
```

---

### 4. Generate serving configs

```bash
quantsage serve-config \
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
quantsage contribute --run-log my_run.json
```

Or run the benchmarks directly via QuantSage:
```bash
quantsage contribute --benchmark --model-size 13b --hardware a100-40gb \
  --engine sglang --quant-algo awq --model your-org/your-model
```

---

## CLI Reference

```
quantsage recommend    --model-size SIZE --hardware HW [--max-latency MS] [--min-quality PCT] [--prompt-tokens INT] [--output-tokens INT] [--prefer-engine ENG]
quantsage serve-config --model-size SIZE --hardware HW --model NAME [--out FILE] [--min-quality PCT] [--max-latency MS] [--prompt-tokens INT] [--output-tokens INT] [--prefer-engine ENG]
quantsage list-hardware
quantsage list-engines
quantsage list-quant-algos
quantsage contribute   [--run-log FILE] [--benchmark --model-size SIZE --hardware HW --engine ENG --quant-algo ALGO --model NAME]
```
