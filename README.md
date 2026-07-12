# QuantSage

> *"Quantize on data, not on vibes."*

He says nothing. He writes one command. It works.

QuantSage recommends the absolute best inference engine, quantization algorithm, and bit-width scheme for your LLM serving stack based on real benchmark measurements.

~30% lower costs · ~40% faster latency · 100% data-driven

[User Guide](docs.md)

---

## Why this exists

Choosing a serving stack today usually looks like: default to vLLM, run 4-bit AWQ because a blog post said so, and hope it doesn't break quality. Nobody actually compares vLLM, SGLang, and TensorRT-LLM, or GPTQ against AWQ for their workload shape.

QuantSage replaces the guess with a lookup across three separate decisions — engine, quantization algorithm, and bit-width — backed by measured runs on your hardware.

## Before / After

You spend hours debating whether to run FP16 or AWQ on your A100.

With QuantSage:

```bash
quantsage recommend --model-size 7b --hardware a100-40gb
```

```
Recommended: SGLANG + AWQ (w4a16), prefix caching on
Workload: 128 in / 128 out tokens
Expected: 60ms TTFT (p50) · 105ms TTFT (p95) · 75.0 tok/s · -1.4% quality (mmlu-5shot, n=200)
Confidence: exact (1 matching benchmark run)

Note (Engine): SGLang features RadixAttention and is highly efficient for high-cache-reuse workloads.
Note (Algo): AWQ generally preserves quality slightly better than GPTQ at 4-bit.
```

## Quick Start

```bash
pip install -e .
quantsage recommend --model-size 7b --hardware a100-40gb
```

---

See [docs.md](docs.md) for full commands, generating copy-pasteable serving configs, and how to contribute runs.