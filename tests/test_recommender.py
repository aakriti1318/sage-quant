import pytest
from quantsage.models import Constraint
from quantsage.recommender import recommend

TEST_DATASET = [
    {
        "model_size_b": 7.0,
        "hardware": "a100-40gb",
        "inference_engine": "vllm",
        "quant_algo": "none",
        "quant_scheme": "fp16",
        "prompt_tokens": 512,
        "output_tokens": 256,
        "prefix_caching": False,
        "ttft_p50_ms": 80.0,
        "ttft_p95_ms": 130.0,
        "throughput_tok_s": 40.0,
        "perplexity_delta": 0.0,
        "task_score_delta": 0.0,
        "eval_method": "mmlu-5shot",
        "eval_sample_size": 200,
        "vram_gb": 16.0,
        "source": "test"
    },
    {
        "model_size_b": 7.0,
        "hardware": "a100-40gb",
        "inference_engine": "sglang",
        "quant_algo": "awq",
        "quant_scheme": "w4a16",
        "prompt_tokens": 512,
        "output_tokens": 256,
        "prefix_caching": True,
        "ttft_p50_ms": 90.0,
        "ttft_p95_ms": 150.0,
        "throughput_tok_s": 50.0,
        "perplexity_delta": 0.2,
        "task_score_delta": -1.2,
        "eval_method": "mmlu-5shot",
        "eval_sample_size": 200,
        "vram_gb": 9.0,
        "source": "test"
    },
    {
        "model_size_b": 7.0,
        "hardware": "a100-40gb",
        "inference_engine": "vllm",
        "quant_algo": "none",
        "quant_scheme": "fp16",
        "prompt_tokens": 128,
        "output_tokens": 128,
        "prefix_caching": False,
        "ttft_p50_ms": 50.0,
        "ttft_p95_ms": 90.0,
        "throughput_tok_s": 60.0,
        "perplexity_delta": 0.0,
        "task_score_delta": 0.0,
        "eval_method": "mmlu-5shot",
        "eval_sample_size": 20,
        "vram_gb": 16.0,
        "source": "test"
    },
    {
        "model_size_b": 70.0,
        "hardware": "a100-40gb",
        "inference_engine": "vllm",
        "quant_algo": "none",
        "quant_scheme": "fp16",
        "prompt_tokens": 512,
        "output_tokens": 256,
        "prefix_caching": False,
        "ttft_p50_ms": 200.0,
        "ttft_p95_ms": 320.0,
        "throughput_tok_s": 15.0,
        "perplexity_delta": 0.0,
        "task_score_delta": 0.0,
        "eval_method": "mmlu-5shot",
        "eval_sample_size": 200,
        "vram_gb": 140.0,
        "source": "test"
    }
]

def test_exact_match():
    constraint = Constraint(
        model_size_b=7.0,
        hardware="a100-40gb",
        max_latency_ms=150.0,
        min_quality_pct=99.0,
        prompt_tokens=512,
        output_tokens=256
    )
    rec = recommend(constraint, TEST_DATASET)
    assert rec is not None
    assert rec.inference_engine == "vllm"
    assert rec.quant_algo == "none"
    assert rec.quant_scheme == "fp16"
    assert rec.confidence == "exact"

def test_prefer_engine():
    constraint = Constraint(
        model_size_b=7.0,
        hardware="a100-40gb",
        max_latency_ms=160.0,
        min_quality_pct=98.0,
        prompt_tokens=512,
        output_tokens=256,
        prefer_engine="sglang"
    )
    rec = recommend(constraint, TEST_DATASET)
    assert rec is not None
    assert rec.inference_engine == "sglang"
    assert rec.quant_algo == "awq"
    assert rec.quant_scheme == "w4a16"
    assert rec.confidence == "exact"

def test_low_sample_downgrade():
    # Matches the 128/128 row where eval_sample_size = 20 < 50
    constraint = Constraint(
        model_size_b=7.0,
        hardware="a100-40gb",
        max_latency_ms=100.0,
        min_quality_pct=99.0,
        prompt_tokens=128,
        output_tokens=128
    )
    rec = recommend(constraint, TEST_DATASET)
    assert rec is not None
    assert rec.confidence == "exact (low sample)"

def test_interpolated_match():
    # Model size 13B (should interpolate)
    constraint = Constraint(
        model_size_b=13.0,
        hardware="a100-40gb",
        max_latency_ms=400.0,
        min_quality_pct=97.0,
        prompt_tokens=512,
        output_tokens=256
    )
    rec = recommend(constraint, TEST_DATASET)
    assert rec is not None
    assert "interpolated" in rec.confidence
    assert rec.inference_engine in ("vllm", "sglang")

def test_no_data():
    constraint = Constraint(
        model_size_b=7.0,
        hardware="impossible-gpu",
        max_latency_ms=10.0,
        min_quality_pct=99.9,
        prompt_tokens=512,
        output_tokens=256
    )
    rec = recommend(constraint, TEST_DATASET)
    assert rec is None
