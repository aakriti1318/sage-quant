from dataclasses import dataclass
from typing import List, Optional

@dataclass
class Constraint:
    model_size_b: float
    hardware: str
    max_latency_ms: Optional[float]
    min_quality_pct: float
    prompt_tokens: int = 128
    output_tokens: int = 128
    prefer_engine: Optional[str] = None

@dataclass
class Recommendation:
    inference_engine: str
    quant_algo: str
    quant_scheme: str
    expected_ttft_p50_ms: float
    expected_ttft_p95_ms: float
    expected_throughput: float
    quality_delta_pct: float
    eval_method: str
    eval_sample_size: int
    prefix_caching: bool
    confidence: str
    source_rows: List[dict]  # list of dicts representing dataset rows this came from
