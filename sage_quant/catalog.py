CATALOG = {
    "engines": {
        "vllm": "vLLM is highly optimized for throughput, features PagedAttention, and has broad hardware/model support.",
        "sglang": "SGLang features RadixAttention and is highly efficient for high-cache-reuse workloads.",
        "tensorrt-llm": "TensorRT-LLM requires a hardware-specific build step but offers top-tier performance on NVIDIA GPUs.",
        "mlx": "MLX is specifically designed for Apple Silicon with unified memory."
    },
    "algos": {
        "gptq": "GPTQ has broad toolchain support and is fast on CUDA.",
        "awq": "AWQ generally preserves quality slightly better than GPTQ at 4-bit.",
        "smoothquant": "SmoothQuant optimizes 8-bit quantization for activation outliers.",
        "fp8": "FP8 provides native hardware acceleration on hopper/ada architectures.",
        "mlx-quant": "MLX native quantization is optimized for unified memory bandwidth.",
        "none": "FP16/BF16 unquantized baseline."
    }
}

def get_engine_note(engine: str) -> str:
    return CATALOG["engines"].get(engine.strip().lower(), "")

def get_algo_note(algo: str) -> str:
    return CATALOG["algos"].get(algo.strip().lower(), "")
