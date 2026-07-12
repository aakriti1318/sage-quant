from typing import Dict, Any
import yaml

def get_vllm_sglang_quant(quant_algo: str) -> str:
    qa = quant_algo.lower().strip()
    if qa in ("awq", "gptq", "fp8", "smoothquant"):
        return qa
    return ""

def generate_serving_config(engine: str, quant_algo: str, quant_scheme: str, model_name: str) -> Dict[str, Any]:
    eng = engine.lower().strip()
    
    if eng == "mlx":
        suffix = ""
        if "4" in quant_scheme.lower():
            suffix = "-4bit"
        elif "8" in quant_scheme.lower():
            suffix = "-8bit"
            
        model_display = f"{model_name}{suffix}"
        command = f"python -m mlx_lm.server --model {model_display} --port 8080"
        
        return {
            "platform": "mlx",
            "model": model_display,
            "port": 8080,
            "launch_command": command,
            "instructions": [
                "1. Install mlx-lm: pip install mlx-lm",
                "2. Run the server using the launch command below",
                "3. Query the endpoint at http://localhost:8080/v1/chat/completions"
            ]
        }
        
    elif eng == "sglang":
        quant = get_vllm_sglang_quant(quant_algo)
        command_parts = ["python -m sglang.launch_server", "--model-path", model_name]
        if quant:
            command_parts.extend(["--quantization", quant])
        command_parts.extend(["--host", "0.0.0.0", "--port", "30000"])
        command = " ".join(command_parts)
        
        return {
            "platform": "sglang",
            "model": model_name,
            "port": 30000,
            "launch_command": command,
            "instructions": [
                "1. Install sglang: pip install sglang",
                "2. Start the server using the launch command",
                "3. Query the endpoint at http://localhost:30000/v1/chat/completions"
            ]
        }
        
    elif eng == "tensorrt-llm":
        return {
            "platform": "tensorrt-llm",
            "model": model_name,
            "launch_command": f"trtllm-build --checkpoint_dir ./trt_ckpt --output_dir ./trt_engines",
            "instructions": [
                "Note: TensorRT-LLM requires a hardware-specific build step before serving.",
                "1. Convert/Quantize HuggingFace checkpoint to TensorRT-LLM format.",
                "2. Run build step: trtllm-build --checkpoint_dir ./trt_ckpt --output_dir ./trt_engines",
                "3. Run server via Triton Inference Server pointing to your engines repository."
            ]
        }
        
    else:  # Default to vLLM
        quant = get_vllm_sglang_quant(quant_algo)
        command_parts = ["vllm serve", model_name]
        if quant:
            command_parts.append(f"--quantization {quant}")
        command_parts.extend(["--host 0.0.0.0", "--port 8000"])
        command = " ".join(command_parts)
        
        yaml_config = {
            "model": model_name,
            "host": "0.0.0.0",
            "port": 8000,
        }
        if quant:
            yaml_config["quantization"] = quant
            
        return {
            "platform": "vllm",
            "model": model_name,
            "quantization": quant or "None",
            "host": "0.0.0.0",
            "port": 8000,
            "yaml_content": yaml.dump(yaml_config, default_flow_style=False),
            "launch_command": command,
            "instructions": [
                "1. Install vLLM: pip install vllm",
                "2. Launch via command line or using the generated config.yaml:",
                "   vllm serve --config config.yaml",
                "3. Query the OpenAI-compatible endpoint at http://localhost:8000/v1/chat/completions"
            ]
        }
