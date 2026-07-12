import sys
import os
from quantsage.data import load_dataset

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_path = os.path.join(base_dir, "data", "benchmarks.csv")
    
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset file not found at {dataset_path}")
        sys.exit(1)
        
    try:
        dataset = load_dataset(dataset_path)
        print(f"Successfully validated dataset! Found {len(dataset)} rows.")
        
        valid_engines = {"vllm", "sglang", "tensorrt-llm", "mlx"}
        for idx, row in enumerate(dataset):
            if row["inference_engine"] not in valid_engines:
                print(f"Error on row {idx + 2}: Invalid engine '{row['inference_engine']}'. Expected one of {valid_engines}")
                sys.exit(1)
                
            if row["model_size_b"] <= 0:
                print(f"Error on row {idx + 2}: Invalid model size '{row['model_size_b']}'")
                sys.exit(1)
                
            if row["prompt_tokens"] <= 0 or row["output_tokens"] <= 0:
                print(f"Error on row {idx + 2}: Workload tokens must be positive.")
                sys.exit(1)
                
            if row["eval_sample_size"] <= 0:
                print(f"Error on row {idx + 2}: eval_sample_size must be positive.")
                sys.exit(1)
                
        sys.exit(0)
    except Exception as e:
        print(f"Validation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
