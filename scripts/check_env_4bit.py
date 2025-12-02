import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

def check_memory_usage():
    model_name = "Qwen/Qwen2.5-Math-1.5B"
    print(f"Loading {model_name} with 4-bit quantization...")

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True
        )
        
        print("Model loaded successfully.")
        
        # Check memory
        if torch.cuda.is_available():
            mem_allocated = torch.cuda.memory_allocated() / 1024**3
            mem_reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"CUDA Memory Allocated: {mem_allocated:.2f} GB")
            print(f"CUDA Memory Reserved:  {mem_reserved:.2f} GB")
            
            if mem_allocated < 5.0:
                 print("SUCCESS: Memory usage is under 5GB as expected for 4-bit quantization.")
            else:
                 print("WARNING: Memory usage is higher than expected (>5GB).")
        else:
            print("CUDA not available. Cannot verify VRAM usage.")

    except Exception as e:
        print(f"FAILED to load model: {e}")

if __name__ == "__main__":
    check_memory_usage()

