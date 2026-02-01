import time
import torch
import os
import types
from funasr import AutoModel

# Force CPU
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PJRT_DEVICE"] = "CPU" 

def test_cpu_encoder():
    print("Loading Encoder on CPU...")
    model_dir = "FunAudioLLM/Fun-ASR-Nano-2512"
    model = AutoModel(
        model=model_dir,
        trust_remote_code=True,
        remote_code="./model.py",
        device="cpu",
        hub="ms"
    )
    
    wav_path = f"{model.model_path}/example/zh.mp3"
    print(f"Running inference on {wav_path} (CPU)...")
    
    # Monkey-patch inference_llm to stop after inference_prepare
    # and return the timing info or just return dummy
    original_inference_llm = model.model.inference_llm
    
    def mocked_inference_llm(self, *args, **kwargs):
        # We assume inference_prepare is called inside or we measure it here 
        # But inference_llm calls inference_prepare
        # So we can copy the logic of inference_llm up to inference_prepare
        
        # Or better, we wrap inference_prepare to measure time?
        # But we want to avoid running the heavy LLM part.
        
        # Let's peek implementation of inference_llm in model.py
        # It does: inputs_embeds, contents, batch, source_ids, meta_data = self.inference_prepare(...)
        # Then t_llm_start = time.time() ...
        
        # We will just run inference_prepare and return
        start_t = time.time()
        res = self.inference_prepare(*args, **kwargs)
        end_t = time.time()
        print(f"Encoder+Prepare Latency: {end_t - start_t:.4f}s")
        return [{"text": "BENCHMARK_DONE"}]

    model.model.inference_llm = types.MethodType(mocked_inference_llm, model.model)

    # Warmup + Benchmark
    for i in range(5):
        print(f"--- Run {i} ---")
        t0 = time.time()
        res = model.generate(
            input=[wav_path],
            batch_size=1,
            hotwords=[],
            language="中文",
            itn=True
        )
        # Time is printed inside mock

if __name__ == "__main__":
    test_cpu_encoder()
