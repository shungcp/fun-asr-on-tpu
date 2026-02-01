import time
import torch
import os
import types
from funasr import AutoModel
from transformers import AutoTokenizer

# Force CPU
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PJRT_DEVICE"] = "CPU" 

def test_cpu_full():
    print("Loading Full Model on CPU...")
    model_dir = "FunAudioLLM/Fun-ASR-Nano-2512"
    model = AutoModel(
        model=model_dir,
        trust_remote_code=True,
        remote_code="./model.py",
        device="cpu",
        hub="ms",
        disable_update=True
    )
    
    # Test 1: ASR
    wav_path = f"{model.model_path}/example/zh.mp3"
    print(f"\n--- Test 1: Audio Inference on {wav_path} (CPU) ---")
    
    t0 = time.time()
    res = model.generate(
        input=wav_path,
        batch_size=1,
        hotwords=[], 
        )
    print(f"ASR Latency: {time.time() - t0:.4f}s")
    print(f"ASR Result: {res}")
    
    # Inspect Embeddings (Hook)
    print("Inspecting CPU embeddings stats...")
    
    # We capture the inputs to the Qwen LLM
    # model.model.llm is the Qwen3ForCausalLM
    # Its forward signature is (inputs_embeds=..., etc)
    # We can hook the forward method of the model
    
    captured = {}
    def hook_forward(module, input, output):
        # input might be a tuple.
        # Qwen forward: input_ids=None, past_key_values=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None, ...
        # But we are calling it via kwargs in model.py? 
        # model.py: self.llm(inputs_embeds=inputs_embeds, ...)
        # Hooking module forward often gets positional args.
        pass

    # Easier: Monkey patch the generate function of the inner LLM or the inference_llm of nano
    original_inference_llm = model.model.inference_llm
    
    def inference_llm_wrapper(self, data_in, data_lengths=None, key=None, tokenizer=None, frontend=None, **kwargs):
        # Call prepare to get embeddings
        inputs_embeds, _, _, _, _ = self.inference_prepare(
            data_in, data_lengths, key, tokenizer, frontend, **kwargs
        )
        print(f"CPU Embeddings (Hook): Shape={inputs_embeds.shape}, Min={inputs_embeds.min():.4f}, Max={inputs_embeds.max():.4f}, Mean={inputs_embeds.mean():.4f}")
        
        # Call original
        return original_inference_llm(data_in, data_lengths, key, tokenizer, frontend, **kwargs)
        
    model.model.inference_llm = types.MethodType(inference_llm_wrapper, model.model)
    
    # Run again to trigger hook
    model.generate(
        input=wav_path,
        batch_size=1, 
        hotwords=[], 
        language="中文",
        itn=True
    )
    
    # Test 2: Pure Text Generation (Sanity Check)
    print("\n--- Test 2: Pure Text Sanity Check (1+1=?) ---")
    try:
        # Construct actual Qwen path (subdirectory)
        qwen_path = os.path.join(model.model_path, "Qwen3-0.6B")
        tokenizer = AutoTokenizer.from_pretrained(qwen_path, trust_remote_code=True)
        # Qwen instruct format
        prompt = "<|im_start|>user\n1+1=?<|im_end|>\n<|im_start|>assistant\n"
        inputs = tokenizer(prompt, return_tensors="pt")
        
        # Run on CPU via internal LLM
        llm = model.model.llm
        t1 = time.time()
        out = llm.generate(**inputs, max_new_tokens=10)
        print(f"Text Latency: {time.time() - t1:.4f}s")
        print(f"Text Output: {tokenizer.decode(out[0], skip_special_tokens=False)}")
        
    except Exception as e:
        print(f"Text Gen Failed: {e}")

if __name__ == "__main__":
    test_cpu_full()
