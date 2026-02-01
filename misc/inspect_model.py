
import os
import torch
from funasr import AutoModel

# Minimal load
model_dir = "/home/admin_shunwang_altostrat_com/.cache/modelscope/hub/models/FunAudioLLM/Fun-ASR-Nano-2512"
if not os.path.exists(model_dir):
    model_dir = "FunAudioLLM/Fun-ASR-Nano-2512"

print(f"Loading model from {model_dir}...")
model = AutoModel(
    model=model_dir,
    trust_remote_code=True,
    remote_code="./model.py",
    device="cpu",
    hub="ms",
    disable_update=True
)

nano_model = model.model
encoder = nano_model.audio_encoder

print("\n--- Searching for 'fsmn_memory' in encoder ---")
found = False
for name, module in encoder.named_modules():
    if hasattr(module, 'fsmn_memory'):
        print(f"FOUND: {name}.fsmn_memory -> {type(getattr(module, 'fsmn_memory'))}")
        val = getattr(module, 'fsmn_memory')
        if isinstance(val, torch.Tensor):
            print(f"  Shape: {val.shape}, Dtype: {val.dtype}, Device: {val.device}")
        found = True
    # Check if it might be in __dict__ but not a submodule attribute (unlikely if named_modules iterates it, but maybe as a buffer?)
    # Buffers are attributes.

if not found:
    print("NOT FOUND via named_modules(). Checking recursively in __dict__...")
    # Manual walk
    def walk(m, path):
        for k, v in m.__dict__.items():
            if k == 'fsmn_memory':
                print(f"FOUND in __dict__: {path}.{k} -> {type(v)}")
                if isinstance(v, torch.Tensor):
                     print(f"  Shape: {v.shape}, Dtype: {v.dtype}, Device: {v.device}")
            if isinstance(v, torch.nn.Module):
                walk(v, f"{path}.{k}")
    
    walk(encoder, "audio_encoder")
