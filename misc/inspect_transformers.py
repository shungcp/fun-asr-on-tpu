import torch
from transformers import StaticCache
try:
    from transformers.cache_utils import StaticLayer
    print(f"StaticLayer found: {StaticLayer}")
except ImportError:
    print("StaticLayer NOT found in cache_utils directly")

print("Inspecting StaticCache...")
try:
    cache = StaticCache(config=None, max_batch_size=4, max_cache_len=128, device="cpu", dtype=torch.float32)
    print(f"StaticCache created. attributes: {dir(cache)}")
    if hasattr(cache, 'layers'):
        print(f"cache.layers type: {type(cache.layers)}")
        if len(cache.layers) > 0:
            print(f"cache.layers[0] type: {type(cache.layers[0])}")
            print(f"cache.layers[0] attributes: {dir(cache.layers[0])}")
    
    # Trigger the property check if it exists
    try:
        mbs = cache.max_batch_size
        print(f"cache.max_batch_size = {mbs}")
    except Exception as e:
        print(f"Accessing max_batch_size failed: {e}")

except Exception as e:
    print(f"Failed to instantiate StaticCache: {e}")
