
import os
import torch
import torchax
import jax.numpy as jnp
import numpy as np
import jax

# Precision
jax.config.update("jax_default_matmul_precision", "float32")

def main():
    print("Initializing Torchax Mutation Test (Standalone)...")
    
    # 1. Test Direct Mutation on Torchax Tensor
    print("\n--- Test 1: Direct __setitem__ ---")
    env = torchax.default_env()
    # Create a torchax tensor wrapping a JAX array
    jax_arr = jnp.zeros((2, 5), dtype=jnp.float32)
    t_jax = torchax.tensor.Tensor(jax_arr, env)
    
    print(f"Original: {t_jax}")
    
    try:
        # Try to modify in-place
        t_jax[0, 1] = 99.0
        print("t_jax[0, 1] = 99.0 executed without error.")
        
        # Check if it actually changed
        print(f"Modified: {t_jax}")
        
        # Verify internal JAX array (it should be updated if Torchax supports it, 
        # typically by replacing the internal _elem with a new JAX array)
        if t_jax._elem[0, 1] == 99.0:
             print("SUCCESS: Internal JAX array updated.")
        else:
             print("FAILURE: Internal JAX array NOT updated (Silent failure?).")
             
    except Exception as e:
        print(f"FAILURE: __setitem__ raised exception: {e}")

    # 2. Test Slice Mutation (used by StaticCache)
    print("\n--- Test 2: Slice Assignment ---")
    jax_arr_2 = jnp.zeros((2, 5), dtype=jnp.float32)
    t_jax_2 = torchax.tensor.Tensor(jax_arr_2, env)
    
    try:
        # t_jax_2[:, 2] = 88.0
        # PyTorch slicing often creates a view. 
        # Torchax might not support view-based mutation.
        t_jax_2[:, 2] = 88.0
        print("t_jax_2[:, 2] = 88.0 executed.")
        print(f"Modified: {t_jax_2}")
        
        if t_jax_2._elem[0, 2] == 88.0 and t_jax_2._elem[1, 2] == 88.0:
             print("SUCCESS: Slice assignment updated values.")
        else:
             print("FAILURE: Slice assignment did NOT update values.")
             
    except Exception as e:
        print(f"FAILURE: Slice assignment raised exception: {e}")
        
    # 3. Test JIT Compilation with Mutation (Mock Cache)
    print("\n--- Test 3: JIT Compiled Mutation (Mock Cache) ---")
    
    class MockCache(torch.nn.Module):
        def __init__(self):
            super().__init__()
            # Register as buffer so it's state
            self.register_buffer("cache", torch.zeros(2, 5))
            
        def forward(self, x, idx):
            # x: [2, 1]
            # In-place update
            self.cache[:, idx] = x.squeeze(-1)
            return self.cache

    model = MockCache()
    # Wrap in Torchax
    # We need to manually wrap params if we were doing full emulation, 
    # but let's see if torchax.compile handles the Module logic.
    
    # Since torchax.compile usually expects a stateless function or handles state via functionalization,
    # in-place mutation of buffers inside forward might be tricky.
    
    # We use a functional approach wrapper similar to demo_tpu
    wrapper = torchax.compile(model)
    
    inp = torch.tensor([[7.0], [7.0]]) # [2, 1]
    idx = 3
    
    try:
        print("Running compiled forward...")
        # Note: torchax.compile returns a function that might return (output, new_state) or similar?
        # Or it updates state in place if it's a wrapper?
        # demo_tpu.py uses a wrapper that delegates.
        
        res = wrapper(inp, idx)
        print("Execution finished.")
        print(f"Result (Should have 7.0 at col 3):\n{res}")
        
    except Exception as e:
        print(f"JIT Execution Failed: {e}")

if __name__ == "__main__":
    main()
