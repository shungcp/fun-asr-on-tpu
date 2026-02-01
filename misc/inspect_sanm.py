import inspect
import torch
import sys

try:
    from funasr.models.sense_voice.model import MultiHeadedAttentionSANM
    print("Successfully imported MultiHeadedAttentionSANM")
    
    print("\n=== Source for MultiHeadedAttentionSANM.forward ===")
    print(inspect.getsource(MultiHeadedAttentionSANM.forward))
    
    print("\n=== Source for MultiHeadedAttentionSANM.forward_fsmn ===")
    print(inspect.getsource(MultiHeadedAttentionSANM.forward_fsmn))
    
    print("\n=== Source for MultiHeadedAttentionSANM.forward_attention ===")
    print(inspect.getsource(MultiHeadedAttentionSANM.forward_attention))

    print("\n=== Source for MultiHeadedAttentionSANM.__init__ ===")
    print(inspect.getsource(MultiHeadedAttentionSANM.__init__))

    print("\n=== Attempting to instantiate and run MultiHeadedAttentionSANM ===")
    sig = inspect.signature(MultiHeadedAttentionSANM.__init__)
    print(f"Constructor signature: {sig}")

    # Args: n_head, in_feat, n_feat, dropout_rate, kernel_size
    n_head = 4
    d_model = 64 # Small for testing
    kernel_size = 15
    # (n_head, in_feat, n_feat, dropout_rate, kernel_size)
    model = MultiHeadedAttentionSANM(n_head, d_model, d_model, 0.1, kernel_size)
    model.eval() # Ensure deterministic execution for verification
    print("Successfully instantiated MultiHeadedAttentionSANM")
    
    print("\n=== Source for MultiHeadedAttentionSANM.forward_qkv ===")
    try:
        print(inspect.getsource(MultiHeadedAttentionSANM.forward_qkv))
    except AttributeError:
        # Maybe it inherits or is defined elsewhere, or dynamic? 
        # Actually it might be 'linear_q_k_v' usage in forward.
        print("Could not find forward_qkv source directly.")

    # Run a dummy forward pass on CPU first
    B, T = 1, 50
    x = torch.randn(B, T, d_model)
    mask = torch.ones(B, 1, T) # Simple mask
    
    print("Running forward pass on CPU...")
    # forward signature: (x, mask, mask_shfit_chunk=None, mask_att_chunk_encoder=None)
    out = model(x, mask)
    print(f"CPU Output shape: {out.shape}")

    # Check for TPU availability and run
    try:
        import torchax
        import jax.numpy as jnp
        import numpy as np
        import jax
        
        # Enforce float32 precision for JAX matmuls
        jax.config.update("jax_default_matmul_precision", "float32")

        
        print(f"\nMoving directly to TPU via torchax...")
        
        # 1. Compile the model
        # We need to compile the module. torchax.compile returns a JIT-compiled version.
        # Note: We should put the model in eval mode if not already (it is).
        print("Compiling model with torchax...")
        # Since strict compilation might fail on control flow if not handled, 
        # but SANM seems to have basic if checks on None which might be static if passed as args?
        # Actually, if we pass tensor inputs, 'if mask is not None' works for Tracers.
        
        # We need a wrapper to ensure we can convert params easily
        compiled_model = torchax.compile(model)
        
        # 2. Convert Parameters to TPU
        env = torchax.default_env()
        
        def convert_params(compiled_obj):
            def convert_tensor(v):
                try:
                    # Cast bfloat16 to float32 for numpy compatibility
                    if v.dtype == torch.bfloat16:
                        v_conv = v.detach().to(torch.float32).cpu().numpy()
                    else:
                        v_conv = v.detach().cpu().numpy()
                    jax_arr = jnp.array(v_conv)
                    return torchax.tensor.Tensor(jax_arr, env)
                except Exception as e:
                    print(f"Failed to convert tensor: {e}")
                    return v

            count = 0
            if hasattr(compiled_obj, 'params'):
                for k, v in list(compiled_obj.params.items()):
                    if isinstance(v, torch.Tensor):
                        compiled_obj.params[k] = convert_tensor(v)
                        count += 1
            print(f"Converted {count} parameters to TPU.")
            
            # Also buffers if any
            if hasattr(compiled_obj, 'buffers'):
                for k, v in list(compiled_obj.buffers.items()):
                    if isinstance(v, torch.Tensor):
                        compiled_obj.buffers[k] = convert_tensor(v)

        convert_params(compiled_model)
        
        # 3. Helper to wrap inputs
        def wrap_input(x):
            with env:
                npy = x.detach().cpu().numpy()
                jax_arr = jnp.array(npy)
                return torchax.tensor.Tensor(jax_arr, env)

        x_tpu = wrap_input(x)
        mask_tpu = wrap_input(mask)
        
        print(f"Inputs wrapped. Running forward pass on TPU...")
        
        try:
            # Run
            out_tpu = compiled_model(x_tpu, mask_tpu)
            
            # Unpack if needed (if it returns a Tensor directly)
            print(f"TPU Output type: {type(out_tpu)}")
            if isinstance(out_tpu, torchax.tensor.Tensor):
                print(f"TPU Output shape: {out_tpu.shape}")
                
                # Compare with CPU result
                out_cpu_ref = out.detach().numpy()
                out_tpu_npy = np.array(out_tpu._elem)
                
                diff = np.abs(out_cpu_ref - out_tpu_npy).max()
                print(f"Max difference between CPU and TPU: {diff}")
            else:
                print(f"TPU Output: {out_tpu}")

        except Exception as e:
            print(f"TPU Run Failed: {e}")
            import traceback
            traceback.print_exc()

            print("\nDebug: checking if we can just run a simple op to verify env...")
            # Simple check
            with env:
                res = x_tpu + x_tpu
                print(f"x_tpu + x_tpu shape: {res.shape}")

            # Or we can test the specific operations we saw in source:
            # 1. forward_fsmn: reshape mask, transpose inputs
            # 2. matmul for attention
            
            # Let's try to replicate the forward pass steps on TPU tensors
            # Inputs to TPU
            x_tpu = torch.randn(B, T, d_model, device='cpu') # Torchax will handle conversion if we are careful?
            # Actually torchax usually works by hijacking torch ops or we use jax arrays and wrap them?
            # The user's previous test `check_tpu_transpose.py` used `env` context and `torch.randn` which might have stayed on CPU 
            # unless `torchax` automatically moves them?
            # Wait, in `check_tpu_transpose.py` the user did:
            # q_cpu = torch.randn(...)
            # ... and then used it. Torchax context might interpret torch ops as XLA ops if inputs are XLA tensors?
            # No, `check_tpu_transpose.py` output says: "Input TPU Tensor: torch.Size(...) (backed by ...)"
            # It seems they just used torch ops.
            
            # Let's just finish the inspection script to create the model and run it if possible, 
            # but since `model` is a PyTorch nn.Module, torchax won't automatically compile it unless we use a specific API.
            # So just ending here is fine for now to confirm the structure.
            pass

    except ImportError:
        print("torchax not found, skipping TPU run")
    except Exception as e:
        print(f"TPU Setup Error: {e}")
    
except ImportError as e:
    print(f"Could not import directly: {e}")
    # Try importing via auto model if path is standard
    try:
        import funasr
        print(f"funasr version: {funasr.__version__}")
        print(f"funasr path: {funasr.__file__}")
    except:
        pass
except Exception as e:
    print(f"Error during inspection/execution: {e}")
