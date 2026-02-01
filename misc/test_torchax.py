import torch
import torch.nn as nn
import torchax
import os

# Set Env Var before anything else if possible, though mostly for XLA
os.environ["PJRT_DEVICE"] = "TPU"

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)

    def forward(self, x):
        return self.linear(x)

def main():
    print("Initializing Torchax...")
    
    # Enable globally to ensure dispatch works
    # if hasattr(torchax, 'enable_globally'):
    #    print("Enabling globally...")
    #    torchax.enable_globally()
        
    # Try enabling performance mode which might set up the environment/dispatch keys
    if hasattr(torchax, 'enable_performance_mode'):
        print("Enabling performance mode...")
        torchax.enable_performance_mode()
    
    # Check default env type
    d_env = torchax.default_env
    if callable(d_env):
        print("Calling default_env()...")
        d_env = d_env()
    print(f"Env: {d_env}")

    print("Inspect modules...")
    print(f"torchax.tensor dir: {dir(torchax.tensor)}")
    # print(f"help(torchax.compile): {help(torchax.compile)}") # capture output effectively
    
    model = SimpleModel()
    # Unfreeze just in case
    for p in model.parameters():
        p.requires_grad = True
        
    print("Compiling model...")
    compiled_model = torchax.compile(model)
    
    print("Checking compiled parameters...")
    if hasattr(compiled_model, 'params'):
        raw_count = 0
        converted_count = 0
        import jax.numpy as jnp
        
        # Get Environment
        env = getattr(compiled_model, 'env', None)
        if env is None:
             env = torchax.default_env() if callable(torchax.default_env) else torchax.default_env

        for k, v in list(compiled_model.params.items()):
            if "Parameter" in str(type(v)):
                raw_count += 1
                try:
                    print(f"Manually converting {k}...")
                    npy = v.detach().cpu().numpy()
                    jax_arr = jnp.array(npy)
                    # Check Tensor constructor signature by trying default
                    # torchax.tensor.Tensor(elem, env)
                    tx_tensor = torchax.tensor.Tensor(jax_arr, env)
                    compiled_model.params[k] = tx_tensor
                    converted_count += 1
                except Exception as e:
                    print(f"Failed to convert {k}: {e}")
            else:
                converted_count += 1
        
        print(f"Result after patch: {raw_count} original raw Parameters (converted {converted_count})")
    else:
        print("WARNING: compiled_model has no 'params' attribute.")

    # Try inference
    x = torch.randn(1, 10)
    print("Running inference...")
    try:
        # Wrap input x as well, as proper context dispatch might be missing
        import jax.numpy as jnp
        if hasattr(compiled_model, 'env'):
             env = compiled_model.env
        else:
             env = torchax.default_env() if callable(torchax.default_env) else torchax.default_env
             
        tx_x = torchax.tensor.Tensor(jnp.array(x.detach().cpu().numpy()), env)
        
        y = compiled_model(tx_x)
        print("Inference successful.")
        print(f"Output type: {type(y)}")
    except Exception as e:
        print(f"Inference failed: {e}")

if __name__ == "__main__":
    main()
