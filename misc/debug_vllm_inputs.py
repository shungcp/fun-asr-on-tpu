import inspect
from typing import get_type_hints

def debug():
    try:
        import vllm.inputs.data
        import vllm.inputs
        
        print("--- vllm.inputs.data content ---")
        print(dir(vllm.inputs.data))
        
        if hasattr(vllm.inputs.data, 'EmbedsPrompt'):
            cls = vllm.inputs.data.EmbedsPrompt
            print(f"\nEmbedsPrompt Class: {cls}")
            print(f"Type: {type(cls)}")
            print(f"Is TypedDict? {getattr(cls, '__required_keys__', 'N/A')}")
            
        print("\n--- vllm.inputs content ---")
        # Check if it is exposed here
        if hasattr(vllm.inputs, 'EmbedsPrompt'):
             print(f"vllm.inputs.EmbedsPrompt: {vllm.inputs.EmbedsPrompt}")
             print(f"Is Same? {vllm.inputs.EmbedsPrompt is vllm.inputs.data.EmbedsPrompt}")
             
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    debug()
