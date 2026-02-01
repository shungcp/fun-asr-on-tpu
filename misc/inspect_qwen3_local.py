
import sys
import inspect

try:
    import transformers.models.qwen3.modeling_qwen3 as m
    if hasattr(m, 'eager_attention_forward'):
        print("FOUND eager_attention_forward")
        print(inspect.getsource(m.eager_attention_forward))
    else:
        print("NOT FOUND eager_attention_forward in modeling_qwen3")
except ImportError:
    print("ImportError: transformers.models.qwen3.modeling_qwen3 not found")
except Exception as e:
    print(f"Error: {e}")
