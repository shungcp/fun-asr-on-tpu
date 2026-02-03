import os
import transformers.masking_utils
# AGGRESSIVE PATCH: Disable is_compileable to force legacy masking BEFORE any other usage
def return_false(*args, **kwargs): return False
transformers.masking_utils.is_compileable = return_false

# -----------------------------------------------------------------------------
# RUNTIME MONKEYPATCH: Force use of local modeling_qwen3 (with TPU fixes)
# -----------------------------------------------------------------------------
import sys
import os
sys.path.insert(0, os.path.dirname(__file__)) # Allow importing local files

try:
    import modeling_qwen3
    import transformers.models.qwen3.modeling_qwen3 as hf_qwen3

    print("[Patch] Replacing transformers Qwen3 impl with local fixed version...")
    hf_qwen3.Qwen3Model = modeling_qwen3.Qwen3Model
    hf_qwen3.Qwen3Attention = modeling_qwen3.Qwen3Attention
    hf_qwen3.Qwen3DecoderLayer = modeling_qwen3.Qwen3DecoderLayer
    hf_qwen3.Qwen3ForCausalLM = modeling_qwen3.Qwen3ForCausalLM
except Exception as e:
    print(f"[Patch] Failed to patch Qwen3: {e}")
# -----------------------------------------------------------------------------

import torch

# -----------------------------------------------------------------------------
# ROBUST MASK PATCH (Fixes vmap crash and recursion)
# -----------------------------------------------------------------------------
def robust_mask_fn(input_ids_shape=None, dtype=None, device=None, past_key_values_length=0, attention_mask=None, **kwargs):
    """
    Simple, unbreakable causal mask generation.
    Ignores complex transformers logic to avoid TPU/vmap bugs.
    """
    if input_ids_shape is None:
        input_ids_shape = kwargs.get("input_tensor", None)
        if input_ids_shape is not None and hasattr(input_ids_shape, "shape"):
             input_ids_shape = input_ids_shape.shape
        elif input_ids_shape is None:
             # Fallback from other kwargs
             bs = kwargs.get("batch_size", 1)
             sl = kwargs.get("target_length", kwargs.get("seq_length", 1))
             input_ids_shape = (bs, sl)

    if dtype is None: dtype = torch.float32
    if device is None: 
         if attention_mask is not None: device = attention_mask.device
         else: device = torch.device("cpu")

    bs = input_ids_shape[0]
    target_len = input_ids_shape[1]

    # Handle attention_mask (Padding) if provided
    padding_mask = None
    if attention_mask is not None:
         # Ensure bool or equivalent for conversion
         if attention_mask.dim() == 2: attention_mask = attention_mask[:, None, None, :]
         elif attention_mask.dim() == 3: attention_mask = attention_mask[:, None, :, :]

         # Convert to additive mask (0.0 for keep, min_val for mask)
         min_val = torch.finfo(dtype).min
         if attention_mask.dtype == torch.bool:
             padding_mask = torch.zeros_like(attention_mask, dtype=dtype)
             padding_mask.masked_fill_(~attention_mask, min_val)
         else:
             padding_mask = (1.0 - attention_mask.to(dtype)) * min_val

    # Create Causal Triangle
    min_val = torch.finfo(dtype).min
    mask = torch.full((target_len, target_len), min_val, device=device, dtype=dtype)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (torch.arange(mask.size(-2), device=device) + past_key_values_length).view(-1, 1), 0)

    mask = mask[None, None, :, :]

    if padding_mask is not None:
        # Broadcast/Slice logic could be complex, but for now we assume shape compatibility or rely on broadcasting
        # If padding_mask is (B, 1, 1, L), it adds to (1, 1, L, L) -> (B, 1, L, L)
        # We need to be careful about shapes.
        # Let's trust 'attention_mask' is correct shape from upstream.
        return padding_mask + mask

    return mask

# Apply Patch IMMEDIATELY
import transformers.masking_utils
transformers.masking_utils.eager_mask = robust_mask_fn
transformers.masking_utils.sdpa_mask = robust_mask_fn
transformers.masking_utils.sdpa_mask_recent_torch = robust_mask_fn
transformers.masking_utils.create_causal_mask = robust_mask_fn
transformers.masking_utils.is_compileable = lambda *args, **kwargs: False
print("[Patch] Applied ROBUST MASK FN to all transformers masking utilities.")

# -----------------------------------------------------------------------------

import torch
import torchax
import jax
import time
import types
import copy
from funasr import AutoModel
import tpu_patch
from torchax_utils import TorchaxWrapper
import logging
import traceback
from transformers.cache_utils import StaticCache
from transformers.cache_utils import StaticCache

# Monkeypatch StaticCache to avoid transformers vmap masking bug
# This forces transformers to use legacy masking which is safer here.
if hasattr(StaticCache, "is_compileable"):
    StaticCache.is_compileable = False

class CacheLayerPair:
    """Helper to mimic the layer object expected by transformers StaticCache"""
    def __init__(self, key, value):
        self.keys = key
        self.values = value

    @property
    def is_compileable(self):
        return False

    def get_mask_sizes(self, cache_position) -> tuple[int, int]:
        return (self.keys.shape[-2], 0)

    def get_seq_length(self, layer_idx=0) -> int:
        return self.keys.shape[-2]

    @property
    def max_batch_size(self):
        return self.keys.shape[0]

    @property
    def max_cache_len(self):
        return self.keys.shape[2]

    def reset(self):
        self.keys.zero_()
        self.values.zero_()

    def __iter__(self):
        yield self.keys
        yield self.values

class PatchedStaticCache(StaticCache):
    """
    A patched version of StaticCache that supports tuple layers (key, value)
    or CacheLayerPair objects.
    """
    def __init__(self, config, max_batch_size=1, max_cache_len=None, device=None, dtype=None):
        # StaticCache signature: (self, config, max_cache_len, offloading=False, offload_only_non_sliding=True, **kwargs)
        # We need to adapt our calling convention to the superclass.
        # max_batch_size, device, and dtype are NOT in super().__init__ in this version.

        # Resolve max_cache_len before calling super, as super creates the buffers.
        if max_cache_len is None:
            if hasattr(config, "static_sequence_length"):
                max_cache_len = config.static_sequence_length
            else:
                max_cache_len = config.max_position_embeddings

        # StaticCache signature: (self, config, max_cache_len, ...)
        super().__init__(config, max_cache_len=max_cache_len)

        # Manually store what super() might miss if we need them, 
        # but StaticCache usually just sets up based on config/max_cache_len.
        self._max_batch_size = max_batch_size # Store it ourselves if super doesn't
        self.device = device
        self.dtype = dtype

        self._seen_tokens = 0
        # self.max_cache_len is a property in StaticCache, so we cannot set it.
        # It should be correct since we passed max_cache_len to super().__init__.

    @property
    def seen_tokens(self):
        print(f"[Debug] Accessing seen_tokens: {self._seen_tokens} (id={id(self)})")
        return self._seen_tokens

    @seen_tokens.setter
    def seen_tokens(self, value):
        print(f"[Debug] Setting seen_tokens: {value} (id={id(self)})")
        self._seen_tokens = value

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        if hasattr(self, "layers"):
            if layer_idx < len(self.layers):
                layer = self.layers[layer_idx]
                k_cache = None
                v_cache = None

                if isinstance(layer, tuple):
                    k_cache, v_cache = layer
                elif hasattr(layer, "keys") and hasattr(layer, "values"):
                    k_cache = layer.keys
                    v_cache = layer.values

                if k_cache is not None and v_cache is not None:
                    if cache_kwargs is not None and "cache_position" in cache_kwargs:
                        cache_position = cache_kwargs["cache_position"]

                        k_tensor = k_cache
                        if not isinstance(k_tensor, torch.Tensor) and hasattr(k_tensor, 'keys'):
                            k_tensor = k_tensor.keys

                        v_tensor = v_cache
                        if not isinstance(v_tensor, torch.Tensor) and hasattr(v_tensor, 'values'):
                            v_tensor = v_tensor.values

                        k_tensor.index_copy_(2, cache_position, key_states)
                        v_tensor.index_copy_(2, cache_position, value_states)

                        if layer_idx == 0:
                            if cache_position.numel() > 0:
                                # We cannot update seen_tokens here inside JIT if cache_position is traced.
                                # Relies on external update (in patched_forward).
                                try:
                                    # Try catch for Eager mode, but skip for JIT
                                    if cache_position.device.type == 'cpu':
                                        current_max_pos = cache_position.max().item() + 1
                                        if isinstance(self.seen_tokens, int):
                                             self.seen_tokens = current_max_pos
                                        elif current_max_pos > self.seen_tokens:
                                             self.seen_tokens = current_max_pos
                                except:
                                    pass
                            else:
                                pass # No tokens updated, seen_tokens remains unchanged

                        return k_cache, v_cache

        return key_states, value_states

    def get_seq_length(self, layer_idx=0):
        # Fix: Must return actual seen_tokens for autoregressive generation!
        val = 0
        if hasattr(self, "seen_tokens"):
            val = self.seen_tokens
        elif hasattr(self, "key_cache") and len(self.key_cache) > layer_idx:
            val = self.key_cache[layer_idx].shape[-2]

        # print(f"[Debug] get_seq_length: {val}") 
        print(f"[Debug] get_seq_length: {val} (id={id(self)})") 
        return val

    def reset(self):
        """Reset the cache state and zero out buffers."""
        self.seen_tokens = 0
        self._position_ids = None

        # Reset buffers
        if hasattr(self, "key_cache") and self.key_cache:
            for k in self.key_cache:
                if isinstance(k, torch.Tensor): k.zero_()
            for v in self.value_cache:
                if isinstance(v, torch.Tensor): v.zero_()

        if hasattr(self, "layers") and self.layers:
            for item in self.layers:
                if hasattr(item, "reset"):
                    item.reset()
                elif isinstance(item, (tuple, list)) and len(item) == 2:
                    if isinstance(item[0], torch.Tensor): item[0].zero_()
                    if isinstance(item[1], torch.Tensor): item[1].zero_()

class LLMForwardWrapper(torch.nn.Module):
    def __init__(self, llm, max_cache_len=128):
        super().__init__()
        self.llm = llm
        self.max_cache_len = max_cache_len

    def forward(self, inputs_embeds, attention_mask, position_ids, *past_key_values_flat):
        device = inputs_embeds.device
        dtype = self.llm.config.torch_dtype if hasattr(self.llm.config, 'torch_dtype') else torch.float32

        pkv = PatchedStaticCache(
            config=self.llm.config,
            max_batch_size=inputs_embeds.shape[0], # Dynamic batch size
            max_cache_len=self.max_cache_len,
            device=device,
            dtype=dtype
        )
        if position_ids is not None:
            pkv._position_ids = position_ids

        num_layers = len(past_key_values_flat) // 2
        keys = list(past_key_values_flat[0::2])
        values = list(past_key_values_flat[1::2])

        if hasattr(pkv, "key_cache"):
            pkv.key_cache = keys
            pkv.value_cache = values

        if hasattr(pkv, "layers"):
            pkv.layers = [CacheLayerPair(k, v) for k, v in zip(keys, values)]

        if pkv._position_ids is not None:
             if hasattr(pkv._position_ids, 'shape') and pkv._position_ids.numel() > 0:
                  start_pos = pkv._position_ids[..., 0].flatten()[0]
             else:
                  start_pos = 0
        else:
             start_pos = 0

        input_len = inputs_embeds.shape[1]
        cache_position = torch.arange(input_len, device=device) + start_pos

        outputs = self.llm(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=pkv,
            use_cache=True,
            cache_position=cache_position,
            return_dict=False,
            output_attentions=False,
            output_hidden_states=False,
        )

        logits = outputs[0]
        new_pkv = outputs[1]

        new_pkv_flat = []
        for k, v in new_pkv:
            new_pkv_flat.extend([k, v])

        return logits, *new_pkv_flat

# TPU Setup
os.environ["PJRT_DEVICE"] = "TPU"
jax.config.update("jax_default_matmul_precision", "float32")

def main():
    print("--- TPU Demo Clean Start ---")

    GENERATION_MAX_LENGTH = 16

    # 1. Apply Patches
    tpu_patch.apply_tpu_patches()

    # 2. Load Model (CPU)
    model_dir = "~/.cache/modelscope/hub/models/FunAudioLLM/Fun-ASR-Nano-2512"
    if not os.path.exists(model_dir):
        model_dir = "FunAudioLLM/Fun-ASR-Nano-2512"

    print(f"Loading model from {model_dir}...")
    model = AutoModel(
        model=model_dir,
        trust_remote_code=True,
        remote_code="./model.py",
        device="cpu", 
        hub="ms",
        disable_update=True,
    )

    nano_model = model.model

    # 3. Wrap Encoder with Torchax
    print("Compiling Audio Encoder with Torchax...")
    nano_model.audio_encoder = TorchaxWrapper(nano_model.audio_encoder)

    # 4. Wrap LLM with Torchax
    print("Compiling LLM with Torchax...")
    try:
        # Force eager attention
        if hasattr(nano_model.llm.config, "_attn_implementation"):
             nano_model.llm.config._attn_implementation = 'eager'
        if hasattr(nano_model.llm.config, "attn_implementation"):
             nano_model.llm.config.attn_implementation = 'eager'

        # Patch config for transformers StaticCache compatibility
        if not hasattr(nano_model.llm.config, "get_text_config"):
            def get_text_config(self, decoder=False, **kwargs):
                return self
            nano_model.llm.config.get_text_config = types.MethodType(get_text_config, nano_model.llm.config)

        llm_clone = copy.copy(nano_model.llm)
        llm_wrapper = LLMForwardWrapper(llm_clone, max_cache_len=GENERATION_MAX_LENGTH)
        llm_helper = TorchaxWrapper(llm_wrapper)

        # Patch forward to use helper
        def patched_forward(self, *args, **kwargs):
            input_ids = kwargs.get('input_ids')
            if input_ids is None and len(args) > 0:
                 input_ids = args[0]

            inputs_embeds = kwargs.get('inputs_embeds', None)

            if inputs_embeds is None and input_ids is not None:
                embed_fn = self.model.get_input_embeddings()
                inputs_embeds = embed_fn(input_ids)

            if inputs_embeds is None:
                print(f"[Debug] patched_forward inputs_embeds is None! input_ids: {input_ids.shape if input_ids is not None else 'None'}")
                raise ValueError("Both input_ids and inputs_embeds are None in patched_forward")
            else:
                if len(inputs_embeds.shape) >= 2 and inputs_embeds.shape[1] == 0:
                     print(f"[Debug] patched_forward inputs_embeds has 0 sequence length! {inputs_embeds.shape}")

            past_key_values = kwargs.get('past_key_values', None)
            attention_mask = kwargs.get('attention_mask', None)
            position_ids = kwargs.get('position_ids', None)

            # Debugging logs
            input_len = inputs_embeds.shape[1]
            p_start, p_end = "?", "?"
            if position_ids is not None:
                p_start = position_ids.min().item()
                p_end = position_ids.max().item()

            # --- AUTO-RESET & UPDATE LOGIC ---
            if past_key_values is not None:
                if input_len > 1:
                    # PROMPT PHASE: Force Reset if dirty
                    if hasattr(past_key_values, "seen_tokens") and past_key_values.seen_tokens > 0:
                        if hasattr(past_key_values, "reset"):
                            past_key_values.reset()
                        else:
                            past_key_values.seen_tokens = 0
                            if hasattr(past_key_values, "key_cache"):
                                 for k in past_key_values.key_cache: 
                                     if isinstance(k, torch.Tensor): k.zero_()
                                 if hasattr(past_key_values, "value_cache"):
                                     for v in past_key_values.value_cache:
                                         if isinstance(v, torch.Tensor): v.zero_()

                    # Set seen_tokens = input_len
                    if hasattr(past_key_values, "seen_tokens"):
                        past_key_values.seen_tokens = input_len

                    # Double check self._static_cache
                    if hasattr(self, "_static_cache") and self._static_cache is not None:
                         if self._static_cache is not past_key_values and hasattr(self._static_cache, "reset"):
                             # Ensure internal cache is also synced/reset
                             self._static_cache.reset()
                             # self._static_cache.seen_tokens = input_len
                             pass

                else:
                    # DECODE PHASE: Increment seen_tokens
                    # TPU/JIT update doesn't touch Python attribute, so we do it here.
                    if hasattr(past_key_values, "seen_tokens") and position_ids is not None:
                        current_max = position_ids.max().item()

            # Infer position_ids if not provided
            if position_ids is None:
                effective_attention_mask = attention_mask
                if isinstance(effective_attention_mask, dict):
                    if "attention_mask" in effective_attention_mask:
                        effective_attention_mask = effective_attention_mask["attention_mask"]
                    else:
                        effective_attention_mask = None

                can_use_mask = (effective_attention_mask is not None and hasattr(effective_attention_mask, 'shape'))
                batch_size = inputs_embeds.shape[0]
                input_len = inputs_embeds.shape[1]

                if can_use_mask:
                    position_ids_list = []
                    for b in range(batch_size):
                        valid_len = effective_attention_mask[b].sum().long()
                        start_pos = valid_len - input_len
                        if start_pos < 0: start_pos = 0

                        pos_ids = torch.arange(start_pos, start_pos + input_len, dtype=torch.long, device=inputs_embeds.device)
                        position_ids_list.append(pos_ids)

                    position_ids = torch.stack(position_ids_list)
                else:
                    start_pos = 0
                    if past_key_values is not None:
                        if hasattr(past_key_values, "seen_tokens") and isinstance(past_key_values.seen_tokens, int):
                             start_pos = past_key_values.seen_tokens
                        elif hasattr(past_key_values, "get_seq_length"):
                             start_pos = past_key_values.get_seq_length()
                        elif isinstance(past_key_values, (list, tuple)) and len(past_key_values) > 0:
                             first_layer = past_key_values[0]
                             if isinstance(first_layer, (list, tuple)) and len(first_layer) > 0:
                                  k_tensor = first_layer[0]
                                  if hasattr(k_tensor, 'shape') and len(k_tensor.shape) >= 3:
                                      start_pos = k_tensor.shape[2]

                    position_ids = torch.arange(start_pos, start_pos + input_len, dtype=torch.long, device=inputs_embeds.device)
                    position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

            # Prepare flattened pkv
            pkv_flat_args = []
            if past_key_values is not None and isinstance(past_key_values, StaticCache):
                is_legacy = hasattr(past_key_values, "key_cache")
                needs_init = False
                if is_legacy:
                    if len(past_key_values.key_cache) == 0 or past_key_values.key_cache[0] is None:
                        needs_init = True
                else:
                    if len(past_key_values) > 0:
                        first_layer = past_key_values[0]
                        if first_layer[0] is None:
                             needs_init = True

                if needs_init:
                     config = self.config
                     max_batch_size = 1
                     if hasattr(past_key_values, "max_batch_size"): max_batch_size = past_key_values.max_batch_size
                     elif hasattr(past_key_values, "_max_batch_size"): max_batch_size = past_key_values._max_batch_size

                     max_cache_len = getattr(self.config, "static_sequence_length", GENERATION_MAX_LENGTH)
                     if hasattr(past_key_values, "max_cache_len"): max_cache_len = past_key_values.max_cache_len
                     elif hasattr(past_key_values, "_max_cache_len"): max_cache_len = past_key_values._max_cache_len

                     num_heads = config.num_key_value_heads

                     head_dim = getattr(config, "head_dim", None)
                     if head_dim is None: head_dim = getattr(config, "kv_channels", None)
                     if head_dim is None: head_dim = getattr(config, "attention_head_dim", None)
                     if head_dim is None: head_dim = config.hidden_size // config.num_attention_heads

                     dtype = config.torch_dtype if hasattr(config, "torch_dtype") and config.torch_dtype is not None else torch.float32
                     device = self.device
                     num_layers = config.num_hidden_layers

                     zeros = torch.zeros(max_batch_size, num_heads, max_cache_len, head_dim, dtype=dtype, device=device)

                     if is_legacy:
                         past_key_values.key_cache = [zeros.clone() for _ in range(num_layers)]
                         past_key_values.value_cache = [zeros.clone() for _ in range(num_layers)]
                     else:
                         if hasattr(past_key_values, "layers"):
                             past_key_values.layers = [CacheLayerPair(zeros.clone(), zeros.clone()) for _ in range(num_layers)]

                if is_legacy:
                    for k, v in zip(past_key_values.key_cache, past_key_values.value_cache):
                        pkv_flat_args.extend([k, v])
                else:
                    if hasattr(past_key_values, "layers"):
                         for layer in past_key_values.layers:
                             if isinstance(layer, tuple):
                                 pkv_flat_args.extend(layer)
                             elif hasattr(layer, "keys") and hasattr(layer, "values"):
                                 pkv_flat_args.extend([layer.keys, layer.values])
                             else:
                                 try:
                                     pkv_flat_args.extend(layer)
                                 except TypeError:
                                     pkv_flat_args.append(layer)
                    else:
                        for k, v in past_key_values:
                            pkv_flat_args.extend([k, v])
                if needs_init or not hasattr(self, "_static_cache"):
                     self._static_cache = past_key_values
                     print(f"[Debug] patched_forward: ATTACHED _static_cache to {type(self).__name__} id={id(self)}")

                # Update seen_tokens explicitly from position_ids (since update inside JIT can't do it)
                if position_ids is not None and hasattr(self, "_static_cache") and self._static_cache is not None:
                    # position_ids is a Tensor here (inputs to forward)
                    # print(f"[Debug] patched_forward KV Cache ID: {id(self._static_cache)} Type: {type(self._static_cache)}")
                    try:
                        cur_max = position_ids.max().item() + 1
                        old_s = getattr(self._static_cache, "seen_tokens", 0)
                        # if cur_max > old_s:
                        #     self._static_cache.seen_tokens = cur_max
                        #     print(f"[Debug] patched_forward: Updated seen_tokens {old_s} -> {cur_max} (id={id(self._static_cache)})")
                    except Exception as e:
                        print(f"[Debug] Failed to update seen_tokens: {e}")

            # Call compiled wrapper
            ret = llm_helper(inputs_embeds, attention_mask, position_ids, *pkv_flat_args)

            logits = ret[0]
            new_pkv_flat = ret[1:]

            # Update original StaticCache
            if past_key_values is not None and isinstance(past_key_values, StaticCache):
                try:
                    num_layers = len(new_pkv_flat) // 2
                    for i in range(num_layers):
                         k_new = new_pkv_flat[2*i]
                         v_new = new_pkv_flat[2*i+1]

                         if is_legacy:
                             if i < len(past_key_values.key_cache):
                                 past_key_values.key_cache[i].copy_(k_new)
                                 past_key_values.value_cache[i].copy_(v_new)
                         else:
                             if i < len(past_key_values):
                                 k_old, v_old = past_key_values[i]
                                 if k_old is not None: k_old.copy_(k_new)
                                 if v_old is not None: v_old.copy_(v_new)
                except Exception as e:
                    logging.error(f"Cache update error: {e}")

                try:
                    input_len = inputs_embeds.shape[1]
                    old_seen = getattr(past_key_values, "seen_tokens", "N/A")

                    if old_seen == "N/A":
                         end_pos = position_ids[0, -1].item() + 1
                         setattr(past_key_values, "seen_tokens", end_pos)
                    else:
                         # Only increment if we didn't just hard-set it (i.e. decoding phase)
                         # OR if we want to be safe: just set it to position_ids max + 1?
                         # But position_ids might be None/inferred contentiously.
                         # Just use input_len logic:
                         if input_len > 1:
                             # We already set it in the 'PROMPT PHASE' block above to = input_len
                             # So do nothing here to avoid double counting.
                             pass
                         else:
                             past_key_values.seen_tokens += input_len
                except Exception as e:
                    logging.error(f"Error updating seen_tokens: {e}")

            if hasattr(logits, 'shape'):
                from transformers.modeling_outputs import CausalLMOutputWithPast
                return CausalLMOutputWithPast(logits=logits, past_key_values=past_key_values)

            return ret

        nano_model.llm.forward = types.MethodType(patched_forward, nano_model.llm)
        nano_model.llm.generation_config.cache_implementation = "static"

    except Exception as e:
        print(f"Failed to wrap LLM: {e}")
        traceback.print_exc()

    # Patch inference_prepare
    def patched_inference_prepare(
        self,
        data_in,
        data_lengths=None,
        key: list = None,
        tokenizer=None,
        frontend=None,
        **kwargs,
    ):
        meta_data = {}
        # if kwargs.get("batch_size", 1) > 1:
        #     raise NotImplementedError("batch decoding is not implemented")

        contents = self.data_template(data_in[0])
        dummy_contents = contents

        output = self.data_load_speech(contents, tokenizer, frontend, meta_data=meta_data, **kwargs)
        batch = output 
        speech = batch["speech"]

        if len(speech) > 0:
            if "audio_embedding" in kwargs and "audio_embedding_lens" in kwargs:
                encoder_out = kwargs["audio_embedding"]
                encoder_out_lens = kwargs["audio_embedding_lens"]
            else:
                speech_lengths = batch["speech_lengths"][:, 0]
                if kwargs.get("fp16", False):
                    speech = speech.to(torch.float16)
                elif kwargs.get("bf16", False):
                    speech = speech.to(torch.bfloat16)

                encoder_out, encoder_out_lens = self.encode(speech, speech_lengths)
                adaptor_out, adaptor_out_lens = self.audio_adaptor(encoder_out, encoder_out_lens)
                meta_data["encoder_out"] = encoder_out
                meta_data["encoder_out_lens"] = encoder_out_lens
                meta_data["audio_adaptor_out"] = adaptor_out
                meta_data["audio_adaptor_out_lens"] = adaptor_out_lens

        input_ids = batch["input_ids"]
        source_ids = batch["source_ids"]
        fbank_beg = batch["fbank_beg"]
        fake_token_len = batch["fake_token_len"]

        if not kwargs.get("teacherforcing", False):
            input_ids = source_ids

        input_ids[input_ids < 0] = 0
        inputs_embeds = self.llm.model.get_input_embeddings()(input_ids)
        batch_size, token_num, dims = inputs_embeds.shape

        fake_token_len[fake_token_len < 0] = 0
        fbank_beg[fbank_beg < 0] = 0

        speech_idx = 0
        for batch_idx in range(batch_size):
            for turn_id in range(fbank_beg.shape[1]):
                fbank_beg_idx = fbank_beg[batch_idx, turn_id]
                if fbank_beg_idx > 0:
                    speech_token_len = fake_token_len[batch_idx, turn_id]
                    speech_token = adaptor_out[speech_idx, :speech_token_len, :]

                    try:
                        inputs_embeds[
                            batch_idx,
                            fbank_beg_idx : fbank_beg_idx + speech_token_len,
                            :,
                        ] = speech_token
                    except Exception as e:
                        logging.error(f"{str(e)}, {traceback.format_exc()}")
                        speech_token_len = adaptor_out_lens[speech_idx]
                        speech_token = adaptor_out[speech_idx, :speech_token_len, :]
                        inputs_embeds[
                            batch_idx,
                            fbank_beg_idx : fbank_beg_idx + speech_token_len,
                            :,
                        ] = speech_token

                    speech_idx += 1

        return inputs_embeds, dummy_contents, batch, batch["source_ids"], meta_data 

    nano_model.inference_prepare = types.MethodType(patched_inference_prepare, nano_model)

    # 4. Run Inference
    wav_path = f"{model_dir}/example/zh.mp3"
    print(f"Running inference on {wav_path}...")

    # Define STATIC_SEQUENCE_LENGTH for TPU Bucket (Must encompass Prompt + New Tokens)
    STATIC_SEQUENCE_LENGTH = 128  # Fixed bucket size for compilation (User requested 128)

    # Define Max NEW Tokens to generate
    GENERATION_MAX_NEW_TOKENS = 16 # User requested 16

    # INJECT static_sequence_length into config for modeling_qwen3.py and PatchedStaticCache to pick up
    nano_model.llm.config.static_sequence_length = STATIC_SEQUENCE_LENGTH

    if os.path.exists(wav_path):
        # Run multiple times to verify JIT caching
        for i in range(3):
            print(f"\n--- Run {i+1} ---")
            start_time = time.time()
            # Pass max_length=GENERATION_MAX_NEW_TOKENS (FunASR interprets this as new tokens)
            # Add repetition_penalty and no_repeat_ngram_size to avoid loop
            res = model.generate(
                input=wav_path, 
                batch_size_s=0, 
                use_itn=False, 
                max_length=GENERATION_MAX_NEW_TOKENS,
                llm_kwargs={}
            )
            end_time = time.time()

            # Post-Generation Cache Probe
            # Post-Generation Cache Probe (Recursive)
            try:
                def find_cache_recursive(module):
                    if hasattr(module, "_static_cache") and module._static_cache is not None:
                        return module._static_cache
                    for name, child in module.named_children():
                        res = find_cache_recursive(child)
                        if res is not None: return res
                    return None

                # Start search from nano_model.llm to avoid picking up unrelated stuff
                sc = find_cache_recursive(nano_model.llm)

                if sc is not None:
                    # Check key_cache
                    has_data = False
                    if hasattr(sc, "key_cache") and len(sc.key_cache) > 0 and isinstance(sc.key_cache[0], torch.Tensor):
                         k0 = sc.key_cache[0]
                         has_data = True
                    elif hasattr(sc, "layers") and len(sc.layers) > 0:
                         # Check first layer
                         l0 = sc.layers[0]
                         if isinstance(l0, tuple): k0 = l0[0]
                         elif hasattr(l0, "keys"): k0 = l0.keys
                         else: k0 = None

                         if isinstance(k0, torch.Tensor):
                             has_data = True

                    if has_data:
                            # Move to CPU for printing
                            k0_cpu = k0.cpu()
                            non_zero = (k0_cpu != 0).sum().item()
                            total = k0_cpu.numel()
                            print(f"[MainProbe] Cache Found! Layer 0 Key Cache: Non-Zero={non_zero}/{total} ({non_zero/total*100:.2f}%)")
                            print(f"[MainProbe] Seen Tokens: {sc.seen_tokens}")
                    else:
                        print(f"[MainProbe] Cache found object but NO tensor data (key_cache & layers empty/invalid).")
                else:
                    print(f"[MainProbe] _static_cache NOT FOUND in `nano_model.llm` hierarchy.")
                    # Debug print structure
                    print(f"Debug Structure: nano_model.llm type: {type(nano_model.llm)}")
                    if hasattr(nano_model.llm, "model"): print(f"  .model type: {type(nano_model.llm.model)}")

            except Exception as e:
                print(f"[MainProbe] Error probing cache: {e}")

            if 'ctc_text' in res[0]:
                 print(f"CTC Text: {res[0]['ctc_text']}")
            print(f"Output (text field only): {res[0]['text']}")
            print(f"Run {i+1} Duration: {end_time - start_time:.4f}s")
    else:
        print(f"Warning: {wav_path} not found. Skipping inference run.")

if __name__ == "__main__":
    main()

