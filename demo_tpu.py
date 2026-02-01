#================================================================================
# 2026-02-01 v0.1 latency: (run2: not yet/run1: 66s w/compile time)
#================================================================================

import os
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
import transformers.masking_utils

# AGGRESSIVE PATCH: Disable is_compileable to force legacy masking
# This prevents 'transformers' from using vmap-based masking which crashes on TPU/Torchax with IndexError.
def return_false(*args, **kwargs): return False
transformers.masking_utils.is_compileable = return_false

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
        return True

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
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.seen_tokens = 0
        self._position_ids = None

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
                            current_max_pos = cache_position.max() + 1
                            if isinstance(self.seen_tokens, int):
                                self.seen_tokens = current_max_pos
                            else:
                                if current_max_pos > self.seen_tokens:
                                    self.seen_tokens = current_max_pos

                        return k_cache, v_cache

        return key_states, value_states

    def get_seq_length(self, layer_idx=0):
        # Fix for ConcretizationTypeError: Return concrete max_cache_len 
        # instead of dynamic tracer derived from position_ids.
        if hasattr(self, "max_cache_len"):
            return self.max_cache_len
        # Fallback to key_cache shape if available
        if hasattr(self, "key_cache") and len(self.key_cache) > layer_idx:
            return self.key_cache[layer_idx].shape[-2]
        return 0

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

    GENERATION_MAX_LENGTH = 32

    # 1. Apply Patches
    tpu_patch.apply_tpu_patches()

    # 2. Load Model (CPU)
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
                raise ValueError("Both input_ids and inputs_embeds are None in patched_forward")

            attention_mask = kwargs.get('attention_mask', None)
            position_ids = kwargs.get('position_ids', None)
            past_key_values = kwargs.get('past_key_values', None)

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

            # Pad attention_mask to static length (GENERATION_MAX_LENGTH) to avoid JIT recompilation
            if attention_mask is not None and isinstance(attention_mask, torch.Tensor):
                 target_len = GENERATION_MAX_LENGTH
                 current_len = attention_mask.shape[1]
                 if current_len < target_len:
                      pad_len = target_len - current_len
                      zeros = torch.zeros((attention_mask.shape[0], pad_len), dtype=attention_mask.dtype, device=attention_mask.device)
                      attention_mask = torch.cat([attention_mask, zeros], dim=1)

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

                     max_cache_len = GENERATION_MAX_LENGTH
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

    if os.path.exists(wav_path):
        # Run multiple times to verify JIT caching
        for i in range(1):
            print(f"\n--- Run {i+1} ---")
            start_time = time.time()
            res = model.generate(input=wav_path, batch_size_s=0, use_itn=False, max_length=GENERATION_MAX_LENGTH)
            end_time = time.time()
            print(f"Output: {res}")
            print(f"Run {i+1} Duration: {end_time - start_time:.4f}s")
    else:
        print(f"Warning: {wav_path} not found. Skipping inference run.")

if __name__ == "__main__":
    main()
