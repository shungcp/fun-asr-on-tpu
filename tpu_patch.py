
import torch
import torch.nn as nn
import transformers.models.qwen2.modeling_qwen2 as qwen2_modeling
from typing import Optional
from funasr.models.sense_voice import model as funasr_model

def patched_eager_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    """
    Patched version of eager_attention_forward for TPU compatibility.
    - Uses contiguous tensors.
    - Explicitly handles causal_mask slicing and types.
    """
    # Repeat KV keys/values to match query heads
    key_states = qwen2_modeling.repeat_kv(key, module.num_key_value_groups)
    value_states = qwen2_modeling.repeat_kv(value, module.num_key_value_groups)

    # Compute attention scores
    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    
    if attention_mask is not None:
        causal_mask = attention_mask
        
        # Ensure dtypes match first to handle padding values correctly
        if causal_mask.dtype != attn_weights.dtype:
            causal_mask = causal_mask.to(attn_weights.dtype)

        if causal_mask.dim() == 4:
            target_k = key_states.shape[-2] # Should be 128 (max_cache_len) or similar
            current_k = causal_mask.shape[-1]
            
            if current_k > target_k:
                causal_mask = causal_mask[:, :, :, :target_k]
            elif current_k < target_k:
                # Pad with large negative value to mask out the extra keys (cache padding)
                # Ensure we use a value compatible with the dtype (e.g. min possible value)
                diff = target_k - current_k
                pad_val = torch.finfo(causal_mask.dtype).min
                # Pad last dimension (right side)
                causal_mask = torch.nn.functional.pad(causal_mask, (0, diff), value=pad_val)
        
        # Ensure causal_mask is a contiguous Tensor.
        if hasattr(causal_mask, "contiguous"):
            causal_mask = causal_mask.contiguous() 

        attn_weights = torch.add(attn_weights, causal_mask)

    attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = torch.nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights

class PatchedMultiHeadedAttentionSANM(nn.Module):
    """
    Patched MultiHeadedAttentionSANM for TPU.
    - Replaces Conv1d with a loop if necessary (or verify if standard Conv1d works with updated torchax).
    - Ensures shape consistency and flattened BMM to avoid broadcasting ambiguity on TPU.
    """
    def __init__(self, n_head, in_feat, n_feat, dropout_rate, kernel_size, sanm_shfit=0, lora_list=None, lora_rank=8, lora_alpha=16, lora_dropout=0.1):
        super().__init__()
        assert n_feat % n_head == 0
        self.d_k = n_feat // n_head
        self.h = n_head
        self.linear_out = nn.Linear(n_feat, n_feat)
        self.linear_q_k_v = nn.Linear(in_feat, n_feat * 3)
        self.dropout = nn.Dropout(p=dropout_rate)
        
        # FSMN Block (Conv1d)
        self.fsmn_block = nn.Conv1d(n_feat, n_feat, kernel_size, stride=1, padding=0, groups=n_feat, bias=False)
        
        left_padding = (kernel_size - 1) // 2
        if sanm_shfit > 0:
            left_padding = left_padding + sanm_shfit
        right_padding = kernel_size - 1 - left_padding
        self.pad_fn = nn.ConstantPad1d((left_padding, right_padding), 0.0)

    def forward_fsmn(self, inputs, mask, mask_shfit_chunk=None):
        b, t, d = inputs.size()
        
        if mask is not None:
             mask = torch.reshape(mask, (b, -1, 1))
             if mask_shfit_chunk is not None:
                 mask = mask * mask_shfit_chunk
             inputs = inputs * mask
        
        x = inputs.transpose(1, 2) # [B, D, T]
        
        # Workaround for potential TPU Conv1d batching issues or just safety
        # We can try to use standard Conv1d first, but previous code used a loop.
        # Let's use a loop for safety as this is 'demo_tpu'.
        # Assuming B is small (1 for demo).
        
        outs = []
        # Ensure b is integer for range
        try:
            b_loop = int(b) 
        except:
             b_loop = b
             
        for i in range(b_loop):
            xi = x[i:i+1] # Keep dim 0
            xi = self.pad_fn(xi)
            xi = self.fsmn_block(xi)
            outs.append(xi)
            
        x = torch.cat(outs, dim=0)
        x = x.transpose(1, 2) # [B, T, D]
        
        x = x + inputs
        x = self.dropout(x)
        if mask is not None:
            x = x * mask
        return x

    def forward_qkv(self, x):
        b, t, d = x.size()
        q_k_v = self.linear_q_k_v(x)
        q, k, v = torch.split(q_k_v, int(self.h * self.d_k), dim=-1)
        q_h = torch.reshape(q, (b, t, self.h, self.d_k)).transpose(1, 2)
        k_h = torch.reshape(k, (b, t, self.h, self.d_k)).transpose(1, 2)
        v_h = torch.reshape(v, (b, t, self.h, self.d_k)).transpose(1, 2)
        return q_h, k_h, v_h, v

    def forward_attention(self, value, scores, mask, mask_att_chunk_encoder=None):
        n_batch = value.size(0)
        
        if mask is not None:
             mask = mask.unsqueeze(1).eq(0)
             if mask_att_chunk_encoder is not None:
                 mask = mask * mask_att_chunk_encoder
             min_value = -1e9 # Fixed large negative
             scores = scores.masked_fill(mask, min_value)
             attn = torch.softmax(scores, dim=-1).masked_fill(mask, 0.0)
        else:
             attn = torch.softmax(scores, dim=-1)
        
        p_attn = self.dropout(attn)
        
        # Flatten to 3D for BMM to avoid potential broadcasting issues
        # p_attn: [B, H, T, T] -> [B*H, T, T]
        # value:  [B, H, T, D] -> [B*H, T, D]
        
        t_q = p_attn.size(-2)
        d_k = value.size(-1)
        
        p_attn_3d = p_attn.flatten(0, -3) 
        value_3d = value.flatten(0, -3)   
        
        x_3d = torch.matmul(p_attn_3d, value_3d)
        x = x_3d.view(n_batch, self.h, t_q, d_k)
        
        x = x.transpose(1, 2).contiguous().view(n_batch, -1, self.h * self.d_k)
        return self.linear_out(x)

    def forward(self, x, mask, mask_shfit_chunk=None, mask_att_chunk_encoder=None):
        q_h, k_h, v_h, v = self.forward_qkv(x)
        fsmn_memory = self.forward_fsmn(v, mask, mask_shfit_chunk)
        
        q_h = q_h * self.d_k ** (-0.5)
        scores = torch.matmul(q_h, k_h.transpose(-2, -1))
        att_outs = self.forward_attention(v_h, scores, mask, mask_att_chunk_encoder)
        
        return torch.add(att_outs, fsmn_memory)

def apply_tpu_patches():
    print("[TPU Patch] Applying Qwen2 Attention Patch...")
    # qwen2_modeling.eager_attention_forward = patched_eager_attention_forward
    print("[TPU Patch] Qwen2 Attention Patch TEMPORARILY DISABLED for isolation")

    # print("[TPU Patch] Applying FunASR SANM Patch...")
    # funasr_model.MultiHeadedAttentionSANM = PatchedMultiHeadedAttentionSANM
    print("[TPU Patch] SANM Patch DISABLED (Confirmed)")
    
    print("[TPU Patch] Patches Applied.")
