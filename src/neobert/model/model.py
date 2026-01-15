# From https://stackoverflow.com/a/23689767
# From https://github.com/pytorch/pytorch/issues/97899
# From https://github.com/facebookresearch/llama/blob/main/llama/model.py

import math
import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader

from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
# from torch.nn.functional import scaled_dot_product_attention

from typing import Any, Dict, List, Optional
from functools import partial

from xformers.ops import SwiGLU, memory_efficient_attention

from datasets import Dataset

from transformers import PreTrainedModel, PretrainedConfig, PreTrainedTokenizerFast, DataCollatorWithPadding
from transformers.modeling_outputs import SequenceClassifierOutput

from tqdm import tqdm

from .rmsnorm import RMSNorm
from .rotary import precompute_freqs_cis, apply_rotary_emb
from .softpick import softpick
from .override_CLS_SEP import CLSSEPAttentionReplacer


# Efficient implementation equivalent to the following:
def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0,
        is_causal=False, scale=None, enable_gqa=False, attn_activation_fct=torch.softmax) -> torch.Tensor:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias = attn_mask + attn_bias

    if enable_gqa:
        key = key.repeat_interleave(query.size(-3)//key.size(-3), -3)
        value = value.repeat_interleave(query.size(-3)//value.size(-3), -3)

    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = attn_activation_fct(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)

    return attn_weight @ value


def posbert_scaled_dot_product_attention(query, key, attn_mask=None, dropout_p=0.0,
        is_causal=False, scale=None, enable_gqa=False, attention_activation = "softmax") -> torch.Tensor:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias = attn_mask + attn_bias

    if enable_gqa:
        key = key.repeat_interleave(query.size(-3)//key.size(-3), -3)

    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
        
    return attn_weight 



class NeoBERTConfig(PretrainedConfig):
    model_type = "neobert"

    # All config parameters must have a default value.
    def __init__(
        self,
        hidden_size: int = 768,
        pos_size: int = 384,
        num_hidden_layers: int = 28,
        num_attention_heads: int = 12,
        pos_intermediate_size: int =1536,
        intermediate_size: int =3072,
        pos_dropout_prob: float =0.1,
        dropout_prob: float =0.1,
        attention_probs_dropout_prob: float =0.1,
        use_only_sem_for_decoding: bool = False,
        mixed_feed_forward: bool = True,
        embedding_init_range: float = 0.02,
        decoder_init_range: float = 0.02,
        rms_norm: bool = True,
        rope: bool = True,
        posneobert: bool = False,
        norm_eps: float = 1e-06,
        hidden_act: str = "SwiGLU",
        vocab_size: int = 32064,
        pad_token_id: int = 0,
        max_length: int = 1024,
        flash_attention: bool = True,
        base_scale: float = 1.0 / (960.0**0.5),
        ngpt: bool = False,
        positional_embed_init: str = "random",
        attention_activation: str = "softmax",
        mix_attentions: str = "sum",
        untie_cls: bool = False,
        random_offset = False,
        shared_pos_keys = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if rope and posneobert :
            raise ValueError("cant be rope and posneobert at the same time")
        if ngpt and posneobert :
            raise NotImplementedError
        if hidden_size % num_attention_heads != 0:
            raise ValueError("Hidden size must be divisible by the number of heads.")
        if pos_size % num_attention_heads != 0 :
            raise ValueError("Pos size must be divisible by the number of heads.")
        if rope and use_only_sem_for_decoding :
            raise ValueError("Cannot use RoPE and use only semantic for decoding.")
        if rope and positional_embed_init == "2dim_cosine" :
            raise ValueError("Cannot use RoPE and setup positional embeds.")
        if rope and shared_pos_keys :
            raise ValueError("Cannot use RoPE and shared positional embeddings.")
        if rope and mix_attentions == "hadamard" :
            raise ValueError("Cannot setup mix attentions with RoPE.")
        
        if positional_embed_init not in ["random", "2dim_cosine"] :
            raise ValueError
        if attention_activation not in ["softmax", "softpick"] :
            raise ValueError
        if mix_attentions not in ["sum", "hadamard"] :
            raise ValueError

        
        self.hidden_size = hidden_size
        self.pos_size = pos_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        
        self.dim_head = ((hidden_size + pos_size) // num_attention_heads) if posneobert else hidden_size // num_attention_heads
        self.pos_intermediate_size = pos_intermediate_size
        self.intermediate_size = intermediate_size
        self.pos_dropout_prob = pos_dropout_prob
        self.dropout_prob = dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.use_only_sem_for_decoding = use_only_sem_for_decoding
        self.mixed_feed_forward = mixed_feed_forward
        self.embedding_init_range = embedding_init_range
        self.decoder_init_range = decoder_init_range
        self.rms_norm = rms_norm
        self.rope = rope
        self.posneobert = posneobert
        self.norm_eps = norm_eps
        self.hidden_act = hidden_act
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        self.max_length = max_length
        self.flash_attention = flash_attention
        self.base_scale = base_scale
        self.ngpt = ngpt
        self.positional_embed_init = positional_embed_init
        self.attention_activation = attention_activation
        self.untie_cls = untie_cls
        self.mix_attentions = mix_attentions
        self.random_offset = random_offset
        self.shared_pos_keys = shared_pos_keys
        self.kwargs = kwargs


class EncoderBlock(nn.Module):
    """Transformer encoder block."""

    def __init__(self, config: NeoBERTConfig):
        super().__init__()

        self.config = config

        # Attention
        if not self.config.posneobert :
            self.qkv = nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size * 3, bias=False)
            self.wo = nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size, bias=False)
            self.resid_dropout = nn.Dropout(config.dropout_prob)
        else :
            if self.config.shared_pos_keys :
                self.q_pos = nn.Linear(in_features=config.pos_size, out_features=(config.hidden_size + config.pos_size), bias=False)
            else : 
                self.qk_pos = nn.Linear(in_features=config.pos_size, out_features=(config.hidden_size + config.pos_size) * 2, bias=False)
            self.qk_sem = nn.Linear(in_features=config.hidden_size, out_features=(config.hidden_size + config.pos_size) * 2, bias=False)
            self.v_pos = nn.Linear(in_features=config.pos_size, out_features=config.pos_size, bias=False)
            self.v_sem = nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size, bias=False)
            self.wo_pos = nn.Linear(in_features=config.pos_size, out_features=config.pos_size, bias=False)
            self.wo_sem = nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size, bias=False)
            self.resid_dropout = nn.Dropout(config.dropout_prob)
            
            # self.mix_fct = torch.mul if self.config.mix_attentions == "hadamard" else torch.add

            self.sem_attention_head_size = int(config.hidden_size / config.num_attention_heads)
            self.pos_attention_head_size = int(config.pos_size / config.num_attention_heads)

            if self.config.untie_cls :
                self.cls_sep_override = CLSSEPAttentionReplacer(self.config.num_attention_heads)
            # self.theta_sep_out = nn.Parameter(torch.tensor(0.5))
            # self.theta_sep_in  = nn.Parameter(torch.tensor(0.5))

        self.attn_activation_fct = torch.softmax if self.config.attention_activation == "softmax" else softpick


        # Feedforward network
        match config.hidden_act.lower():
            case "swiglu":
                # To keep the number of parameters and the amount of computation constant, we reduce the number of
                # hidden units by a factor of 2/3 (https://arxiv.org/pdf/2002.05202.pdf) and make it a multiple of 8 to
                # avoid RuntimeError due to misaligned operand
                multiple_of = 8
                if not self.config.posneobert :
                    intermediate_size = int(2 * (config.intermediate_size) / 3)
                    intermediate_size = multiple_of * ((intermediate_size + multiple_of - 1) // multiple_of)
                    self.ffn = SwiGLU(config.hidden_size, intermediate_size, config.hidden_size, bias=False)
                else :
                    # FOR POSBERT
                    if self.config.mixed_feed_forward :
                        intermediate_size = int(2 * (config.pos_intermediate_size + config.intermediate_size) / 3)
                        intermediate_size = multiple_of * ((intermediate_size + multiple_of - 1) // multiple_of)
                        self.ffn = SwiGLU(config.hidden_size + config.pos_size, intermediate_size, config.hidden_size + config.pos_size, bias=False)
                    else :
                        pos_intermediate_size = int(2 * (config.pos_intermediate_size) / 3)
                        pos_intermediate_size = multiple_of * ((pos_intermediate_size + multiple_of - 1) // multiple_of)
                        self.pos_ffn = SwiGLU(config.pos_size, pos_intermediate_size, config.pos_size, bias=False)

                        sem_intermediate_size = int(2 * (config.intermediate_size) / 3)
                        sem_intermediate_size = multiple_of * ((sem_intermediate_size + multiple_of - 1) // multiple_of)
                        self.sem_ffn = SwiGLU(config.hidden_size, sem_intermediate_size, config.hidden_size, bias=False)

            case "gelu":
                if not self.config.posneobert :
                    self.ffn = nn.Sequential(
                        nn.Linear(config.hidden_size, config.intermediate_size, bias=False),
                        nn.GELU(),
                        nn.Linear(config.intermediate_size, config.hidden_size, bias=False),
                    )
                else :
                    # FOR POSBERT
                    if self.config.mixed_feed_forward :
                        self.ffn = nn.Sequential(
                            nn.Linear(config.hidden_size + config.pos_size, config.intermediate_size + config.pos_intermediate_size, bias=False),
                            nn.GELU(),
                            nn.Linear(config.intermediate_size + config.pos_intermediate_size, config.hidden_size + config.pos_size, bias=False),
                        )
                    
                    else :
                        self.pos_ffn = nn.Sequential(
                            nn.Linear(config.pos_size,config.pos_intermediate_size, bias=False),
                            nn.GELU(),
                            nn.Linear(config.pos_intermediate_size, config.pos_size, bias=False),
                        )

                        self.sem_ffn = nn.Sequential(
                            nn.Linear(config.hidden_size,config.intermediate_size, bias=False),
                            nn.GELU(),
                            nn.Linear(config.intermediate_size, config.hidden_size, bias=False),
                        )


        # Pre-Layer Norm
        if not self.config.posneobert :
            self.attention_norm = (
                RMSNorm(config.hidden_size, config.norm_eps) if config.rms_norm else nn.LayerNorm(config.hidden_size, config.norm_eps)
            )
            self.ffn_norm = (
                RMSNorm(config.hidden_size, config.norm_eps) if config.rms_norm else nn.LayerNorm(config.hidden_size, config.norm_eps)
            )
        else :
            # separate LayerNorm
            self.sem_attention_norm = (
                RMSNorm(config.hidden_size, config.norm_eps) if config.rms_norm else nn.LayerNorm(config.hidden_size, config.norm_eps)
            )
            self.sem_ffn_norm = (
                RMSNorm(config.hidden_size, config.norm_eps) if config.rms_norm else nn.LayerNorm(config.hidden_size, config.norm_eps)
            )
            self.pos_attention_norm = (
                RMSNorm(config.pos_size, config.norm_eps) if config.rms_norm else nn.LayerNorm(config.pos_size, config.norm_eps)
            )
            self.pos_ffn_norm = (
                RMSNorm(config.pos_size, config.norm_eps) if config.rms_norm else nn.LayerNorm(config.pos_size, config.norm_eps)
            )

        
        # FFN dropout
        self.ffn_dropout = nn.Dropout(config.dropout_prob)


    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor, freqs_cis: torch.Tensor, shared_pos_keys: torch.Tensor | None = None):
        attn_weight = None
        pos_sem_weights = None
        if self.config.posneobert :
            #separated normalization
            # print("here is x", x)
            # print("cut as", x[..., :self.config.pos_size])

            x_pos = self.pos_attention_norm(x[..., :self.config.pos_size])
            x_sem = self.sem_attention_norm(x[..., self.config.pos_size:])
            x = torch.cat([x_pos, x_sem], dim=-1)
            
            new_x, attn_weight, pos_sem_weights = self._posneobert_att_block(x=x, pad_mask=pad_mask, freqs_cis=freqs_cis, shared_pos_keys = shared_pos_keys)
            x = x + new_x

            x_pos = self.pos_ffn_norm(x[..., :self.config.pos_size])
            x_sem = self.sem_ffn_norm(x[..., self.config.pos_size:])
            x = torch.cat([x_pos, x_sem], dim=-1).contiguous()

            # print("input of ffblock", x.shape)
            x = x + self._posneobert_ff_block(x)
            # print("x in forward", x)

        else :
            x = x + self._att_block(self.attention_norm(x), pad_mask, freqs_cis)
            x = x + self._ff_block(self.ffn_norm(x))

        return x, attn_weight, pos_sem_weights

    def _att_block(self, x: torch.Tensor, pad_mask: torch.Tensor, freqs_cis: torch.Tensor):
        if self.config.attention_activation == "softpick" :
            raise NotImplementedError
        
        batch_size, seq_len, _ = x.shape

        xq, xk, xv = self.qkv(x).view(batch_size, seq_len, self.config.num_attention_heads, self.config.dim_head * 3).chunk(3, axis=-1)

        if self.config.rope:
            xq, xk = apply_rotary_emb(xq, xk, freqs_cis)
        

        # print("xqxk", xq, xk)
        if self.config.flash_attention:
            attn = memory_efficient_attention(query=xq, key=xk, value=xv, attn_bias=pad_mask, p=0)
        else:
            # Input and output are of dimension (B, H, M, K) (b_size, num_head, seqlength, h_dim)

            attn = scaled_dot_product_attention(
                query=xq.transpose(1, 2),
                key=xk.transpose(1, 2),
                value=xv.transpose(1, 2),
                attn_mask=pad_mask,
                dropout_p=self.config.dropout_prob if self.training else 0,
                attn_activation_fct = self.attn_activation_fct
            ).transpose(1, 2)

        # print("attention", attn)
        return self.resid_dropout(self.wo(attn.reshape(batch_size, seq_len, self.config.num_attention_heads * self.config.dim_head)))

    def _posneobert_att_block(self, x: torch.Tensor, pad_mask: torch.Tensor, freqs_cis: torch.Tensor, shared_pos_keys: torch.Tensor | None = None):
        batch_size, seq_len, _ = x.shape

        # print("x shape", x.shape)
        if self.config.shared_pos_keys :
            xq_pos = self.q_pos(x[..., :self.config.pos_size]).view(batch_size, seq_len, self.config.num_attention_heads, ((self.config.pos_size + self.config.hidden_size) // self.config.num_attention_heads))
            xk_pos = shared_pos_keys.view(batch_size, seq_len, self.config.num_attention_heads, ((self.config.pos_size + self.config.hidden_size) // self.config.num_attention_heads))
        else :
            xq_pos, xk_pos = self.qk_pos(x[..., :self.config.pos_size]).view(batch_size, seq_len, self.config.num_attention_heads, ((self.config.pos_size + self.config.hidden_size) // self.config.num_attention_heads) * 2).chunk(2, axis=-1)
        xq_sem, xk_sem = self.qk_sem(x[..., self.config.pos_size:]).view(batch_size, seq_len, self.config.num_attention_heads, ((self.config.pos_size + self.config.hidden_size) // self.config.num_attention_heads) * 2).chunk(2, axis=-1)
        xv_pos = self.v_pos(x[..., :self.config.pos_size])
        xv_sem = self.v_sem(x[..., self.config.pos_size:])

        # print("xqp, xkp, xqs, xks, xvp, xvs", xq_pos.shape, xk_pos.shape, xq_sem.shape, xk_sem.shape, xv_pos.shape, xv_sem.shape)

        if self.config.flash_attention:
            raise NotImplementedError
            #doesnt work as is
            # pos_attn = memory_efficient_attention(query=xq, key=xk, value=xv_pos, attn_bias=pad_mask, p=0) # (b_size, num_head, seqlength, pos_head_dim)
            # sem_attn = memory_efficient_attention(query=xq, key=xk, value=xv_sem, attn_bias=pad_mask, p=0) # (b_size, num_head, seqlength, sem_head_dim)
        else:
            # #TODO => make sure, but it seems that the dropout is the same for pos and sem
            
            # Input are of dimension (B, H, M, K) (b_size, num_head, seqlength, h_dim)
            # output are of dimension (B, H, M, M) (b_size, num_head, seqlength, seqlength)
            pos_attn_weight = posbert_scaled_dot_product_attention(
                query=xq_pos.transpose(1, 2),
                key=xk_pos.transpose(1, 2),
                attn_mask=pad_mask,
                dropout_p=self.config.pos_dropout_prob if self.training else 0,
                attention_activation = self.config.attention_activation
            )


            sem_attn_weight = posbert_scaled_dot_product_attention(
                query=xq_sem.transpose(1, 2),
                key=xk_sem.transpose(1, 2),
                attn_mask=pad_mask,
                # dropout_p=self.config.pos_dropout_prob if self.training else 0,
                # attention_activation = self.config.attention_activation
            )

        if self.config.untie_cls :
            self.cls_sep_override(pos_attn_weight, pad_mask)

        if self.config.mix_attentions == "sum" :
            attn_weight = torch.add(pos_attn_weight,sem_attn_weight)
            attn_weight = self.attn_activation_fct(attn_weight,  dim=-1).to(xq_sem.dtype)
        elif self.config.mix_attentions == "hadamard" :
            pos_p = torch.softmax(pos_attn_weight, dim=-1)
            sem_p = torch.softmax(sem_attn_weight, dim=-1)
            attn_weight = self.attn_activation_fct(pos_p*sem_p, dim=-1).to(xq_sem.dtype)

            # print("after softpick", attn_weight)
            # print("attention weight", attn_weight.shape)
        attn_weight = torch.dropout(attn_weight, self.config.pos_dropout_prob if self.training else 0, train=True)

        xv_pos = xv_pos.reshape(batch_size, seq_len, self.config.num_attention_heads, self.pos_attention_head_size)
        xv_sem = xv_sem.reshape(batch_size, seq_len, self.config.num_attention_heads, self.sem_attention_head_size)

        pos_attn = (attn_weight @ xv_pos.transpose(1,2)).transpose(1, 2) # [b_size, seq_length, num_head, pos_size]
        sem_attn = (attn_weight @ xv_sem.transpose(1,2)).transpose(1, 2) # [b_size, seq_length, num_head, sem_size]

        pos_attn = self.wo_pos(pos_attn.reshape(batch_size, seq_len, self.config.num_attention_heads * self.pos_attention_head_size))
        sem_attn = self.wo_sem(sem_attn.reshape(batch_size, seq_len, self.config.num_attention_heads * self.sem_attention_head_size))
        attn = torch.cat([pos_attn, sem_attn], dim=-1).to(x.dtype).contiguous()

        return self.resid_dropout(attn), attn_weight, [pos_attn_weight, sem_attn_weight]

    def _ff_block(self, x: torch.Tensor):
        return self.ffn_dropout(self.ffn(x))


    def _posneobert_ff_block(self, x:torch.Tensor):
        if self.config.mixed_feed_forward :
            x = self.ffn(x.clone().contiguous())
        else :
            x_pos = self.pos_ffn(x[..., :self.config.pos_size])
            x_sem = self.sem_ffn(x[..., self.config.pos_size:])
            x = torch.cat([x_pos, x_sem], dim=-1)
        return self.ffn_dropout(x)
    



class NormEncoderBlock(nn.Module):
    """Transformer encoder block."""

    def __init__(self, config: NeoBERTConfig):
        super().__init__()

        self.config = config

        self.attention_head_size = int((config.hidden_size + config.pos_size) / config.num_attention_heads)
        self.sem_attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.pos_attention_head_size = int(config.pos_size / config.num_attention_heads)

        self.all_head_size = config.num_attention_heads * self.attention_head_size
        self.sem_all_head_size = config.num_attention_heads * self.sem_attention_head_size
        self.pos_all_head_size = config.num_attention_heads * self.pos_attention_head_size

        # Attention
        if not self.config.posneobert :
            self.qkv = nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size * 3, bias=False)
            self.wo = nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size, bias=False)
            self.resid_dropout = nn.Dropout(config.dropout)
        else :
            self.qk = nn.Linear(in_features=config.hidden_size + config.pos_size, out_features=(config.hidden_size + config.pos_size) * 2, bias=False)
            self.v_pos = nn.Linear(in_features=config.pos_size, out_features=config.pos_size, bias=False)
            self.v_pos = nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size, bias=False)
            self.pos_resid_dropout = nn.Dropout(config.pos_dropout_prob)
            self.sem_resid_dropout = nn.Dropout(config.dropout_prob)

        
        self.c_fc = nn.Linear(config.hidden_size, 2 * config.intermediate_size, bias=False)
        self.silu = nn.SiLU()
        self.mlp_c_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

        self.ffn_dropout = nn.Dropout(config.dropout)

        self.attn_alpha_init_value = 0.05
        self.attn_alpha_init_scaling = config.base_scale
        self.attn_alpha = torch.nn.Parameter(self.attn_alpha_init_scaling * torch.ones(config.hidden_size))

        self.mlp_alpha_init_value = 0.05
        self.mlp_alpha_init_scaling = config.base_scale
        self.mlp_alpha = torch.nn.Parameter(self.mlp_alpha_init_scaling * torch.ones(config.hidden_size))

        self.sqk_init_value = 1.0
        self.sqk_init_scaling = config.base_scale
        self.sqk = torch.nn.Parameter(self.sqk_init_scaling * torch.ones(config.hidden_size))

        self.suv_init_value = 1.0
        self.suv_init_scaling = 1.0
        self.suv = torch.nn.Parameter(self.suv_init_scaling * torch.ones(2 * config.intermediate_size))

    def justnorm(self, x):
        res = x / x.norm(p=2, dim=-1, keepdim=True)
        return res

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor, freqs_cis: torch.Tensor):
        x_attn = self._att_block(x, pad_mask, freqs_cis)

        lr = self.attn_alpha * (self.attn_alpha_init_value / self.attn_alpha_init_scaling)
        lr = torch.abs(lr)

        A_norm = self.justnorm(x)
        B_norm = self.justnorm(x_attn)
        x = self.justnorm(A_norm + lr * (B_norm - A_norm))

        x_ff = self._ff_block(x)

        lr = self.mlp_alpha * (self.mlp_alpha_init_value / self.mlp_alpha_init_scaling)
        lr = torch.abs(lr)

        A_norm = self.justnorm(x)
        B_norm = self.justnorm(x_ff)
        x = self.justnorm(A_norm + lr * (B_norm - A_norm))

        return x

    def _att_block(self, x: torch.Tensor, pad_mask: torch.Tensor, freqs_cis: torch.Tensor):
        batch_size, seq_len, _ = x.shape

        xq, xk, xv = self.qkv(x).view(batch_size, seq_len, self.config.num_attention_heads, self.config.dim_head * 3).chunk(3, axis=-1)

        if self.config.rope:
            xq, xk = apply_rotary_emb(xq, xk, freqs_cis)

        sqk = (self.sqk * (self.sqk_init_value / self.sqk_init_scaling)).view(
            1, 1, self.config.num_attention_heads, self.config.hidden_size // self.config.num_attention_heads
        )
        xq = sqk * self.justnorm(xq)
        xk = sqk * self.justnorm(xk)

        softmax_scale = (self.config.hidden_size / self.config.num_attention_heads) ** 0.5

        if self.config.flash_attention:
            attn = memory_efficient_attention(query=xq, key=xk, value=xv, attn_bias=pad_mask, p=0, scale=softmax_scale)
        else:
            # Input and output are of dimension (B, H, M, K)
            attn = scaled_dot_product_attention(
                query=xq.transpose(1, 2),
                key=xk.transpose(1, 2),
                value=xv.transpose(1, 2),
                attn_mask=pad_mask,
                dropout_p=self.config.dropout_prob if self.training else 0,
                scale=softmax_scale,
            ).transpose(1, 2)

        return self.resid_dropout(self.wo(attn.reshape(batch_size, seq_len, self.config.hidden_size)))

    def _ff_block(self, x: torch.Tensor):
        uv = self.c_fc(x)
        suv = self.suv * ((self.suv_init_value / self.suv_init_scaling) * (self.config.hidden_size**0.5))
        uv = suv * uv

        u, v = torch.chunk(uv, 2, dim=-1)
        x = u * self.silu(v)
        x = self.mlp_c_proj(x)

        return self.ffn_dropout(x)


class NeoBERTPreTrainedModel(PreTrainedModel):
    config_class = NeoBERTConfig
    _supports_cache_class = True

    def _init_weights(self, module):
        if getattr(module, "_skip_weight_init", False):
            return  #  Skip this one

        if isinstance(module, nn.Linear):
            module.weight.data.uniform_(-self.config.decoder_init_range, self.config.decoder_init_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.uniform_(-self.config.embedding_init_range, self.config.embedding_init_range)
        elif isinstance(module, CLSSEPAttentionReplacer):
            # Initialize head-specific thetas randomly
            module.theta_cls_out.data.uniform_(-self.config.decoder_init_range, self.config.decoder_init_range)
            module.theta_cls_in.data.uniform_(-self.config.decoder_init_range, self.config.decoder_init_range)
            module.theta_sep_out.data.uniform_(-self.config.decoder_init_range, self.config.decoder_init_range)
            module.theta_sep_in.data.uniform_(-self.config.decoder_init_range, self.config.decoder_init_range)


class NeoBERT(NeoBERTPreTrainedModel):
    config_class = NeoBERTConfig

    def __init__(self, config: NeoBERTConfig):
        super().__init__(config)

        self.config = config

        self.encoder = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.shared_pos_encoder = nn.Linear(in_features=config.pos_size, out_features=(config.hidden_size + config.pos_size)) if config.shared_pos_keys else None

        if self.config.rope:
            self.freqs_cis = precompute_freqs_cis(config.hidden_size // config.num_attention_heads, config.max_length)
        elif self.config.posneobert:
            match config.positional_embed_init :
                case "random" :
                    self.positional_embedding = nn.Embedding(config.max_length + 1, config.pos_size, padding_idx=config.pad_token_id)
                case "2dim_cosine" :
                    embs = torch.zeros((config.max_length + 1, config.pos_size))
                    rows = torch.arange(config.max_length + 1, dtype=torch.float32)
                    angles = math.pi * rows / config.max_length
                    embs[:, :2] = torch.stack([torch.cos(angles)/10, torch.sin(angles)/10], dim=1)
                    self.positional_embedding = nn.Embedding.from_pretrained(embs, freeze=False)
                    self.positional_embedding._skip_weight_init = True
        
        else:
            self.positional_embedding = nn.Embedding(config.max_length + 1, config.hidden_size, padding_idx=config.pad_token_id)

        self.transformer_encoder = nn.ModuleList()
        for _ in range(config.num_hidden_layers):
            self.transformer_encoder.append(EncoderBlock(config))

        if not self.config.posneobert :
            self.layer_norm = (
                RMSNorm(config.hidden_size, config.norm_eps) if config.rms_norm else nn.LayerNorm(config.hidden_size, config.norm_eps)
            )
        else :
            self.sem_layer_norm = (
                RMSNorm(config.hidden_size, config.norm_eps) if config.rms_norm else nn.LayerNorm(config.hidden_size, config.norm_eps)
            )
            self.pos_layer_norm = (
                RMSNorm(config.pos_size, config.norm_eps) if config.rms_norm else nn.LayerNorm(config.pos_size, config.norm_eps)
            )
        # Initialize weights and apply final processing
        self.post_init()


    def forward(self, src, pad_mask=None):
        # Expand and repeat: (Batch, Length) -> (Batch, Heads, Length, Length)
        all_attentions = []
        all_hidden_states = []
        all_pos_sem_attentions = []

        if pad_mask is not None:
            assert pad_mask.dtype != torch.bool and 1.0 not in pad_mask, "NeoBERT expects an additive pad_mask"
            pad_mask = pad_mask.unsqueeze(1).unsqueeze(1).repeat(1, self.config.num_attention_heads, pad_mask.size(-1), 1)

        # RoPE
        freqs_cis = None
        if self.config.rope:
            self.freqs_cis = self.freqs_cis.to(src.device, non_blocking=True)
            freqs_cis = self.freqs_cis[: src.shape[1]]

        # Embedding
        x = self.encoder(src)

        # Positional embedding
        if not self.config.rope:
            if not self.config.posneobert :
                mask = src.ne(self.config.pad_token_id).int()
                incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask)) * mask  #
                incremental_indices = incremental_indices.long() + self.config.pad_token_id
                x += self.positional_embedding(incremental_indices)
            else :
                mask = src.ne(self.config.pad_token_id).int()
                incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask)) * mask  #
                incremental_indices = incremental_indices.long() + self.config.pad_token_id
                if self.training and self.config.random_offset:
                    valid_lengths = mask.sum(dim=1)  # How many non-pad tokens per example
                    max_offsets = (self.config.max_length - valid_lengths).clamp(min=0)
         
                    # Generate random offsets for all examples in a single call
                    random_offsets =  torch.randint(0, max_offsets.max() + 1, (len(max_offsets),)).to(mask.device)
                    random_offsets = random_offsets * (random_offsets <= max_offsets)

                    # Add the random offsets to the positional indices
                    incremental_indices += random_offsets.unsqueeze(1) * mask  # Apply offset only to non-pad tokens

                positional_embed = self.positional_embedding(incremental_indices)
                x = torch.concat([positional_embed, x], dim=-1)

        # Transformer encoder

        shared_pos_keys = self.shared_pos_encoder(positional_embed) if self.config.shared_pos_keys else None
        for layer in self.transformer_encoder:
            # print("getting in x", x)
            
            x, attention, pos_sem_attentions = layer(x, pad_mask, freqs_cis, shared_pos_keys = shared_pos_keys)
            all_hidden_states.append(x)
            all_attentions.append(attention)
            all_pos_sem_attentions.append(pos_sem_attentions)

        # Final normalization layer
        if not self.config.posneobert :
            x = self.layer_norm(x)
        else :
            x_pos = self.pos_layer_norm(x[..., :self.config.pos_size])
            x_sem = self.sem_layer_norm(x[..., self.config.pos_size:])
            x = torch.cat([x_pos, x_sem], dim=-1)
        
        # Return the output of the last hidden layer
        return x, all_attentions, all_hidden_states, all_pos_sem_attentions


class NormNeoBERT(NeoBERTPreTrainedModel):
    config_class = NeoBERTConfig

    def __init__(self, config: NeoBERTConfig):
        super().__init__(config)

        self.config = config

        self.encoder = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)

        if self.config.rope:
            self.freqs_cis = precompute_freqs_cis(config.hidden_size // config.num_attention_heads, config.max_length)
        else:
            self.positional_embedding = nn.Embedding(config.max_length + 1, config.hidden_size, padding_idx=config.pad_token_id)

        self.transformer_encoder = nn.ModuleList()
        for _ in range(config.num_hidden_layers):
            self.transformer_encoder.append(NormEncoderBlock(config))

        self.layer_norm = (
            RMSNorm(config.hidden_size, config.norm_eps) if config.rms_norm else nn.LayerNorm(config.hidden_size, config.norm_eps)
        )

        # Initialize weights and apply final processing
        self.post_init()

        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=config.base_scale / math.sqrt(2 * config.num_hidden_layers))

        self.sz_init_value = 1.00
        self.sz_init_scaling = config.base_scale
        self.sz = torch.nn.Parameter(self.sz_init_scaling * torch.ones(config.vocab_size, dtype=torch.float32))

    def forward(self, src, pad_mask=None):
        # Expand and repeat: (Batch, Length) -> (Batch, Heads, Length, Length)
        if pad_mask is not None:
            assert pad_mask.dtype != torch.bool and 1.0 not in pad_mask, "NeoBERT expects an additive pad_mask"
            pad_mask = pad_mask.unsqueeze(1).unsqueeze(1).repeat(1, self.config.num_attention_heads, pad_mask.size(-1), 1)

        # RoPE
        freqs_cis = None
        if self.config.rope:
            self.freqs_cis = self.freqs_cis.to(src.device, non_blocking=True)
            freqs_cis = self.freqs_cis[: src.shape[1]]

        # Embedding
        x = self.encoder(src)

        # Positional embedding
        if not self.config.rope:
            mask = src.ne(self.config.pad_token_id).int()
            incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask)) * mask  #
            incremental_indices = incremental_indices.long() + self.config.pad_token_id
            x += self.positional_embedding(incremental_indices)

        # Transformer encoder
        for layer in self.transformer_encoder:
            x = layer(x, pad_mask, freqs_cis)

        # Return the output of the last hidden layer
        return x


class NeoBERTLMHead(NeoBERTPreTrainedModel):
    config_class = NeoBERTConfig

    def __init__(self, config: NeoBERTConfig):
        super().__init__(config)

        self.config = config

        self.model = NormNeoBERT(config) if self.config.ngpt else NeoBERT(config)

        if not self.config.posneobert :
            self.decoder = nn.Linear(config.hidden_size, config.vocab_size)

        else :
            if self.config.use_only_sem_for_decoding :
                self.decoder = nn.Linear(config.hidden_size, config.vocab_size)
            else :
                self.decoder = nn.Linear(config.hidden_size + config.pos_size, config.vocab_size)

        self.post_init()

    def forward(self, src, pad_mask=None):

        hidden_representation, all_attentions, all_hidden_states, all_pos_sem_attentions = self.model.forward(src, pad_mask)

        if not self.config.posneobert :
            logits = self.decoder(hidden_representation)
        else :
            if self.config.use_only_sem_for_decoding :
                logits = self.decoder(hidden_representation[..., self.config.pos_size:])
            else :
                logits = self.decoder(hidden_representation)

        return {"hidden_representation": hidden_representation, 
                "logits": logits, 
                "all_attentions": all_attentions,
                "all_hidden_states": all_hidden_states, 
                "all_pos_sem_attentions":all_pos_sem_attentions}


class PosOnlyNeoBERTLMHead(NeoBERTLMHead) :
     
    def load_state_dict(self, state_dict, strict=True, assign=False, layers=[], *model_args, **kwargs):

        out = super().load_state_dict(state_dict, strict=strict, assign=assign)

        # Now modify the weights
        with torch.no_grad():
            for layer in layers:
                self.model.transformer_encoder[layer].qk.weight[:, self.config.pos_size:] = 0

        return out
    

class SemOnlyNeoBERTLMHead(NeoBERTLMHead) :

    def load_state_dict(self, state_dict, strict=True, assign=False, layers=[], *model_args, **kwargs):

        out = super().load_state_dict(state_dict, strict=strict, assign=assign)

        # Now modify the weights
        with torch.no_grad():
            for layer in layers:
                self.model.transformer_encoder[layer].qk.weight[:, :self.config.pos_size] = 0

        return out

class NeoBERTForSequenceClassification(NeoBERTPreTrainedModel):

    def __init__(
        self,
        config: NeoBERTConfig,
        num_labels: int = 2,
        classifier_dropout: float = 0.1,
        classifier_init_range: float = 0.02,
        **kwargs,
    ):
        super().__init__(config)

        self.config = config

        self.num_labels = num_labels
        self.classifier_dropout = classifier_dropout
        self.classifier_init_range = classifier_init_range

        self.model = NeoBERT(config)

        self.dense = nn.Linear(self.config.hidden_size, self.config.hidden_size)
        self.dropout = nn.Dropout(self.classifier_dropout)
        self.classifier = nn.Linear(self.config.hidden_size, self.num_labels)

        self.post_init()

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.classifier_init_range)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, src, pad_mask=None):
        hidden_representation = self.model.forward(src, pad_mask)

        x = hidden_representation[:, 0, :]
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)

        logits = self.classifier(x)

        return {"hidden_representation": hidden_representation, "logits": logits}


class NeoBERTHFForSequenceClassification(NeoBERTPreTrainedModel):
    config_class = NeoBERTConfig

    def __init__(self, config: NeoBERTConfig):
        super().__init__(config)

        self.config = config

        self.num_labels = getattr(config, "num_labels", 2)
        self.classifier_dropout = getattr(config, "classifier_dropout", 0.1)
        self.classifier_init_range = getattr(config, "classifier_init_range", 0.02)

        self.model = NeoBERT(config)

        self.dense = nn.Linear(self.config.hidden_size, self.config.hidden_size)
        self.dropout = nn.Dropout(self.classifier_dropout)
        self.classifier = nn.Linear(self.config.hidden_size, self.num_labels)

        self.post_init()

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.classifier_init_range)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):

        hidden_representation = self.model.forward(input_ids, attention_mask)

        x = hidden_representation[:, 0, :]
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)

        logits = self.classifier(x)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "ression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,)
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=hidden_representation,
            attentions=None,
        )


class NeoBERTForMTEB(NeoBERTPreTrainedModel):
    config_class = NeoBERTConfig

    def __init__(
        self,
        config: NeoBERTConfig,
        tokenizer: PreTrainedTokenizerFast,
        max_length: int = 1024,
        batch_size: int = 8,
        pooling: str = "avg",
        **kwargs,
    ):
        super().__init__(config)

        self.config = config
        self.model = NeoBERT(config)

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.batch_size = batch_size
        self.pooling = pooling

    def encode_queries(self, queries: List[str], **kwargs):
        if "instructions" in kwargs:
            if kwargs["instructions"] is not None:
                queries = [(query + " " + kwargs["instructions"][query]).strip() for query in queries]
            new_kwargs = {k: v for k, v in kwargs.items() if k not in ["instructions", "qid"]}
        else:
            new_kwargs = kwargs

        return self.encode(
            queries,
            **new_kwargs,
        )

    def encode_corpus(self, corpus: List[Dict[str, str]], batch_size: int, **kwargs):
        if isinstance(corpus, dict):
            sentences = [
                (corpus["title"][i] + " " + corpus["text"][i]).strip() if "title" in corpus else corpus["text"][i].strip()
                for i in range(len(corpus["text"]))
            ]
        else:
            if isinstance(corpus[0], dict):
                sentences = [(doc["title"] + " " + doc["text"]).strip() if "title" in doc else doc["text"].strip() for doc in corpus]
            else:
                sentences = corpus

        if "instructions" in kwargs:  # not used on the doc side
            new_kwargs = {k: v for k, v in kwargs.items() if k not in ["instructions", "qid"]}
        else:
            new_kwargs = kwargs

        return self.encode(
            sentences,
            **new_kwargs,
        )

    @torch.no_grad()
    def encode(self, sentences: list[str], **kwargs: Any) -> torch.Tensor:
        """Encodes the given sentences using the encoder.

        Args:
            sentences: The sentences to encode.
            **kwargs: Additional arguments to pass to the encoder.

        Returns:
            The encoded sentences.
        """

        device = "cuda" if torch.cuda.is_available() else "cpu"

        def _transform_func(tokenizer: PreTrainedTokenizerFast, x: Dict[str, List]):
            batch_dict = tokenizer(
                x["input_texts"],
                truncation=True,
                max_length=self.max_length,
                padding=False,
                return_token_type_ids=False,
            )

            return batch_dict

        dataset: Dataset = Dataset.from_dict({"input_texts": sentences})
        dataset.set_transform(partial(_transform_func, self.tokenizer))

        data_collator = data_collator = DataCollatorWithPadding(self.tokenizer, pad_to_multiple_of=8)
        dataloader = DataLoader(
            dataset,
            collate_fn=data_collator,
            batch_size=self.batch_size,
            num_workers=2,
            shuffle=False,
            pin_memory=True,
        )

        encodings = []
        for batch in tqdm(dataloader, desc="encoding", mininterval=10, disable=len(sentences) < 128):
            input_ids = batch["input_ids"].to(device)

            pad_mask = batch["attention_mask"].to(device)
            xformers_mask = torch.where(pad_mask == 1, float(0.0), float("-inf")).type(torch.float16)

            outputs = self.model(input_ids, xformers_mask)

            if self.pooling == "avg":
                outputs = outputs * pad_mask.unsqueeze(-1).expand(-1, -1, outputs.shape[-1])
                outputs = outputs.sum(dim=1) / pad_mask.to(device).sum(dim=1).unsqueeze(-1)
            else:
                outputs = outputs[:, 0, :]

            encodings.append(outputs.cpu().numpy())

        return np.concatenate(encodings, axis=0)


if __name__ == "__main__" :

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    # Test model
    config = NeoBERTConfig(
        hidden_size = 720,
        pos_size=48,
        num_hidden_layers = 12,
        num_attention_heads = 12,
        pos_intermediate_size=336,
        intermediate_size=2880,
        pos_dropout_prob=0.1,
        dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        use_only_sem_for_decoding = False,
        mixed_feed_forward = False,
        embedding_init_range = 0.02,
        decoder_init_range = 0.02,
        rms_norm = False,
        rope = False,
        posneobert = True,
        norm_eps = 1e-06,
        hidden_act = "SwiGLU",
        vocab_size = tokenizer.vocab_size,
        pad_token_id = 0,
        max_length = 1024,
        flash_attention = False,
        base_scale = 1.0 / (960.0**0.5),
        ngpt = False,
        positional_embed_init = "2dim_cosine")
    
    print(config)
    model = NeoBERTLMHead(config)

    text = "This is a text, and this is a [MASK]."
    input = tokenizer(text, return_tensors="pt")
    print(input)
    output = model(input["input_ids"])
    print(output["logits"].shape)


    # TEST DE DROPOUT
    # emb = torch.rand((3,3))
    # other1 = torch.rand((3,3))
    # other2 = torch.rand((3,3))
    # print(emb)
    # dropout = nn.Dropout(0.2)
    # emb = dropout(emb)
    # print(emb)
    # new = emb @ other1
    # print(emb)
    # new2 = emb @ other2