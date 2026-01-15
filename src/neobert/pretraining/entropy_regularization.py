import torch
import torch.nn.functional as F
import numpy as np


def absolute_to_relative(attn: torch.Tensor) -> torch.Tensor:
    """
    Convert tensor with shape (..., L, L) -> (..., L, 2L-1)
    Works for attn shaped [num_layers, batch_size, num_heads, L, L] (or any leading dims).
    Fully vectorized, uses as_strided to create sliding windows without Python loops.
    """
    *prefix, L1, L2 = attn.shape
    assert L1 == L2, "Last two dims must both be sequence length L"
    L = L1
    pad = L - 1
    # Pad last dimension (keys) on both sides
    # F.pad with (pad_left, pad_right) pads the last dimension
    padded = F.pad(attn, (pad, pad)).contiguous()  # shape: (*prefix, L, 3L-2)

    # original strides
    strides = padded.stride()
    # stride for the second-last and last dims
    s_second = strides[-2]
    s_last = strides[-1]

    # New view size and strides to produce sliding window of length (2L-1)
    new_size = (*prefix, L, 2 * L - 1)
    # when incrementing the "query" index we want to move one step along the second-last dim
    # AND also advance the start in the last dim by 1, so stride for that dim = s_second + s_last
    new_strides = (*strides[:-2], s_second + s_last, s_last)

    relative = padded.as_strided(size=new_size, stride=new_strides)
    return relative  # shape: (*prefix, L, 2L-1)


def compute_head_entropy(attn) :
    """
    input:
    attn of shape [batch_size, num_heads, seqlen, seqlen]

    output:
    entropy ratio [num_heads] 
    """
    relative_probs = absolute_to_relative(torch.softmax(attn, dim=-1))  # [batch_size, num_heads, seqlen, 2*seqlen - 1]
    relative_probs = relative_probs.mean(dim=2)  # average over tokens
    relative_probs = relative_probs.permute(1, 2, 0)  # [num_heads, 2*seqlen - 1, batch_size]

    # Compute entropy
    entropy = -(relative_probs * relative_probs.clamp_min(1e-12).log()).sum(dim=1) # [num_heads, batch_size]

    return entropy.mean(dim=-1) # [num_heads]