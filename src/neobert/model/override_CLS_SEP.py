import torch
from torch import nn
class CLSSEPAttentionReplacer(nn.Module):
    """
    Fully vectorized attention replacer for [CLS] and [SEP] with head-specific learnable scalars.
    Column replacements overwrite row replacements at intersections.
    """
    def __init__(self, num_heads: int, init_value: float = 0.05):
        super().__init__()
        self.num_heads = num_heads
        self.register_parameter("theta_cls_out", nn.Parameter(torch.full((num_heads,), init_value)))
        self.register_parameter("theta_cls_in",  nn.Parameter(torch.full((num_heads,), init_value)))
        self.register_parameter("theta_sep_out", nn.Parameter(torch.full((num_heads,), init_value)))
        self.register_parameter("theta_sep_in",  nn.Parameter(torch.full((num_heads,), init_value)))

    def forward(self, attn: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            attn: Tensor [B, H, L, L]
            pad_mask: BoolTensor [B, H, L, L] (True = non-pad token)
        Returns:
            Tensor [B, H, L, L] with replaced entries.
        """
        B, H, L, _ = attn.shape
        device = attn.device
        dtype = attn.dtype


        cls_idx = pad_mask.any(dim=-1).float().argmax(dim=-1)  # [B,H]
        sep_idx = (pad_mask.any(dim=-1).flip(dims=[-1]).float().argmax(dim=-1))
        sep_idx = L - 1 - sep_idx                                # [B,H] # [B]

        # Copy attention to avoid in-place issues
        out = attn.clone()

        batch_idx = torch.arange(B, device=device)[:, None]  # [B,1]
        head_idx  = torch.arange(H, device=device)[None, :]  # [1,H]

        # Expand theta to [B,H,L] for broadcasting
        theta_cls_out_exp = self.theta_cls_out.view(1,H,1).expand(B,H,L)
        theta_sep_out_exp = self.theta_sep_out.view(1,H,1).expand(B,H,L)
        theta_cls_in_exp  = self.theta_cls_in.view(1,H,1).expand(B,H,L)
        theta_sep_in_exp  = self.theta_sep_in.view(1,H,1).expand(B,H,L)

        # Replace rows
        out[batch_idx, head_idx, cls_idx, :] = theta_cls_out_exp
        out[batch_idx, head_idx, sep_idx, :] = theta_sep_out_exp

        # Replace columns
        out[batch_idx, head_idx, :, cls_idx] = theta_cls_in_exp
        out[batch_idx, head_idx, :, sep_idx] = theta_sep_in_exp

        # # --- Replace CLS rows (source -> others) ---
        # theta_cls_out_exp = self.theta_cls_out.view(1,H,1).expand(B,H,L)
        # print('theta_cls_out_exp', theta_cls_out_exp.shape)
        # out[batch_idx[:, None], torch.arange(H, device=device)[None,:], cls_idx[:, None], :] = theta_cls_out_exp


        # # --- Replace SEP rows ---
        # theta_sep_out_exp = self.theta_sep_out.view(1,H,1).expand(B,H,L)
        # out[batch_idx[:, None], torch.arange(H)[None,:], sep_idx[:, None], :] = theta_sep_out_exp

        # # --- Replace CLS columns (target <- others) ---
        # theta_cls_in_exp = self.theta_cls_in.view(1,H,1).expand(B,H,L)
        # out[batch_idx[:, None], torch.arange(H)[None,:], :, cls_idx[:, None]] = theta_cls_in_exp

        # # --- Replace SEP columns ---
        # theta_sep_in_exp = self.theta_sep_in.view(1,H,1).expand(B,H,L)
        # out[batch_idx[:, None], torch.arange(H)[None,:], :, sep_idx[:, None]] = theta_sep_in_exp

        return out