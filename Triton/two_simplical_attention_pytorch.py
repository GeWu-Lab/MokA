from typing import Optional

import torch


def two_simplicial_attention_pytorch(
    q: torch.Tensor,
    k1: torch.Tensor,
    k2: torch.Tensor,
    v1: torch.Tensor,
    v2: torch.Tensor,
    w1: Optional[int] = None,
    w2: Optional[int] = None,
    k2_bias: float = 0.0,
    v2_bias: float = 0.0,
    sm_scale: Optional[float] = None,
    prescale: bool = False,
) -> torch.Tensor:
    """Pure PyTorch implementation of Two Simplicial Attention.

    Computes:
        O = softmax((Q ⊙ K1) @ K2^T / √d) @ (V1 ⊙ V2)

    where the softmax is over the flattened (j, k) pairs per query i.

    :param q: Query tensor [batch, seq_len, num_heads, head_dim]
    :param k1: First key tensor [batch, seq_len, num_heads, head_dim]
    :param k2: Second key tensor [batch, seq_len, num_heads, head_dim]
    :param v1: First value tensor [batch, seq_len, num_heads, head_dim]
    :param v2: Second value tensor [batch, seq_len, num_heads, head_dim]
    :param w1: Local attention window for K1/V1 (None = full sequence)
    :param w2: Local attention window for K2/V2 (None = full sequence)
    :param k2_bias: Additive bias for K2
    :param v2_bias: Additive bias for V2
    :param sm_scale: Attention scale (default: head_dim^-0.5)
    :param prescale: If True, prescale Q, K1, K2 by d^(-1/6) and use sm_scale=1
    :return: Output tensor [batch, seq_len, num_heads, head_dim]
    """
    batch_size, seq_len, num_heads, head_dim = q.shape

    w1 = w1 if w1 is not None else seq_len
    w2 = w2 if w2 is not None else seq_len

    if prescale:
        scale_factor = head_dim ** (-1.0 / 6.0)
        q = q * scale_factor
        k1 = k1 * scale_factor
        k2 = k2 * scale_factor
        sm_scale = 1.0
    else:
        if sm_scale is None:
            sm_scale = head_dim ** -0.5

    # Apply biases
    k2 = k2 + k2_bias
    v2 = v2 + v2_bias

    # Transpose to [B, H, S, D]
    q = q.permute(0, 2, 1, 3)
    k1 = k1.permute(0, 2, 1, 3)
    k2 = k2.permute(0, 2, 1, 3)
    v1 = v1.permute(0, 2, 1, 3)
    v2 = v2.permute(0, 2, 1, 3)

    # Logits: score[i,j,k] = (q[i] ⊙ k1[j]) · k2[k] * scale
    qk1 = q[:, :, :, None, :] * k1[:, :, None, :, :]  # [B, H, S_i, S_j, D]
    logits = torch.einsum("bhijd,bhkd->bhijk", qk1, k2) * sm_scale  # [B, H, S_i, S_j, S_k]

    # Window mask: |i - j| < w1 and |i - k| < w2
    pos = torch.arange(seq_len, device=q.device)
    mask_j = (pos.view(-1, 1, 1) - pos.view(1, -1, 1)).abs() < w1  # [S_i, S_j, 1]
    mask_k = (pos.view(-1, 1, 1) - pos.view(1, 1, -1)).abs() < w2  # [S_i, 1, S_k]
    logits = logits.masked_fill(~(mask_j & mask_k)[None, None], -1e38)

    # Softmax over flattened (j, k) pairs
    attn = torch.softmax(logits.reshape(batch_size, num_heads, seq_len, -1), dim=-1)
    attn = attn.reshape_as(logits)

    # Values: v12[j,k] = v1[j] ⊙ v2[k]
    v12 = v1[:, :, :, None, :] * v2[:, :, None, :, :]  # [B, H, S_j, S_k, D]

    # Output: o[i] = sum_{j,k} attn[i,j,k] * v12[j,k]
    output = torch.einsum("bhijk,bhjkd->bhid", attn, v12)

    return output.to(q.dtype).permute(0, 2, 1, 3)
