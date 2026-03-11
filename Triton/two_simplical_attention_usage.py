"""
Two Simplicial Attention — quick-start usage guide.

Standard attention computes:
    O = softmax(Q @ K^T / √d) @ V

Two Simplicial Attention generalises this to *pairs* of keys and values:
    O = softmax((Q ⊙ K1) @ K2^T / √d) @ (V1 ⊙ V2)

In the simplest case you can set K1=K2=K and V1=V2=V (with biases +1) to
recover something close to standard attention, so the interface is a strict
superset.

Both the Triton and PyTorch implementations share the same interface:
    f(Q, K1, K2, V1, V2, w1, w2, k2_bias, v2_bias, sm_scale, prescale)
"""

import torch

# ── Triton (fast, CUDA-only) ────────────────────────────────────────────────
from two_simplicial_attention import TwoSimplicialAttentionFunction

# ── Pure PyTorch (reference, any device) ────────────────────────────────────
from two_simplicial_attention_pytorch import two_simplicial_attention_pytorch


def main():
    # Shape convention: [batch, seq_len, num_heads, head_dim]
    B, S, H, D = 2, 128, 4, 64
    device = "cuda"
    dtype = torch.bfloat16

    torch.manual_seed(0)
    Q  = torch.randn(B, S, H, D, device=device, dtype=dtype)
    K1 = torch.randn(B, S, H, D, device=device, dtype=dtype)
    K2 = torch.randn(B, S, H, D, device=device, dtype=dtype)
    V1 = torch.randn(B, S, H, D, device=device, dtype=dtype)
    V2 = torch.randn(B, S, H, D, device=device, dtype=dtype)

    # ── 1. Minimal call — identical interface ─────────────────────────────────
    #
    # Both implementations take the same arguments in the same order.
    # w1/w2 control the local-attention window size for K1/V1 and K2/V2.
    # Set them to seq_len (or None) for full (global) attention.

    out_triton = TwoSimplicialAttentionFunction.apply(
        Q, K1, K2, V1, V2,
        S, S,  # w1, w2
    )
    out_pytorch = two_simplicial_attention_pytorch(
        Q, K1, K2, V1, V2,
        w1=S, w2=S,
    )
    print(f"1. Triton shape:  {out_triton.shape}")   # [B, S, H, D]
    print(f"   PyTorch shape: {out_pytorch.shape}")   # [B, S, H, D]

    # ── 2. Local (sliding-window) attention ───────────────────────────────────
    #
    # Restrict each key stream to a window — the higher-order analogue of
    # sliding-window attention.

    W = 32
    out_triton_local = TwoSimplicialAttentionFunction.apply(
        Q, K1, K2, V1, V2,
        W, W,  # w1, w2
    )
    out_pytorch_local = two_simplicial_attention_pytorch(
        Q, K1, K2, V1, V2,
        w1=W, w2=W,
    )
    print(f"2. Local (w={W}) — Triton shape:  {out_triton_local.shape}")
    print(f"   Local (w={W}) — PyTorch shape: {out_pytorch_local.shape}")

    # ── 3. Biases and prescaling ──────────────────────────────────────────────
    #
    # k2_bias/v2_bias add constants to K2/V2 before the computation.
    # prescale=True applies d^(-1/6) to Q, K1, K2 (useful for stability).

    out_triton_bias = TwoSimplicialAttentionFunction.apply(
        Q, K1, K2, V1, V2,
        S, S,          # w1, w2
        0.5, -0.3,     # k2_bias, v2_bias
        None,          # sm_scale (auto)
        True,          # prescale
    )
    out_pytorch_bias = two_simplicial_attention_pytorch(
        Q, K1, K2, V1, V2,
        w1=S, w2=S,
        k2_bias=0.5, v2_bias=-0.3,
        prescale=True,
    )
    print(f"3. With biases + prescale — Triton shape:  {out_triton_bias.shape}")
    print(f"   With biases + prescale — PyTorch shape: {out_pytorch_bias.shape}")

    # ── 4. Backward pass (gradients flow through both) ────────────────────────

    Q_t  = Q.clone().requires_grad_(True)
    K1_t = K1.clone().requires_grad_(True)
    out = TwoSimplicialAttentionFunction.apply(Q_t, K1_t, K2, V1, V2, S, S)
    out.sum().backward()
    print(f"4. Triton  — dQ norm: {Q_t.grad.norm().item():.4f}, "
          f"dK1 norm: {K1_t.grad.norm().item():.4f}")

    Q_p  = Q.clone().requires_grad_(True)
    K1_p = K1.clone().requires_grad_(True)
    out = two_simplicial_attention_pytorch(Q_p, K1_p, K2, V1, V2, w1=S, w2=S)
    out.sum().backward()
    print(f"   PyTorch — dQ norm: {Q_p.grad.norm().item():.4f}, "
          f"dK1 norm: {K1_p.grad.norm().item():.4f}")


if __name__ == "__main__":
    main()
