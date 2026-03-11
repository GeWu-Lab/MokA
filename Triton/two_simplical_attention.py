from typing import Optional

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from triton import Config


@triton.autotune(
    configs=[
        Config(
            {
                "BLOCK_SIZE_Q": 64,
                "BLOCK_SIZE_KV": 32,
                "num_stages": 1,
            },
            num_warps=4,
        )
    ],
    key=["HEAD_DIM"],
)
@triton.jit
def two_simplicial_attn_fwd_kernel(
    Q_ptr,
    K1_ptr,
    K2_ptr,
    V1_ptr,
    V2_ptr,
    O_ptr,
    M_ptr,
    SLOPES_ptr,  # ALiBi slopes tensor [num_heads]
    bs,
    seq_len,
    num_heads,
    head_dim,
    w1: tl.constexpr,
    w2: tl.constexpr,
    # Stride parameters for memory layout
    q_stride_b,
    q_stride_s,
    q_stride_k,
    q_stride_h,
    k1_stride_b,
    k1_stride_s,
    k1_stride_k,
    k1_stride_h,
    k2_stride_b,
    k2_stride_s,
    k2_stride_k,
    k2_stride_h,
    v1_stride_b,
    v1_stride_s,
    v1_stride_k,
    v1_stride_h,
    v2_stride_b,
    v2_stride_s,
    v2_stride_k,
    v2_stride_h,
    out_stride_b,
    out_stride_s,
    out_stride_k,
    out_stride_h,
    m_stride_b,
    m_stride_k,
    m_stride_s,
    slopes_stride_h,
    # Compile-time constants
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_KV: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    SM_SCALE: tl.constexpr,
    K2_BIAS: tl.constexpr,
    V2_BIAS: tl.constexpr,
    USE_ALIBI3D: tl.constexpr,
    ALIBI_ALPHA: tl.constexpr,
    CAUSAL: tl.constexpr,
    num_stages: tl.constexpr,
    # dtype parameters
    DATA_DTYPE: tl.constexpr,
    COMPUTE_DTYPE: tl.constexpr,
    GEMM_DTYPE: tl.constexpr,
):
    """Two Simplicial Attention Forward Kernel with ALiBi3D Positional Bias.

    Implements the Two Simplicial Attention mechanism from https://arxiv.org/abs/2507.02754
    with our custom ALiBi3D positional bias extension for improved length extrapolation.

    The core simplicial attention mechanism computes attention over pairs of
    key-value vectors (K1, V1) and (K2, V2). The ALiBi3D positional bias is our
    novel addition to enable better extrapolation to longer sequences.

    Mathematical Formulation
    ------------------------

    The core attention computation follows::

        Attention(Q, K1, K2, V1, V2) = softmax((Q ⊙ K1) @ K2^T / √d + B_ALiBi3D) @ (V1 ⊙ V2)

    ALiBi3D Positional Bias::

        B_ALiBi3D[i,k] = -m * (|i - j|^α + |i - k|^α)

    Extends ALiBi to 3D by penalizing attention based on the sum of distances
    from query position i to both key positions j and k, encouraging locality.
    The α parameter prevents degenerate cases where |i-j| and |i-k| differ greatly:
    α > 1 ensures large distances dominate the penalty.

    Where:
        - i: query position index
        - j: first key (K1) position index
        - k: second key (K2) position index
        - m: head-specific slope parameter (from SLOPES_ptr)
        - α: distance power factor (ALIBI_ALPHA)

    Local Attention Windows::

        j ∈ [max(0, i - w1), min(n, i)]
        k ∈ [max(0, i - w2), min(n, i)]

    Causal Masking: j ≤ i and k ≤ i for all valid positions

    Kernel Design Choices
    ---------------------

    The computation is split strategically for efficiency:

    - Q ⊙ K1: Element-wise multiplication using ALU (avoids tensor cores but enables
    efficient broadcasting since K1 is a single vector per iteration)
    - (Q ⊙ K1) @ K2^T: Matrix multiplication using tensor cores (the computational
    bottleneck with O(n²) complexity, worth the tensor core overhead)

    :param Q_ptr: Query tensor [batch, seq_len, num_heads, head_dim]
    :type Q_ptr: Tensor pointer
    :param K1_ptr: First key tensor with same shape as Q
    :type K1_ptr: Tensor pointer
    :param K2_ptr: Second key tensor with same shape as Q
    :type K2_ptr: Tensor pointer
    :param V1_ptr: First value tensor with same shape as Q
    :type V1_ptr: Tensor pointer
    :param V2_ptr: Second value tensor with same shape as Q
    :type V2_ptr: Tensor pointer
    :param O_ptr: Output tensor with same shape as Q
    :type O_ptr: Tensor pointer
    :param M_ptr: Log-sum-exp values for numerical stability [batch, num_heads, seq_len]
    :type M_ptr: Tensor pointer
    :param SLOPES_ptr: ALiBi slopes per head [num_heads]
    :type SLOPES_ptr: Tensor pointer
    :param bs: Batch size dimension
    :type bs: int
    :param seq_len: Sequence length dimension
    :type seq_len: int
    :param num_heads: Number of attention heads dimension
    :type num_heads: int
    :param head_dim: Head dimension
    :type head_dim: int
    :param w1: Local attention window size for K1/V1
    :type w1: int
    :param w2: Local attention window size for K2/V2
    :type w2: int
    :param Q_stride_b: Query tensor batch stride
    :type Q_stride_b: int
    :param Q_stride_s: Query tensor sequence stride
    :type Q_stride_s: int
    :param Q_stride_k: Query tensor head stride
    :type Q_stride_k: int
    :param Q_stride_h: Query tensor dimension stride
    :type Q_stride_h: int
    :param K1_stride_b: K1 tensor batch stride
    :type K1_stride_b: int
    :param K1_stride_s: K1 tensor sequence stride
    :type K1_stride_s: int
    :param K1_stride_k: K1 tensor head stride
    :type K1_stride_k: int
    :param K1_stride_h: K1 tensor dimension stride
    :type K1_stride_h: int
    :param K2_stride_b: K2 tensor batch stride
    :type K2_stride_b: int
    :param K2_stride_s: K2 tensor sequence stride
    :type K2_stride_s: int
    :param K2_stride_k: K2 tensor head stride
    :type K2_stride_k: int
    :param K2_stride_h: K2 tensor dimension stride
    :type K2_stride_h: int
    :param V1_stride_b: V1 tensor batch stride
    :type V1_stride_b: int
    :param V1_stride_s: V1 tensor sequence stride
    :type V1_stride_s: int
    :param V1_stride_k: V1 tensor head stride
    :type V1_stride_k: int
    :param V1_stride_h: V1 tensor dimension stride
    :type V1_stride_h: int
    :param V2_stride_b: V2 tensor batch stride
    :type V2_stride_b: int
    :param V2_stride_s: V2 tensor sequence stride
    :type V2_stride_s: int
    :param V2_stride_k: V2 tensor head stride
    :type V2_stride_k: int
    :param V2_stride_h: V2 tensor dimension stride
    :type V2_stride_h: int
    :param O_stride_b: Output tensor batch stride
    :type O_stride_b: int
    :param O_stride_s: Output tensor sequence stride
    :type O_stride_s: int
    :param O_stride_k: Output tensor head stride
    :type O_stride_k: int
    :param O_stride_h: Output tensor dimension stride
    :type O_stride_h: int
    :param M_stride_b: M tensor batch stride
    :type M_stride_b: int
    :param M_stride_k: M tensor head stride
    :type M_stride_k: int
    :param M_stride_s: M tensor sequence stride
    :type M_stride_s: int
    :param BLOCK_SIZE_Q: Tile size for query processing
    :type BLOCK_SIZE_Q: int
    :param BLOCK_SIZE_KV: Tile size for key-value processing
    :type BLOCK_SIZE_KV: int
    :param HEAD_DIM: Head dimension (must match head_dim)
    :type HEAD_DIM: int
    :param SM_SCALE: Attention scaling factor (typically 1/√head_dim)
    :type SM_SCALE: float
    :param K2_BIAS: Additive bias for K2 tensor
    :type K2_BIAS: float
    :param V2_BIAS: Additive bias for V2 tensor
    :type V2_BIAS: float
    :param USE_ALIBI3D: Whether to apply ALiBi3D positional bias
    :type USE_ALIBI3D: bool
    :param ALIBI_ALPHA: Power factor for distance computation in ALiBi3D
    :type ALIBI_ALPHA: float
    :param num_stages: Pipeline stages for memory optimization
    :type num_stages: int
    :param DATA_DTYPE: Storage precision for input/output tensors
    :type DATA_DTYPE: tl.dtype
    :param COMPUTE_DTYPE: Computation precision for accumulation and arithmetic
    :type COMPUTE_DTYPE: tl.dtype
    :param GEMM_DTYPE: Matrix multiplication precision for performance
    :type GEMM_DTYPE: tl.dtype
    """

    # ============================================================================
    # THREAD BLOCK SETUP AND MEMORY INDEXING
    # ============================================================================
    # Each thread block processes BLOCK_SIZE_Q query positions
    q_start = tl.program_id(0) * BLOCK_SIZE_Q
    q_end = q_start + BLOCK_SIZE_Q

    # Decode batch and head indices from program_id(1)
    bk = tl.program_id(1)
    offs_b = bk // num_heads  # Batch index
    offs_k = bk % num_heads  # Head index

    # Calculate base memory offsets for current batch/head
    qkv_offs_bk = offs_b * q_stride_b + offs_k * q_stride_k
    Q_ptr += qkv_offs_bk
    K1_ptr += qkv_offs_bk
    K2_ptr += qkv_offs_bk
    V1_ptr += qkv_offs_bk
    V2_ptr += qkv_offs_bk
    O_ptr += qkv_offs_bk
    M_ptr += offs_b * m_stride_b + offs_k * m_stride_k

    # ============================================================================
    # INITIALIZE ONLINE SOFTMAX ACCUMULATORS AND LOAD Q-TILE
    # ============================================================================
    # Online softmax state: m_i tracks max logits, l_i tracks sum of exp
    m_i = tl.zeros((BLOCK_SIZE_Q,), dtype=COMPUTE_DTYPE) - float("inf")
    l_i = tl.zeros((BLOCK_SIZE_Q,), dtype=COMPUTE_DTYPE)
    acc = tl.zeros((BLOCK_SIZE_Q, HEAD_DIM), dtype=COMPUTE_DTYPE)

    # Load query tile with proper masking
    q_offs_s = q_start + tl.arange(0, BLOCK_SIZE_Q)
    qkv_offs_h = tl.arange(0, HEAD_DIM)
    q_mask_s = q_offs_s < seq_len
    qkv_mask_h = qkv_offs_h < head_dim

    q_offs = q_offs_s[:, None] * q_stride_s + qkv_offs_h[None, :] * q_stride_h
    q_mask = q_mask_s[:, None] & (qkv_mask_h[None, :])
    q_tile = tl.load(Q_ptr + q_offs, mask=q_mask).to(COMPUTE_DTYPE)
    softmax_scale = tl.cast(SM_SCALE, GEMM_DTYPE)

    # ============================================================================
    # DUAL-KEY ATTENTION WITH ONLINE ALiBi3D BIAS
    # ============================================================================
    # Load ALiBi slope for current head (done once per kernel)
    if USE_ALIBI3D:
        slope = tl.load(SLOPES_ptr + offs_k * slopes_stride_h)

    # Outer loop: iterate over K1/V1 positions within local window
    kv1_upper = tl.minimum(seq_len, q_end) if CAUSAL else tl.minimum(seq_len, q_end + w1)
    for kv1_idx in tl.range(tl.maximum(0, q_start - w1), kv1_upper):
        # Load single K1 vector and broadcast for matrix multiplication
        k1_offs = kv1_idx * k1_stride_s + qkv_offs_h * k1_stride_h
        k1_tile = (tl.load(K1_ptr + k1_offs, mask=qkv_mask_h).to(COMPUTE_DTYPE))[None, :]

        # Compute Q * K1 (element-wise, will be used in (Q ⊙ K1) @ K2^T)
        qk1 = (q_tile * k1_tile).to(GEMM_DTYPE)

        # Load corresponding V1 vector
        v1_offs = kv1_idx * v1_stride_s + qkv_offs_h * v1_stride_h
        v1_tile = (tl.load(V1_ptr + v1_offs, mask=qkv_mask_h).to(COMPUTE_DTYPE))[None, :]

        # Inner loop: iterate over K2/V2 positions in blocks for efficiency
        kv2_upper = tl.minimum(seq_len, q_end) if CAUSAL else tl.minimum(seq_len, q_end + w2)
        for kv2_idx in tl.range(
            tl.maximum(0, q_start - w2),
            kv2_upper,
            BLOCK_SIZE_KV,
            num_stages=num_stages,
        ):
            # Load K2 and V2 blocks with masking
            kv2_offs_s = kv2_idx + tl.arange(0, BLOCK_SIZE_KV)
            kv2_mask_s = kv2_offs_s < seq_len
            k2t_mask = kv2_mask_s[None, :] & qkv_mask_h[:, None]  # For transposed K2
            v2_mask = kv2_mask_s[:, None] & qkv_mask_h[None, :]

            k2_offs = kv2_offs_s[None, :] * k2_stride_s + qkv_offs_h[:, None] * k2_stride_h
            v2_offs = kv2_offs_s[:, None] * v2_stride_s + qkv_offs_h[None, :] * v2_stride_h

            k2t_tile = tl.load(K2_ptr + k2_offs, mask=k2t_mask).to(COMPUTE_DTYPE)
            v2_tile = tl.load(V2_ptr + v2_offs, mask=v2_mask).to(COMPUTE_DTYPE)

            # Apply biases to K2 and V2
            k2t_tile += K2_BIAS
            v2_tile += V2_BIAS
            k2t_tile = k2t_tile.to(GEMM_DTYPE)
            v2_tile = v2_tile.to(COMPUTE_DTYPE)

            # Compute attention logits: (Q * K1) @ K2^T * scale
            qk = tl.dot(qk1 * softmax_scale, k2t_tile, input_precision="tf32", out_dtype=tl.float32)

            # --- ONLINE ALiBi3D POSITIONAL BIAS ---
            if USE_ALIBI3D:
                # Compute 3D distances: i=query_pos, j=k1_pos, k=k2_pos
                # Distance metric: |i-j|^α + |i-k|^α
                dist_ij = tl.abs(q_offs_s[:, None] - kv1_idx)
                dist_ik = tl.abs(q_offs_s[:, None] - kv2_offs_s[None, :])

                # Apply power scaling if α ≠ 1
                if ALIBI_ALPHA != 1.0:
                    # eps = 1e-7  # Numerical stability
                    eps = 1e-7
                    dist_ij = tl.exp(ALIBI_ALPHA * tl.log(dist_ij.to(tl.float32) + eps))
                    dist_ik = tl.exp(ALIBI_ALPHA * tl.log(dist_ik.to(tl.float32) + eps))

                combined_dist = dist_ij + dist_ik
                alibi_bias = -slope * combined_dist
                qk += alibi_bias.to(qk.dtype)

            # --- CAUSAL AND LOCAL ATTENTION MASKING ---
            # Base sequence masking
            qk_mask = q_mask_s[:, None] & kv2_mask_s[None, :]

            # Local window constraints
            if CAUSAL:
                kv1_local_mask = ((q_offs_s[:, None] - w1) < kv1_idx) & (kv1_idx <= q_offs_s[:, None])
                kv2_local_mask = ((q_offs_s[:, None] - w2) < kv2_offs_s[None, :]) & (
                    kv2_offs_s[None, :] <= q_offs_s[:, None]
                )
            else:
                kv1_local_mask = ((q_offs_s[:, None] - w1) < kv1_idx) & (kv1_idx < (q_offs_s[:, None] + w1))
                kv2_local_mask = ((q_offs_s[:, None] - w2) < kv2_offs_s[None, :]) & (
                    kv2_offs_s[None, :] < (q_offs_s[:, None] + w2)
                )

            # Combine all masking conditions
            qk_mask &= kv1_local_mask & kv2_local_mask
            qk += tl.where(qk_mask, 0, -1.0e38)  # Large negative for masked positions

            # --- ONLINE SOFTMAX UPDATE ---
            # Update running maximum and compute probabilities
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            p = tl.math.exp(qk - m_ij[:, None])
            l_ij = tl.sum(p, 1)

            # Rescale previous accumulator and add current contribution
            alpha = tl.math.exp(m_i - m_ij)
            l_i = l_i * alpha + l_ij
            acc = acc * alpha[:, None]

            # Compute value contribution: V1 ⊙ V2 weighted by attention
            v12_tile = v1_tile * v2_tile  # Element-wise product
            acc += tl.dot(p.to(GEMM_DTYPE), v12_tile.to(GEMM_DTYPE), input_precision="ieee", out_dtype=tl.float32)
            m_i = m_ij

    # ============================================================================
    # FINALIZE AND STORE OUTPUT
    # ============================================================================
    # Apply final softmax normalization
    acc = acc / l_i[:, None]
    acc = tl.where(q_mask, acc, 0.0).to(DATA_DTYPE)

    # Store output tensor
    out_offs = q_offs_s[:, None] * out_stride_s + qkv_offs_h[None, :] * out_stride_h
    tl.store(O_ptr + out_offs, acc, mask=q_mask)

    # Store log-sum-exp values for potential backward pass
    m = m_i + tl.log(l_i)
    m_offs = q_offs_s * m_stride_s
    m_mask = q_offs_s < seq_len
    tl.store(M_ptr + m_offs, m, mask=m_mask)


@triton.jit
def two_simplicial_attn_bwd_kv1_kernel(
    Q_ptr,
    K1_ptr,
    K2_ptr,
    V1_ptr,
    V2_ptr,
    dO_ptr,
    M_ptr,
    D_ptr,  # Input tensors
    SLOPES_ptr,  # ALiBi slopes tensor [num_heads]
    dQ_ptr,
    dK1_ptr,
    dV1_ptr,  # Output gradient tensors
    bs,
    seq_len,
    num_heads,
    head_dim,
    w1,
    w2,  # Shape and window parameters
    # Stride parameters grouped by tensor
    q_stride_b,
    q_stride_s,
    q_stride_k,
    q_stride_h,
    k1_stride_b,
    k1_stride_s,
    k1_stride_k,
    k1_stride_h,
    k2_stride_b,
    k2_stride_s,
    k2_stride_k,
    k2_stride_h,
    v1_stride_b,
    v1_stride_s,
    v1_stride_k,
    v1_stride_h,
    v2_stride_b,
    v2_stride_s,
    v2_stride_k,
    v2_stride_h,
    dO_stride_b,
    dO_stride_s,
    dO_stride_k,
    dO_stride_h,
    m_stride_b,
    m_stride_k,
    m_stride_s,
    d_stride_b,
    d_stride_k,
    d_stride_s,
    slopes_stride_h,  # Stride for slopes tensor
    dq_stride_b,
    dq_stride_s,
    dq_stride_k,
    dq_stride_h,
    dk1_stride_b,
    dk1_stride_s,
    dk1_stride_k,
    dk1_stride_h,
    dv1_stride_b,
    dv1_stride_s,
    dv1_stride_k,
    dv1_stride_h,
    # Compile-time constants
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_KV: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    SM_SCALE: tl.constexpr,
    K2_BIAS: tl.constexpr,
    V2_BIAS: tl.constexpr,
    USE_ALIBI3D: tl.constexpr,
    ALIBI_ALPHA: tl.constexpr,
    CAUSAL: tl.constexpr,
    COMPUTE_DQ: tl.constexpr,
    is_flipped: tl.constexpr,
    # dtype parameters
    DATA_DTYPE: tl.constexpr,
    COMPUTE_DTYPE: tl.constexpr,
    GEMM_DTYPE: tl.constexpr,
):
    """Two Simplicial Attention Backward Kernel for K1/V1 Gradients.

    Computes gradients for K1, V1, and optionally Q in the Two Simplicial Attention
    mechanism from https://arxiv.org/abs/2507.02754. This kernel processes blocks
    of K1/V1 positions and computes their gradients by iterating over all relevant
    K2/V2 and query positions.

    Mathematical Background
    -----------------------

    For the Two Simplicial Attention mechanism::

        Attention(Q, K1, K2, V1, V2) = softmax((Q ⊙ K1) @ K2^T / √d) @ (V1 ⊙ V2)

    The backward pass computes:

    - ∂L/∂K1: Gradient w.r.t. first key tensor
    - ∂L/∂V1: Gradient w.r.t. first value tensor
    - ∂L/∂Q: Gradient w.r.t. query tensor (optional)

    Gradient Computation Strategy
    -----------------------------

    This kernel uses a K1/V1-centric approach:

    1. Each thread block processes a tile of K1/V1 positions
    2. For each K1/V1 position, iterate over all valid K2/V2 positions
    3. For each (K1,K2) pair, iterate over all query positions that can attend to this pair
    4. Accumulate gradients using the chain rule and saved forward pass values

    The gradients are computed as::

        dV1[j] = Σ_i,k P[i,j,k] * dO[i] * V2[k]
        dK1[j] = Σ_i,k (∂s/∂K1[j]) * K2[k] * scale
        dQ[i] = Σ_j,k (∂s/∂Q[i]) * (K1[j] ⊙ K2[k]) * scale

    where P[i,j,k] are the attention probabilities and s are the pre-softmax scores.

    Kernel Architecture
    -------------------

    - Thread blocks are launched per (batch, head, K1/V1_block)
    - Each block processes BLOCK_SIZE_KV K1/V1 positions simultaneously
    - Triple nested loops: K2 positions → Q positions → accumulate gradients
    - Uses atomic operations for dQ to handle overlapping query contributions
    - Leverages saved forward pass values (M, D) for efficient gradient computation

    :param Q_ptr: Query tensor from forward pass [batch, seq_len, num_heads, head_dim]
    :type Q_ptr: Tensor pointer
    :param K1_ptr: First key tensor from forward pass with same shape as Q
    :type K1_ptr: Tensor pointer
    :param K2_ptr: Second key tensor from forward pass with same shape as Q
    :type K2_ptr: Tensor pointer
    :param V1_ptr: First value tensor from forward pass with same shape as Q
    :type V1_ptr: Tensor pointer
    :param V2_ptr: Second value tensor from forward pass with same shape as Q
    :type V2_ptr: Tensor pointer
    :param dO_ptr: Gradient of output [batch, seq_len, num_heads, head_dim]
    :type dO_ptr: Tensor pointer
    :param M_ptr: Saved max values from forward pass [batch, num_heads, seq_len]
    :type M_ptr: Tensor pointer
    :param D_ptr: Saved row sums from forward pass [batch, num_heads, seq_len]
    :type D_ptr: Tensor pointer
    :param dQ_ptr: Output gradient tensor for queries with same shape as Q
    :type dQ_ptr: Tensor pointer
    :param dK1_ptr: Output gradient tensor for K1 with same shape as K1
    :type dK1_ptr: Tensor pointer
    :param dV1_ptr: Output gradient tensor for V1 with same shape as V1
    :type dV1_ptr: Tensor pointer
    :param bs: Batch size dimension
    :type bs: int
    :param seq_len: Sequence length dimension
    :type seq_len: int
    :param num_heads: Number of attention heads dimension
    :type num_heads: int
    :param head_dim: Head dimension
    :type head_dim: int
    :param w1: Local attention window size for K1/V1
    :type w1: int
    :param w2: Local attention window size for K2/V2
    :type w2: int
    :param Q_stride_b: Query tensor batch stride
    :type Q_stride_b: int
    :param Q_stride_s: Query tensor sequence stride
    :type Q_stride_s: int
    :param Q_stride_k: Query tensor head stride
    :type Q_stride_k: int
    :param Q_stride_h: Query tensor dimension stride
    :type Q_stride_h: int
    :param K1_stride_b: K1 tensor batch stride
    :type K1_stride_b: int
    :param K1_stride_s: K1 tensor sequence stride
    :type K1_stride_s: int
    :param K1_stride_k: K1 tensor head stride
    :type K1_stride_k: int
    :param K1_stride_h: K1 tensor dimension stride
    :type K1_stride_h: int
    :param K2_stride_b: K2 tensor batch stride
    :type K2_stride_b: int
    :param K2_stride_s: K2 tensor sequence stride
    :type K2_stride_s: int
    :param K2_stride_k: K2 tensor head stride
    :type K2_stride_k: int
    :param K2_stride_h: K2 tensor dimension stride
    :type K2_stride_h: int
    :param V1_stride_b: V1 tensor batch stride
    :type V1_stride_b: int
    :param V1_stride_s: V1 tensor sequence stride
    :type V1_stride_s: int
    :param V1_stride_k: V1 tensor head stride
    :type V1_stride_k: int
    :param V1_stride_h: V1 tensor dimension stride
    :type V1_stride_h: int
    :param V2_stride_b: V2 tensor batch stride
    :type V2_stride_b: int
    :param V2_stride_s: V2 tensor sequence stride
    :type V2_stride_s: int
    :param V2_stride_k: V2 tensor head stride
    :type V2_stride_k: int
    :param V2_stride_h: V2 tensor dimension stride
    :type V2_stride_h: int
    :param dO_stride_b: Output gradient tensor batch stride
    :type dO_stride_b: int
    :param dO_stride_s: Output gradient tensor sequence stride
    :type dO_stride_s: int
    :param dO_stride_k: Output gradient tensor head stride
    :type dO_stride_k: int
    :param dO_stride_h: Output gradient tensor dimension stride
    :type dO_stride_h: int
    :param M_stride_b: M tensor batch stride
    :type M_stride_b: int
    :param M_stride_k: M tensor head stride
    :type M_stride_k: int
    :param M_stride_s: M tensor sequence stride
    :type M_stride_s: int
    :param D_stride_b: D tensor batch stride
    :type D_stride_b: int
    :param D_stride_k: D tensor head stride
    :type D_stride_k: int
    :param D_stride_s: D tensor sequence stride
    :type D_stride_s: int
    :param dQ_stride_b: Query gradient tensor batch stride
    :type dQ_stride_b: int
    :param dQ_stride_s: Query gradient tensor sequence stride
    :type dQ_stride_s: int
    :param dQ_stride_k: Query gradient tensor head stride
    :type dQ_stride_k: int
    :param dQ_stride_h: Query gradient tensor dimension stride
    :type dQ_stride_h: int
    :param dK1_stride_b: K1 gradient tensor batch stride
    :type dK1_stride_b: int
    :param dK1_stride_s: K1 gradient tensor sequence stride
    :type dK1_stride_s: int
    :param dK1_stride_k: K1 gradient tensor head stride
    :type dK1_stride_k: int
    :param dK1_stride_h: K1 gradient tensor dimension stride
    :type dK1_stride_h: int
    :param dV1_stride_b: V1 gradient tensor batch stride
    :type dV1_stride_b: int
    :param dV1_stride_s: V1 gradient tensor sequence stride
    :type dV1_stride_s: int
    :param dV1_stride_k: V1 gradient tensor head stride
    :type dV1_stride_k: int
    :param dV1_stride_h: V1 gradient tensor dimension stride
    :type dV1_stride_h: int
    :param BLOCK_SIZE_Q: Tile size for query processing
    :type BLOCK_SIZE_Q: int
    :param BLOCK_SIZE_KV: Tile size for key-value processing
    :type BLOCK_SIZE_KV: int
    :param HEAD_DIM: Head dimension (must match head_dim)
    :type HEAD_DIM: int
    :param SM_SCALE: Attention scaling factor (typically 1/√head_dim)
    :type SM_SCALE: float
    :param K2_BIAS: Additive bias for K2 tensor
    :type K2_BIAS: float
    :param V2_BIAS: Additive bias for V2 tensor
    :type V2_BIAS: float
    :param COMPUTE_DQ: Whether to compute query gradients (may be handled by separate kernel)
    :type COMPUTE_DQ: bool
    :param is_flipped: Whether this kernel handles the "flipped" case (K1↔K2, V1↔V2 swapped)
    :type is_flipped: bool
    :param DATA_DTYPE: Storage precision for input/output tensors
    :type DATA_DTYPE: tl.dtype
    :param COMPUTE_DTYPE: Computation precision for accumulation and arithmetic
    :type COMPUTE_DTYPE: tl.dtype
    :param GEMM_DTYPE: Matrix multiplication precision for performance
    :type GEMM_DTYPE: tl.dtype
    """
    # ============================================================================
    # THREAD BLOCK SETUP - PROCESS K1/V1 POSITIONS
    # ============================================================================
    # Each thread block processes BLOCK_SIZE_KV K1/V1 positions
    kv1_start = tl.program_id(0) * BLOCK_SIZE_KV
    kv1_end = kv1_start + BLOCK_SIZE_KV

    # Decode batch and head indices from program_id(1)
    bk = tl.program_id(1)
    offs_b = bk // num_heads  # Batch index
    offs_k = bk % num_heads  # Head index

    # Calculate base memory offsets for current batch/head
    qkv_offs_bk = offs_b * q_stride_b + offs_k * q_stride_k
    Q_ptr += qkv_offs_bk
    K1_ptr += qkv_offs_bk
    K2_ptr += qkv_offs_bk
    V1_ptr += qkv_offs_bk
    V2_ptr += qkv_offs_bk

    dO_ptr += offs_b * dO_stride_b + offs_k * dO_stride_k
    M_ptr += offs_b * m_stride_b + offs_k * m_stride_k
    D_ptr += offs_b * d_stride_b + offs_k * d_stride_k
    dK1_ptr += offs_b * dk1_stride_b + offs_k * dk1_stride_k
    dV1_ptr += offs_b * dv1_stride_b + offs_k * dv1_stride_k
    if COMPUTE_DQ:  # Query gradients may be computed in separate kernel
        dQ_ptr += offs_b * dq_stride_b + offs_k * dq_stride_k

    softmax_scale = tl.cast(SM_SCALE, GEMM_DTYPE)
    qkv_offs_h = tl.arange(0, HEAD_DIM)
    qkv_mask_h = qkv_offs_h < head_dim

    # Load ALiBi slope for current head (done once per kernel)
    if USE_ALIBI3D:
        slope = tl.load(SLOPES_ptr + offs_k * slopes_stride_h)

    # ============================================================================
    # LOAD CURRENT K1/V1 BLOCK
    # ============================================================================
    kv1_offs_s = kv1_start + tl.arange(0, BLOCK_SIZE_KV)

    k1_offs = kv1_offs_s[:, None] * k1_stride_s + qkv_offs_h[None, :] * k1_stride_h
    kv1_mask_s = kv1_offs_s < seq_len
    kv1_mask = kv1_mask_s[:, None] & qkv_mask_h[None, :]
    k1_tile = tl.load(K1_ptr + k1_offs, mask=kv1_mask).to(COMPUTE_DTYPE)  # [BLOCK_SIZE_KV, HEAD_DIM]
    v1_offs = kv1_offs_s[:, None] * v1_stride_s + qkv_offs_h[None, :] * v1_stride_h
    v1_tile = tl.load(V1_ptr + v1_offs, mask=kv1_mask).to(COMPUTE_DTYPE)  # [BLOCK_SIZE_KV, HEAD_DIM]

    # Apply biases based on kernel mode (normal vs flipped)
    if is_flipped:
        k1_tile += K2_BIAS
        v1_tile += V2_BIAS

    # Initialize gradient accumulation for this K1/V1 block
    dv1 = tl.zeros((BLOCK_SIZE_KV, HEAD_DIM), COMPUTE_DTYPE)
    dk1 = tl.zeros((BLOCK_SIZE_KV, HEAD_DIM), COMPUTE_DTYPE)

    # ============================================================================
    # ITERATE OVER K2/V2 POSITIONS
    # ============================================================================
    # For each K1 position, we need to consider all K2 positions that could
    # form valid attention pairs. The range accounts for local window constraints.
    kv2_bwd_upper = tl.minimum(seq_len, kv1_end + w1) if CAUSAL else tl.minimum(seq_len, kv1_end + w1 + w2)
    for kv2_idx in tl.range(tl.maximum(0, kv1_start - w2), kv2_bwd_upper):
        # Load single K2/V2 vectors for current position
        k2_offs = kv2_idx * k2_stride_s + qkv_offs_h * k2_stride_h
        k2_tile = (tl.load(K2_ptr + k2_offs, mask=qkv_mask_h).to(COMPUTE_DTYPE))[None, :]  # [1, HEAD_DIM]
        v2_offs = kv2_idx * v2_stride_s + qkv_offs_h * v2_stride_h
        v2_tile = (tl.load(V2_ptr + v2_offs, mask=qkv_mask_h).to(COMPUTE_DTYPE))[None, :]  # [1, HEAD_DIM]

        # Apply biases based on kernel mode
        if not is_flipped:
            k2_tile += K2_BIAS
            v2_tile += V2_BIAS

        # Compute element-wise products for attention and value computation
        k1k2 = k1_tile * k2_tile  # [BLOCK_SIZE_KV, HEAD_DIM]
        v1v2 = v1_tile * v2_tile  # [BLOCK_SIZE_KV, HEAD_DIM]
        k1k2 = k1k2.to(GEMM_DTYPE)
        v1v2 = v1v2.to(GEMM_DTYPE)

        # ========================================================================
        # DETERMINE VALID QUERY RANGE FOR CURRENT (K1, K2) PAIR
        # ========================================================================
        # Query positions must satisfy local window constraints for both K1 and K2:
        # Causal:     kv1 ∈ (q-w1, q] and kv2 ∈ (q-w2, q]
        # Non-causal: kv1 ∈ (q-w1, q+w1) and kv2 ∈ (q-w2, q+w2)
        if CAUSAL:
            q_start = tl.maximum(kv1_start, kv2_idx)
            q_end = tl.minimum(seq_len, tl.minimum(kv1_end + w1, kv2_idx + w2))
        else:
            q_start = tl.maximum(tl.maximum(0, kv1_start - w1), kv2_idx - w2)
            q_end = tl.minimum(seq_len, tl.minimum(kv1_end + w1, kv2_idx + w2))

        # ========================================================================
        # ITERATE OVER QUERY POSITIONS IN BLOCKS
        # ========================================================================
        for q_idx in tl.range(q_start, q_end, BLOCK_SIZE_Q):
            # Load query block and saved forward pass values
            q_offs_s = q_idx + tl.arange(0, BLOCK_SIZE_Q)
            q_offs = q_offs_s[None, :] * q_stride_s + qkv_offs_h[:, None] * q_stride_h
            q_mask_s = q_offs_s < seq_len
            qt_mask = q_mask_s[None, :] & qkv_mask_h[:, None]
            qt_tile = tl.load(Q_ptr + q_offs, mask=qt_mask).to(GEMM_DTYPE)  # [HEAD_DIM, BLOCK_SIZE_Q]

            # Load saved forward pass values for numerical stability
            m_offs = q_offs_s * m_stride_s
            m_tile = tl.load(M_ptr + m_offs, mask=q_mask_s).to(COMPUTE_DTYPE)[
                None, :
            ]  # [1, BLOCK_SIZE_Q] - saved max values
            d_offs = q_offs_s * d_stride_s
            d_tile = tl.load(D_ptr + d_offs, mask=q_mask_s).to(COMPUTE_DTYPE)[
                None, :
            ]  # [1, BLOCK_SIZE_Q] - saved row sums

            # Load gradient of output
            dO_offs = q_offs_s[:, None] * dO_stride_s + qkv_offs_h[None, :] * dO_stride_h
            dO_tile = tl.load(dO_ptr + dO_offs, mask=q_mask_s[:, None] & qkv_mask_h[None, :]).to(
                COMPUTE_DTYPE
            )  # [BLOCK_SIZE_Q, HEAD_DIM]

            if COMPUTE_DQ:
                dq = tl.zeros((BLOCK_SIZE_Q, HEAD_DIM), tl.float32)

            # ====================================================================
            # COMPUTE ATTENTION SCORES AND PROBABILITIES
            # ====================================================================
            # Reconstruct attention scores: (K1 ⊙ K2) @ (Q * scale)^T
            qkkT = tl.dot(k1k2, qt_tile * softmax_scale, input_precision="tf32", out_dtype=tl.float32)  # [BLOCK_SIZE_KV, BLOCK_SIZE_Q]

            # --- RECOMPUTE ALiBi3D BIAS (same as forward) ---
            if USE_ALIBI3D:
                # Compute 3D distances: i=q_offs_s, j=kv1_offs_s, k=kv2_idx
                dist_ij = tl.abs(q_offs_s[None, :] - kv1_offs_s[:, None])
                dist_ik = tl.abs(q_offs_s[None, :] - kv2_idx)

                # Apply power scaling if α ≠ 1
                if ALIBI_ALPHA != 1.0:
                    eps = 1e-7
                    dist_ij = tl.exp(ALIBI_ALPHA * tl.log(dist_ij.to(tl.float32) + eps))
                    dist_ik = tl.exp(ALIBI_ALPHA * tl.log(dist_ik.to(tl.float32) + eps))

                combined_dist = dist_ij + dist_ik
                alibi_bias = -slope * combined_dist
                qkkT += alibi_bias.to(qkkT.dtype)

            # Apply local attention window constraints
            if CAUSAL:
                kv1_local_mask = ((q_offs_s[None, :] - w1) < kv1_offs_s[:, None]) & (
                    kv1_offs_s[:, None] <= q_offs_s[None, :]
                )
                kv2_local_mask = ((q_offs_s - w2) < kv2_idx) & (kv2_idx <= q_offs_s)
            else:
                kv1_local_mask = ((q_offs_s[None, :] - w1) < kv1_offs_s[:, None]) & (
                    kv1_offs_s[:, None] < (q_offs_s[None, :] + w1)
                )
                kv2_local_mask = ((q_offs_s - w2) < kv2_idx) & (kv2_idx < (q_offs_s + w2))
            local_mask = kv1_local_mask & kv2_local_mask[None, :]  # [BLOCK_SIZE_KV, BLOCK_SIZE_Q]

            # Mask invalid positions
            qkkT = tl.where(local_mask, qkkT, -1.0e38)

            # Compute attention probabilities using saved max values from forward pass
            pT = tl.exp(qkkT - m_tile)  # [BLOCK_SIZE_KV, BLOCK_SIZE_Q]
            pT = tl.where(local_mask, pT, 0.0)  # Zero out invalid positions

            # ====================================================================
            # COMPUTE GRADIENTS USING CHAIN RULE
            # ====================================================================
            # Gradient w.r.t. V1: dV1 = P^T @ (dO ⊙ V2)
            dOv2 = dO_tile * v2_tile  # [BLOCK_SIZE_Q, HEAD_DIM]
            dv1 += tl.dot(pT.to(GEMM_DTYPE), dOv2.to(GEMM_DTYPE), input_precision="ieee", out_dtype=tl.float32)  # [BLOCK_SIZE_KV, HEAD_DIM]

            # Compute gradient w.r.t. pre-softmax scores
            # First: gradient w.r.t. attention output = (V1 ⊙ V2)^T @ dO
            dpT = tl.dot(v1v2, tl.trans(dO_tile.to(GEMM_DTYPE)), input_precision="ieee", out_dtype=tl.float32)  # [BLOCK_SIZE_KV, BLOCK_SIZE_Q]

            # Apply softmax backward: ds = P ⊙ (dp - d) where d is row sum from forward
            dsT = pT * (dpT - d_tile)  # [BLOCK_SIZE_KV, BLOCK_SIZE_Q]
            dsT = tl.where(local_mask, dsT, 0.0)
            dsT = dsT.to(GEMM_DTYPE)

            # Gradient w.r.t. K1: dK1 = (ds @ Q^T) ⊙ K2 * scale
            dk1 += tl.dot(dsT, tl.trans(qt_tile), input_precision="ieee", out_dtype=tl.float32) * k2_tile.to(tl.float32) * softmax_scale

            # Gradient w.r.t. Q (optional): dQ = ds^T @ (K1 ⊙ K2) * scale
            if COMPUTE_DQ:
                dq += tl.dot(tl.trans(dsT), k1k2, input_precision="ieee", out_dtype=tl.float32) * softmax_scale  # [BLOCK_SIZE_Q, HEAD_DIM]

                # Store query gradients using atomic operations (multiple blocks contribute)
                dq_offs = q_offs_s[:, None] * dq_stride_s + qkv_offs_h[None, :] * dq_stride_h
                tl.atomic_add(dQ_ptr + dq_offs, dq, mask=q_mask_s[:, None] & qkv_mask_h[None, :])

    # ============================================================================
    # STORE K1/V1 GRADIENTS
    # ============================================================================
    dv1_offs = kv1_offs_s[:, None] * dv1_stride_s + qkv_offs_h[None, :] * dv1_stride_h
    dk1_offs = kv1_offs_s[:, None] * dk1_stride_s + qkv_offs_h[None, :] * dk1_stride_h
    tl.store(dV1_ptr + dv1_offs, dv1.to(DATA_DTYPE), mask=kv1_mask)
    tl.store(dK1_ptr + dk1_offs, dk1.to(DATA_DTYPE), mask=kv1_mask)


@triton.autotune(
    configs=[
        Config(
            {
                "BLOCK_SIZE_Q": 32,
                "BLOCK_SIZE_KV2": 64,
                "num_stages": 1,
            },
            num_warps=4,
        )
    ],
    key=["HEAD_DIM"],
)
@triton.jit
def two_simplicial_attn_bwd_kv2q_kernel(
    Q_ptr,
    K1_ptr,
    K2_ptr,
    V1_ptr,
    V2_ptr,
    dO_ptr,
    M_ptr,
    D_ptr,  # Input tensors
    SLOPES_ptr,  # ALiBi slopes tensor [num_heads]
    dQ_ptr,
    dK2_ptr,
    dV2_ptr,  # Output gradient tensors
    bs,
    seq_len,
    num_heads,
    head_dim,
    w1,
    w2,  # Shape and window parameters
    # Stride parameters grouped by tensor
    q_stride_b,
    q_stride_s,
    q_stride_k,
    q_stride_h,
    k1_stride_b,
    k1_stride_s,
    k1_stride_k,
    k1_stride_h,
    k2_stride_b,
    k2_stride_s,
    k2_stride_k,
    k2_stride_h,
    v1_stride_b,
    v1_stride_s,
    v1_stride_k,
    v1_stride_h,
    v2_stride_b,
    v2_stride_s,
    v2_stride_k,
    v2_stride_h,
    dO_stride_b,
    dO_stride_s,
    dO_stride_k,
    dO_stride_h,
    m_stride_b,
    m_stride_k,
    m_stride_s,
    d_stride_b,
    d_stride_k,
    d_stride_s,
    slopes_stride_h,  # Stride for slopes tensor
    dq_stride_b,
    dq_stride_s,
    dq_stride_k,
    dq_stride_h,
    dk2_stride_b,
    dk2_stride_s,
    dk2_stride_k,
    dk2_stride_h,
    dv2_stride_b,
    dv2_stride_s,
    dv2_stride_k,
    dv2_stride_h,
    # Compile-time constants
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_KV2: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    SM_SCALE: tl.constexpr,
    K2_BIAS: tl.constexpr,
    V2_BIAS: tl.constexpr,
    USE_ALIBI3D: tl.constexpr,
    ALIBI_ALPHA: tl.constexpr,
    CAUSAL: tl.constexpr,
    num_stages: tl.constexpr,
    IS_SECOND_PASS: tl.constexpr,
    # dtype parameters
    DATA_DTYPE: tl.constexpr,
    COMPUTE_DTYPE: tl.constexpr,
    GEMM_DTYPE: tl.constexpr,
):
    """Two Simplicial Attention Backward Kernel for K2/V2/Q Gradients.

    Computes gradients for K2, V2, and Q in the Two Simplicial Attention
    mechanism from https://arxiv.org/abs/2507.02754. This kernel uses a
    Q/K2/V2-centric approach with a two-pass strategy to handle overlapping
    memory accesses efficiently.

    Mathematical Background
    -----------------------

    For the Two Simplicial Attention mechanism::

        Attention(Q, K1, K2, V1, V2) = softmax((Q ⊙ K1) @ K2^T / √d) @ (V1 ⊙ V2)

    The backward pass computes:

    - ∂L/∂K2: Gradient w.r.t. second key tensor
    - ∂L/∂V2: Gradient w.r.t. second value tensor
    - ∂L/∂Q: Gradient w.r.t. query tensor

    Two-Pass Strategy
    -----------------

    Due to overlapping memory access patterns in the local attention windows,
    this kernel uses a two-pass approach:

    **Pass 1 (IS_SECOND_PASS=False)**: Processes "even" tiles
        - q_start = program_id(0) * BLOCK_SIZE_KV2
        - Processes positions [q_start, q_start + BLOCK_SIZE_Q]
        - Writes initial gradients to output tensors

    **Pass 2 (IS_SECOND_PASS=True)**: Processes "odd" tiles
        - q_start = program_id(0) * BLOCK_SIZE_KV2 + BLOCK_SIZE_Q
        - Processes positions [q_start, q_start + BLOCK_SIZE_Q]
        - Reads previous gradients and accumulates additional contributions

    This prevents race conditions while maximizing parallelism.

    Kernel Architecture
    -------------------

    Unlike the K1/V1 kernel, this one processes:

    1. **Fixed Q block**: BLOCK_SIZE_Q query positions
    2. **Fixed K2/V2 block**: BLOCK_SIZE_KV2 = BLOCK_SIZE_Q + w2 positions
    3. **Variable K1/V1**: Iterates over individual K1/V1 positions

    The constraint BLOCK_SIZE_KV2 = BLOCK_SIZE_Q + w2 ensures that all
    K2/V2 positions that can attend to the Q block are loaded together.

    Gradient Computation Strategy
    -----------------------------

    For each (Q, K2/V2) block pair:

    1. Load Q, K2, V2 blocks and saved forward pass values (M, D)
    2. For each K1/V1 position that can attend to this Q block:

    a. Compute attention scores: (Q ⊙ K1) @ K2^T / √d
    b. Reconstruct probabilities using saved max values
    c. Apply local window masking
    d. Accumulate gradients::

            dV2 += P^T @ (dO ⊙ V1)
            dK2 += ds @ (Q ⊙ K1)
            dQ += ds^T @ (K1 ⊙ K2)

    Design Rationale
    ----------------

    This Q/K2/V2-centric approach is complementary to the K1/V1-centric kernel:

    - **K1/V1 kernel**: Efficient for computing K1, V1 gradients (processes K1/V1 blocks)
    - **K2/V2/Q kernel**: Efficient for K2, V2, Q gradients (processes Q/K2/V2 blocks)

    The two-pass strategy handles the fundamental challenge that multiple thread
    blocks may need to write to overlapping memory regions due to the local
    attention windows, which would cause race conditions in a naive implementation.

    :param Q_ptr: Query tensor from forward pass [batch, seq_len, num_heads, head_dim]
    :type Q_ptr: Tensor pointer
    :param K1_ptr: First key tensor from forward pass with same shape as Q
    :type K1_ptr: Tensor pointer
    :param K2_ptr: Second key tensor from forward pass with same shape as Q
    :type K2_ptr: Tensor pointer
    :param V1_ptr: First value tensor from forward pass with same shape as Q
    :type V1_ptr: Tensor pointer
    :param V2_ptr: Second value tensor from forward pass with same shape as Q
    :type V2_ptr: Tensor pointer
    :param dO_ptr: Gradient of output [batch, seq_len, num_heads, head_dim]
    :type dO_ptr: Tensor pointer
    :param M_ptr: Saved max values from forward pass [batch, num_heads, seq_len]
    :type M_ptr: Tensor pointer
    :param D_ptr: Saved row sums from forward pass [batch, num_heads, seq_len]
    :type D_ptr: Tensor pointer
    :param dQ_ptr: Output gradient tensor for queries with same shape as Q
    :type dQ_ptr: Tensor pointer
    :param dK2_ptr: Output gradient tensor for K2 with same shape as K2
    :type dK2_ptr: Tensor pointer
    :param dV2_ptr: Output gradient tensor for V2 with same shape as V2
    :type dV2_ptr: Tensor pointer
    :param bs: Batch size dimension
    :type bs: int
    :param seq_len: Sequence length dimension
    :type seq_len: int
    :param num_heads: Number of attention heads dimension
    :type num_heads: int
    :param head_dim: Head dimension
    :type head_dim: int
    :param w1: Local attention window size for K1/V1
    :type w1: int
    :param w2: Local attention window size for K2/V2
    :type w2: int
    :param Q_stride_b: Query tensor batch stride
    :type Q_stride_b: int
    :param Q_stride_s: Query tensor sequence stride
    :type Q_stride_s: int
    :param Q_stride_k: Query tensor head stride
    :type Q_stride_k: int
    :param Q_stride_h: Query tensor dimension stride
    :type Q_stride_h: int
    :param K1_stride_b: K1 tensor batch stride
    :type K1_stride_b: int
    :param K1_stride_s: K1 tensor sequence stride
    :type K1_stride_s: int
    :param K1_stride_k: K1 tensor head stride
    :type K1_stride_k: int
    :param K1_stride_h: K1 tensor dimension stride
    :type K1_stride_h: int
    :param K2_stride_b: K2 tensor batch stride
    :type K2_stride_b: int
    :param K2_stride_s: K2 tensor sequence stride
    :type K2_stride_s: int
    :param K2_stride_k: K2 tensor head stride
    :type K2_stride_k: int
    :param K2_stride_h: K2 tensor dimension stride
    :type K2_stride_h: int
    :param V1_stride_b: V1 tensor batch stride
    :type V1_stride_b: int
    :param V1_stride_s: V1 tensor sequence stride
    :type V1_stride_s: int
    :param V1_stride_k: V1 tensor head stride
    :type V1_stride_k: int
    :param V1_stride_h: V1 tensor dimension stride
    :type V1_stride_h: int
    :param V2_stride_b: V2 tensor batch stride
    :type V2_stride_b: int
    :param V2_stride_s: V2 tensor sequence stride
    :type V2_stride_s: int
    :param V2_stride_k: V2 tensor head stride
    :type V2_stride_k: int
    :param V2_stride_h: V2 tensor dimension stride
    :type V2_stride_h: int
    :param dO_stride_b: Output gradient tensor batch stride
    :type dO_stride_b: int
    :param dO_stride_s: Output gradient tensor sequence stride
    :type dO_stride_s: int
    :param dO_stride_k: Output gradient tensor head stride
    :type dO_stride_k: int
    :param dO_stride_h: Output gradient tensor dimension stride
    :type dO_stride_h: int
    :param M_stride_b: M tensor batch stride
    :type M_stride_b: int
    :param M_stride_k: M tensor head stride
    :type M_stride_k: int
    :param M_stride_s: M tensor sequence stride
    :type M_stride_s: int
    :param D_stride_b: D tensor batch stride
    :type D_stride_b: int
    :param D_stride_k: D tensor head stride
    :type D_stride_k: int
    :param D_stride_s: D tensor sequence stride
    :type D_stride_s: int
    :param dQ_stride_b: Query gradient tensor batch stride
    :type dQ_stride_b: int
    :param dQ_stride_s: Query gradient tensor sequence stride
    :type dQ_stride_s: int
    :param dQ_stride_k: Query gradient tensor head stride
    :type dQ_stride_k: int
    :param dQ_stride_h: Query gradient tensor dimension stride
    :type dQ_stride_h: int
    :param dK2_stride_b: K2 gradient tensor batch stride
    :type dK2_stride_b: int
    :param dK2_stride_s: K2 gradient tensor sequence stride
    :type dK2_stride_s: int
    :param dK2_stride_k: K2 gradient tensor head stride
    :type dK2_stride_k: int
    :param dK2_stride_h: K2 gradient tensor dimension stride
    :type dK2_stride_h: int
    :param dV2_stride_b: V2 gradient tensor batch stride
    :type dV2_stride_b: int
    :param dV2_stride_s: V2 gradient tensor sequence stride
    :type dV2_stride_s: int
    :param dV2_stride_k: V2 gradient tensor head stride
    :type dV2_stride_k: int
    :param dV2_stride_h: V2 gradient tensor dimension stride
    :type dV2_stride_h: int
    :param BLOCK_SIZE_Q: Tile size for query processing
    :type BLOCK_SIZE_Q: int
    :param BLOCK_SIZE_KV2: Tile size for K2/V2 processing (equals BLOCK_SIZE_Q + w2)
    :type BLOCK_SIZE_KV2: int
    :param HEAD_DIM: Head dimension (must match head_dim)
    :type HEAD_DIM: int
    :param SM_SCALE: Attention scaling factor (typically 1/√head_dim)
    :type SM_SCALE: float
    :param K2_BIAS: Additive bias for K2 tensor
    :type K2_BIAS: float
    :param V2_BIAS: Additive bias for V2 tensor
    :type V2_BIAS: float
    :param num_stages: Pipeline stages for memory optimization
    :type num_stages: int
    :param IS_SECOND_PASS: Whether this is the second pass (accumulates to existing gradients)
    :type IS_SECOND_PASS: bool
    :param DATA_DTYPE: Storage precision for input/output tensors
    :type DATA_DTYPE: tl.dtype
    :param COMPUTE_DTYPE: Computation precision for accumulation and arithmetic
    :type COMPUTE_DTYPE: tl.dtype
    :param GEMM_DTYPE: Matrix multiplication precision for performance
    :type GEMM_DTYPE: tl.dtype
    """
    # Constraint verification for correct memory access patterns
    assert BLOCK_SIZE_KV2 == BLOCK_SIZE_Q + w2

    # ============================================================================
    # THREAD BLOCK SETUP - TWO-PASS STRATEGY
    # ============================================================================
    # Implement two-pass strategy to handle overlapping memory accesses
    q_start = tl.program_id(0) * BLOCK_SIZE_KV2
    if IS_SECOND_PASS:
        q_start += BLOCK_SIZE_Q  # Offset for second pass
    q_end = q_start + BLOCK_SIZE_Q
    kv2_start = q_start - w2  # K2/V2 block starts w2 positions earlier

    # Decode batch and head indices from program_id(1)
    bk = tl.program_id(1)
    offs_b = bk // num_heads  # Batch index
    offs_k = bk % num_heads  # Head index

    # Calculate base memory offsets for current batch/head
    qkv_offs_bk = offs_b * q_stride_b + offs_k * q_stride_k
    Q_ptr += qkv_offs_bk
    K1_ptr += qkv_offs_bk
    K2_ptr += qkv_offs_bk
    V1_ptr += qkv_offs_bk
    V2_ptr += qkv_offs_bk

    dO_ptr += offs_b * dO_stride_b + offs_k * dO_stride_k
    M_ptr += offs_b * m_stride_b + offs_k * m_stride_k
    D_ptr += offs_b * d_stride_b + offs_k * d_stride_k
    dQ_ptr += offs_b * dq_stride_b + offs_k * dq_stride_k
    dK2_ptr += offs_b * dk2_stride_b + offs_k * dk2_stride_k
    dV2_ptr += offs_b * dv2_stride_b + offs_k * dv2_stride_k

    softmax_scale = tl.cast(SM_SCALE, GEMM_DTYPE)
    qkv_offs_h = tl.arange(0, HEAD_DIM)
    qkv_mask_h = qkv_offs_h < head_dim

    # Load ALiBi slope for current head (done once per kernel)
    if USE_ALIBI3D:
        slope = tl.load(SLOPES_ptr + offs_k * slopes_stride_h)

    # ============================================================================
    # LOAD Q/K2/V2 BLOCKS AND FORWARD PASS VALUES
    # ============================================================================
    # Set up memory offsets and masks for current blocks
    q_offs_s = q_start + tl.arange(0, BLOCK_SIZE_Q)
    kv2_offs_s = kv2_start + tl.arange(0, BLOCK_SIZE_KV2)
    q_offs = q_offs_s[:, None] * q_stride_s + qkv_offs_h[None, :] * q_stride_h
    kv2_offs = kv2_offs_s[:, None] * k2_stride_s + qkv_offs_h[None, :] * k2_stride_h
    m_offs = q_offs_s * m_stride_s
    d_offs = q_offs_s * d_stride_s
    dO_offs = q_offs_s[:, None] * dO_stride_s + qkv_offs_h[None, :] * dO_stride_h

    # Create masks for valid positions
    q_mask_s = q_offs_s < seq_len
    q_mask = q_mask_s[:, None] & qkv_mask_h[None, :]
    kv2_mask_s = (0 <= kv2_offs_s) & (kv2_offs_s < seq_len)
    kv2_mask = kv2_mask_s[:, None] & qkv_mask_h[None, :]

    # Load all required tensors for current blocks
    q_tile = tl.load(Q_ptr + q_offs, mask=q_mask).to(COMPUTE_DTYPE)  # [BLOCK_SIZE_Q, HEAD_DIM]
    k2_tile = tl.load(K2_ptr + kv2_offs, mask=kv2_mask).to(GEMM_DTYPE)  # [BLOCK_SIZE_KV2, HEAD_DIM]
    v2_tile = tl.load(V2_ptr + kv2_offs, mask=kv2_mask).to(GEMM_DTYPE)  # [BLOCK_SIZE_KV2, HEAD_DIM]
    m_tile = tl.load(M_ptr + m_offs, mask=q_mask_s).to(COMPUTE_DTYPE)  # [BLOCK_SIZE_Q]
    d_tile = tl.load(D_ptr + d_offs, mask=q_mask_s).to(COMPUTE_DTYPE)  # [BLOCK_SIZE_Q]
    dO_tile = tl.load(dO_ptr + dO_offs, mask=q_mask).to(GEMM_DTYPE)  # [BLOCK_SIZE_Q, HEAD_DIM]

    # Apply biases to K2 and V2
    k2_tile += K2_BIAS
    v2_tile += V2_BIAS
    k2_tile = k2_tile.to(GEMM_DTYPE)
    v2_tile = v2_tile.to(GEMM_DTYPE)

    # Initialize gradient accumulators
    dq = tl.zeros((BLOCK_SIZE_Q, HEAD_DIM), tl.float32)
    dk2 = tl.zeros((BLOCK_SIZE_KV2, HEAD_DIM), tl.float32)
    dv2 = tl.zeros((BLOCK_SIZE_KV2, HEAD_DIM), tl.float32)

    # ============================================================================
    # ITERATE OVER K1/V1 POSITIONS
    # ============================================================================
    # Determine valid K1/V1 range based on local attention windows
    kv1_start = tl.maximum(0, q_start - w1)
    kv1_end = tl.minimum(seq_len, q_end) if CAUSAL else tl.minimum(seq_len, q_end + w1)

    for kv1_idx in tl.range(kv1_start, kv1_end, num_stages=num_stages):
        # Load single K1 and V1 vectors
        k1_offs = kv1_idx * k1_stride_s + qkv_offs_h * k1_stride_h
        v1_offs = kv1_idx * v1_stride_s + qkv_offs_h * v1_stride_h
        k1_tile = tl.load(K1_ptr + k1_offs, mask=qkv_mask_h).to(COMPUTE_DTYPE)  # [HEAD_DIM]

        v1_tile = tl.load(V1_ptr + v1_offs, mask=qkv_mask_h).to(COMPUTE_DTYPE)  # [HEAD_DIM]

        # ====================================================================
        # COMPUTE ATTENTION SCORES AND PROBABILITIES
        # ====================================================================
        # Compute Q ⊙ K1 with scaling applied early for efficiency
        qk1_s = q_tile * (k1_tile[None, :] * softmax_scale)  # [BLOCK_SIZE_Q, HEAD_DIM]
        qk1_s = qk1_s.to(GEMM_DTYPE)

        # Compute attention scores: K2 @ (Q ⊙ K1)^T
        qkkT = tl.dot(k2_tile, qk1_s.T, input_precision="tf32", out_dtype=tl.float32)  # [BLOCK_SIZE_KV2, BLOCK_SIZE_Q]

        # --- RECOMPUTE ALiBi3D BIAS (same as forward) ---
        if USE_ALIBI3D:
            # Compute 3D distances: i=q_offs_s, j=kv1_idx, k=kv2_offs_s
            dist_ij = tl.abs(q_offs_s[None, :] - kv1_idx)
            dist_ik = tl.abs(q_offs_s[None, :] - kv2_offs_s[:, None])

            # Apply power scaling if α ≠ 1
            if ALIBI_ALPHA != 1.0:
                eps = 1e-7
                dist_ij = tl.exp(ALIBI_ALPHA * tl.log(dist_ij.to(tl.float32) + eps))
                dist_ik = tl.exp(ALIBI_ALPHA * tl.log(dist_ik.to(tl.float32) + eps))

            combined_dist = dist_ij + dist_ik
            alibi_bias = -slope * combined_dist
            qkkT += alibi_bias.to(qkkT.dtype)

        # Apply local attention window constraints
        qkT_mask = kv2_mask_s[:, None] & q_mask_s[None, :]
        if CAUSAL:
            kv1_local_mask = ((q_offs_s[None, :] - w1) < kv1_idx) & (
                kv1_idx <= q_offs_s[None, :]
            )
            kv2_local_mask = ((q_offs_s[None, :] - w2) < kv2_offs_s[:, None]) & (
                kv2_offs_s[:, None] <= q_offs_s[None, :]
            )
        else:
            kv1_local_mask = ((q_offs_s[None, :] - w1) < kv1_idx) & (
                kv1_idx < (q_offs_s[None, :] + w1)
            )
            kv2_local_mask = ((q_offs_s[None, :] - w2) < kv2_offs_s[:, None]) & (
                kv2_offs_s[:, None] < (q_offs_s[None, :] + w2)
            )
        local_mask = kv1_local_mask & kv2_local_mask  # [BLOCK_SIZE_KV2, BLOCK_SIZE_Q]
        qkT_mask &= kv1_local_mask & kv2_local_mask

        # Compute attention probabilities using saved max values
        pT = tl.exp(qkkT - m_tile[None, :])  # [BLOCK_SIZE_KV2, BLOCK_SIZE_Q]
        pT = tl.where(qkT_mask, pT, 0.0)

        # Mask invalid positions in scores (used later for gradient computation)
        qkkT = tl.where(local_mask, qkkT, -1.0e38)

        # ====================================================================
        # COMPUTE GRADIENTS USING CHAIN RULE
        # ====================================================================
        # Gradient w.r.t. V2: dV2 += P^T @ (dO ⊙ V1)
        dOv1 = dO_tile * v1_tile[None, :]  # [BLOCK_SIZE_Q, HEAD_DIM]
        dOv1 = dOv1.to(GEMM_DTYPE)
        dv2 += tl.dot(pT.to(GEMM_DTYPE), dOv1, input_precision="ieee", out_dtype=tl.float32)

        # Compute gradient w.r.t. pre-softmax scores
        # First: gradient w.r.t. attention output = V2^T @ (dO ⊙ V1)
        dpT = tl.dot(v2_tile, dOv1.T, input_precision="ieee", out_dtype=tl.float32)  # [BLOCK_SIZE_KV2, BLOCK_SIZE_Q]

        # Apply softmax backward: ds = P ⊙ (dp - d)
        dsT = pT * (dpT - d_tile[None, :])  # [BLOCK_SIZE_KV2, BLOCK_SIZE_Q]
        dsT = tl.where(qkT_mask, dsT, 0.0)
        dsT = dsT.to(GEMM_DTYPE)  # [BLOCK_SIZE_KV2, BLOCK_SIZE_Q]

        # Gradient w.r.t. K2: dK2 += ds @ (Q ⊙ K1)
        dk2 += tl.dot(dsT, qk1_s, input_precision="ieee", out_dtype=tl.float32)

        # Gradient w.r.t. Q: dQ += ds^T @ (K1 ⊙ K2) * scale
        k1k2 = k1_tile[None, :] * k2_tile  # [BLOCK_SIZE_KV2, HEAD_DIM]
        k1k2 = k1k2.to(GEMM_DTYPE)
        dq += tl.dot(dsT.T, k1k2, input_precision="ieee", out_dtype=tl.float32)  # Scale applied at the end

    # ============================================================================
    # HANDLE TWO-PASS ACCUMULATION AND STORE RESULTS
    # ============================================================================
    # In the second pass, accumulate to existing gradients
    if IS_SECOND_PASS:
        prev_dk2 = tl.load(dK2_ptr + kv2_offs, kv2_mask)
        prev_dv2 = tl.load(dV2_ptr + kv2_offs, kv2_mask)
        dk2 += prev_dk2
        dv2 += prev_dv2

    # Apply final scaling to query gradients
    dq *= softmax_scale

    # Store all computed gradients
    tl.store(dK2_ptr + kv2_offs, dk2.to(DATA_DTYPE), kv2_mask)
    tl.store(dV2_ptr + kv2_offs, dv2.to(DATA_DTYPE), kv2_mask)
    tl.store(dQ_ptr + q_offs, dq.to(DATA_DTYPE), q_mask)


# ===============================================
# TRITON KERNEL WRAPPER
# ===============================================


class TwoSimplicialAttentionFunction(torch.autograd.Function):
    """PyTorch autograd Function wrapper for Two Simplicial Attention with optional ALiBi3D.

    Implements the forward and backward passes for the Two Simplicial Attention mechanism
    from https://arxiv.org/abs/2507.02754 with our custom ALiBi3D positional bias extension.

    The core computation is:
        Attention(Q, K1, K2, V1, V2) = softmax((Q ⊙ K1) @ K2^T / √d + B_ALiBi3D) @ (V1 ⊙ V2)

    Where B_ALiBi3D is an optional 3D positional bias that encourages local attention patterns.
    """

    BLOCK_SIZE_Q = 32

    @staticmethod
    def forward(
        ctx,
        q,
        k1,
        k2,
        v1,
        v2,
        w1,
        w2,
        alibi_slopes: Optional[torch.Tensor] = None,
        alibi_alpha: Optional[float] = None,
        k2_bias=0.0,
        v2_bias=0.0,
        sm_scale=None,
        prescale: bool = False,
        causal: bool = False,
        data_dtype=tl.bfloat16,
        compute_dtype=tl.float32,
        gemm_dtype=tl.bfloat16,
    ):
        """Forward pass through Two Simplicial Attention.

        :param q: Query tensor with shape [batch_size, seq_len, num_heads, head_dim]
        :type q: torch.Tensor
        :param k1: First key tensor with shape [batch_size, seq_len, num_heads, head_dim]
        :type k1: torch.Tensor
        :param k2: Second key tensor with shape [batch_size, seq_len, num_heads, head_dim]
        :type k2: torch.Tensor
        :param v1: First value tensor with shape [batch_size, seq_len, num_heads, head_dim]
        :type v1: torch.Tensor
        :param v2: Second value tensor with shape [batch_size, seq_len, num_heads, head_dim]
        :type v2: torch.Tensor
        :param w1: Local attention window size for K1/V1
        :type w1: int
        :param w2: Local attention window size for K2/V2
        :type w2: int
        :param alibi_slopes: ALiBi3D slope parameters [num_heads]. If None, no positional bias is applied.
        :type alibi_slopes: torch.Tensor, optional
        :param alibi_alpha: Power factor for distance computation in ALiBi3D bias
        :type alibi_alpha: float
        :param k2_bias: Additive bias applied to K2 tensor
        :type k2_bias: float
        :param v2_bias: Additive bias applied to V2 tensor
        :type v2_bias: float
        :param sm_scale: Attention scaling factor (only used if prescale=False), defaults to None
        :type sm_scale: float, optional
        :param prescale: If True, pre-scale Q, K1, K2 by d**(-1/3) instead of applying 1/√d at logits
        :type prescale: bool
        :param causal: If True, apply causal masking (j <= i, k <= i). Default False.
        :type causal: bool
        :param data_dtype: Triton data type for storage operations
        :type data_dtype: tl.dtype
        :param compute_dtype: Triton data type for computation operations
        :type compute_dtype: tl.dtype
        :param gemm_dtype: Triton data type for GEMM operations
        :type gemm_dtype: tl.dtype
        :return: Attention output with same shape as input tensors
        :rtype: torch.Tensor
        """
        batch_size, seq_len, num_heads, head_dim = q.shape
        device = q.device
        if not device.type == "cuda":
            raise RuntimeError("TwoSimplicialAttention requires CUDA tensors.")

        w1 = w1 if w1 is not None else seq_len
        w2 = w2 if w2 is not None else seq_len

        # Scaling
        if prescale:
            scale_factor = head_dim**-0.1666666667  # d^(-1/6)
            q = q * scale_factor
            k1 = k1 * scale_factor
            k2 = k2 * scale_factor
            sm_scale = 1.0  # already normalized
        else:
            if sm_scale is None:
                sm_scale = head_dim**-0.5  # d^(-1/2)

        # Ensure contiguous
        q, k1, k2, v1, v2 = [t.contiguous() for t in (q, k1, k2, v1, v2)]

        # Handle ALiBi3D slopes
        use_alibi3d = alibi_slopes is not None
        if use_alibi3d:
            alibi_slopes = alibi_slopes.contiguous()
        else:
            alibi_slopes = torch.zeros(num_heads, dtype=q.dtype, device=device)

        # Output + scratch
        output = torch.empty_like(q)
        m = torch.full((batch_size, num_heads, seq_len), -float("inf"), dtype=torch.float32, device=device)

        # Strides
        q_s, k1_s, k2_s, v1_s, v2_s, o_s, m_s = [t.stride() for t in (q, k1, k2, v1, v2, output, m)]

        def grid(meta):
            return (
                triton.cdiv(seq_len, meta["BLOCK_SIZE_Q"]),
                batch_size * num_heads,
            )

        # Kernel call
        two_simplicial_attn_fwd_kernel[grid](
            Q_ptr=q,
            K1_ptr=k1,
            K2_ptr=k2,
            V1_ptr=v1,
            V2_ptr=v2,
            O_ptr=output,
            M_ptr=m,
            SLOPES_ptr=alibi_slopes,
            bs=batch_size,
            seq_len=seq_len,
            num_heads=num_heads,
            head_dim=head_dim,
            w1=w1,
            w2=w2,
            q_stride_b=q_s[0],
            q_stride_s=q_s[1],
            q_stride_k=q_s[2],
            q_stride_h=q_s[3],
            k1_stride_b=k1_s[0],
            k1_stride_s=k1_s[1],
            k1_stride_k=k1_s[2],
            k1_stride_h=k1_s[3],
            k2_stride_b=k2_s[0],
            k2_stride_s=k2_s[1],
            k2_stride_k=k2_s[2],
            k2_stride_h=k2_s[3],
            v1_stride_b=v1_s[0],
            v1_stride_s=v1_s[1],
            v1_stride_k=v1_s[2],
            v1_stride_h=v1_s[3],
            v2_stride_b=v2_s[0],
            v2_stride_s=v2_s[1],
            v2_stride_k=v2_s[2],
            v2_stride_h=v2_s[3],
            out_stride_b=o_s[0],
            out_stride_s=o_s[1],
            out_stride_k=o_s[2],
            out_stride_h=o_s[3],
            m_stride_b=m_s[0],
            m_stride_k=m_s[1],
            m_stride_s=m_s[2],
            slopes_stride_h=alibi_slopes.stride(0),
            HEAD_DIM=head_dim,
            SM_SCALE=sm_scale,
            K2_BIAS=k2_bias,
            V2_BIAS=v2_bias,
            USE_ALIBI3D=use_alibi3d,
            ALIBI_ALPHA=alibi_alpha,
            CAUSAL=causal,
            DATA_DTYPE=data_dtype,
            COMPUTE_DTYPE=compute_dtype,
            GEMM_DTYPE=gemm_dtype,
        )

        # Save for backward (including dtype parameters)
        ctx.save_for_backward(q, k1, k2, v1, v2, output, m, alibi_slopes)
        ctx.w1, ctx.w2 = w1, w2
        ctx.k2_bias, ctx.v2_bias, ctx.sm_scale = k2_bias, v2_bias, sm_scale
        ctx.batch_size, ctx.seq_len, ctx.num_heads, ctx.head_dim = batch_size, seq_len, num_heads, head_dim
        ctx.use_alibi3d, ctx.prescale = use_alibi3d, prescale
        ctx.causal = causal
        ctx.alibi_alpha = alibi_alpha if alibi_alpha is not None else 1.0
        # Save dtype parameters for backward pass
        ctx.data_dtype = data_dtype
        ctx.compute_dtype = compute_dtype
        ctx.gemm_dtype = gemm_dtype

        return output

    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass for Two Simplicial Attention.

        Computes gradients for all input tensors using a three-kernel approach:
        1. K1/V1-centric kernel for ∂L/∂K1 and ∂L/∂V1
        2. Two-pass K2/V2/Q-centric kernel for ∂L/∂K2, ∂L/∂V2, and ∂L/∂Q

        The approach splits gradient computation to maximize memory efficiency
        and handle overlapping access patterns in local attention windows.
        """

        # Retrieve saved tensors from forward pass
        q, k1, k2, v1, v2, output, m, alibi_slopes = ctx.saved_tensors

        # Retrieve saved dtype parameters
        data_dtype = ctx.data_dtype
        compute_dtype = ctx.compute_dtype
        gemm_dtype = ctx.gemm_dtype

        # Ensure contiguous grad_output on correct device
        grad_output = grad_output.contiguous()
        device = grad_output.device

        if not device.type == "cuda":
            raise RuntimeError(
                f"TwoSimplicialAttention requires CUDA tensors for backward pass. Got grad_output on device: {device}"
            )

        # Initialize gradient tensors on GPU
        grad_q = torch.zeros_like(q)
        grad_k1 = torch.zeros_like(k1)
        grad_k2 = torch.zeros_like(k2)
        grad_v1 = torch.zeros_like(v1)
        grad_v2 = torch.zeros_like(v2)

        # Compute diagonal values for softmax backward pass
        # D[i] = Σ_j P[i,j] * ∂L/∂output[i,j] (needed for softmax gradient)
        d = torch.sum(grad_output * output, dim=-1)  # [B, S, H]
        d = d.permute(0, 2, 1).contiguous()  # Reorder to [B, H, S] to match M layout

        # Extract memory strides for all tensors
        q_strides = q.stride()
        k1_strides = k1.stride()
        k2_strides = k2.stride()
        v1_strides = v1.stride()
        v2_strides = v2.stride()
        do_strides = grad_output.stride()
        dq_strides = grad_q.stride()
        dk1_strides = grad_k1.stride()
        dk2_strides = grad_k2.stride()
        dv1_strides = grad_v1.stride()
        dv2_strides = grad_v2.stride()
        m_strides = m.stride()
        d_strides = d.stride()

        # ====================================================================
        # KERNEL 1: K1/V1-CENTRIC BACKWARD PASS
        # ====================================================================
        # Compute gradients for K1 and V1 by processing blocks of K1/V1 positions
        # Each thread block handles BLOCK_SIZE_KV K1/V1 positions across all relevant K2/Q pairs

        def grid_kv1(meta):
            return (
                triton.cdiv(ctx.seq_len, meta["BLOCK_SIZE_KV"]),  # K1/V1 position blocks
                ctx.batch_size * ctx.num_heads,  # Batch * head combinations
            )

        two_simplicial_attn_bwd_kv1_kernel[grid_kv1](
            # Input tensors
            Q_ptr=q,
            K1_ptr=k1,
            K2_ptr=k2,
            V1_ptr=v1,
            V2_ptr=v2,
            dO_ptr=grad_output,
            M_ptr=m,
            D_ptr=d,
            SLOPES_ptr=alibi_slopes,
            # Output gradient tensors
            dQ_ptr=grad_q,
            dK1_ptr=grad_k1,
            dV1_ptr=grad_v1,
            # Shape parameters
            bs=ctx.batch_size,
            seq_len=ctx.seq_len,
            num_heads=ctx.num_heads,
            head_dim=ctx.head_dim,
            w1=ctx.w1,
            w2=ctx.w2,
            # Stride parameters
            q_stride_b=q_strides[0],
            q_stride_s=q_strides[1],
            q_stride_k=q_strides[2],
            q_stride_h=q_strides[3],
            k1_stride_b=k1_strides[0],
            k1_stride_s=k1_strides[1],
            k1_stride_k=k1_strides[2],
            k1_stride_h=k1_strides[3],
            k2_stride_b=k2_strides[0],
            k2_stride_s=k2_strides[1],
            k2_stride_k=k2_strides[2],
            k2_stride_h=k2_strides[3],
            v1_stride_b=v1_strides[0],
            v1_stride_s=v1_strides[1],
            v1_stride_k=v1_strides[2],
            v1_stride_h=v1_strides[3],
            v2_stride_b=v2_strides[0],
            v2_stride_s=v2_strides[1],
            v2_stride_k=v2_strides[2],
            v2_stride_h=v2_strides[3],
            dO_stride_b=do_strides[0],
            dO_stride_s=do_strides[1],
            dO_stride_k=do_strides[2],
            dO_stride_h=do_strides[3],
            m_stride_b=m_strides[0],
            m_stride_k=m_strides[1],
            m_stride_s=m_strides[2],
            d_stride_b=d_strides[0],
            d_stride_k=d_strides[1],
            d_stride_s=d_strides[2],
            slopes_stride_h=alibi_slopes.stride(0),
            dq_stride_b=dq_strides[0],
            dq_stride_s=dq_strides[1],
            dq_stride_k=dq_strides[2],
            dq_stride_h=dq_strides[3],
            dk1_stride_b=dk1_strides[0],
            dk1_stride_s=dk1_strides[1],
            dk1_stride_k=dk1_strides[2],
            dk1_stride_h=dk1_strides[3],
            dv1_stride_b=dv1_strides[0],
            dv1_stride_s=dv1_strides[1],
            dv1_stride_k=dv1_strides[2],
            dv1_stride_h=dv1_strides[3],
            # Compile-time constants
            BLOCK_SIZE_Q=32,
            BLOCK_SIZE_KV=64,
            HEAD_DIM=ctx.head_dim,
            SM_SCALE=ctx.sm_scale,
            K2_BIAS=ctx.k2_bias,
            V2_BIAS=ctx.v2_bias,
            USE_ALIBI3D=ctx.use_alibi3d,
            ALIBI_ALPHA=ctx.alibi_alpha,
            CAUSAL=ctx.causal,
            COMPUTE_DQ=False,
            is_flipped=False,
            DATA_DTYPE=data_dtype,
            COMPUTE_DTYPE=compute_dtype,
            GEMM_DTYPE=gemm_dtype,
        )

        # ====================================================================
        # KERNEL 2: K2/V2/Q-CENTRIC BACKWARD PASS (TWO PASSES)
        # ====================================================================
        # Compute gradients for K2, V2, and Q using a two-pass strategy to handle
        # overlapping memory accesses due to local attention windows

        block_size_kv2 = (
            TwoSimplicialAttentionFunction.BLOCK_SIZE_Q + ctx.w2
        )  # BLOCK_SIZE_Q + w2 (ensures all relevant K2 positions are loaded)

        def grid_kv2q(meta):
            return (
                triton.cdiv(ctx.seq_len, block_size_kv2),
                ctx.batch_size * ctx.num_heads,
            )

        # First pass: Process "even" tiles
        two_simplicial_attn_bwd_kv2q_kernel[grid_kv2q](
            # Input tensors
            Q_ptr=q,
            K1_ptr=k1,
            K2_ptr=k2,
            V1_ptr=v1,
            V2_ptr=v2,
            dO_ptr=grad_output,
            M_ptr=m,
            D_ptr=d,
            SLOPES_ptr=alibi_slopes,
            # Output gradient tensors
            dQ_ptr=grad_q,
            dK2_ptr=grad_k2,
            dV2_ptr=grad_v2,
            # Shape parameters
            bs=ctx.batch_size,
            seq_len=ctx.seq_len,
            num_heads=ctx.num_heads,
            head_dim=ctx.head_dim,
            w1=ctx.w1,
            w2=ctx.w2,
            # Stride parameters
            q_stride_b=q_strides[0],
            q_stride_s=q_strides[1],
            q_stride_k=q_strides[2],
            q_stride_h=q_strides[3],
            k1_stride_b=k1_strides[0],
            k1_stride_s=k1_strides[1],
            k1_stride_k=k1_strides[2],
            k1_stride_h=k1_strides[3],
            k2_stride_b=k2_strides[0],
            k2_stride_s=k2_strides[1],
            k2_stride_k=k2_strides[2],
            k2_stride_h=k2_strides[3],
            v1_stride_b=v1_strides[0],
            v1_stride_s=v1_strides[1],
            v1_stride_k=v1_strides[2],
            v1_stride_h=v1_strides[3],
            v2_stride_b=v2_strides[0],
            v2_stride_s=v2_strides[1],
            v2_stride_k=v2_strides[2],
            v2_stride_h=v2_strides[3],
            dO_stride_b=do_strides[0],
            dO_stride_s=do_strides[1],
            dO_stride_k=do_strides[2],
            dO_stride_h=do_strides[3],
            m_stride_b=m_strides[0],
            m_stride_k=m_strides[1],
            m_stride_s=m_strides[2],
            d_stride_b=d_strides[0],
            d_stride_k=d_strides[1],
            d_stride_s=d_strides[2],
            slopes_stride_h=alibi_slopes.stride(0),
            dq_stride_b=dq_strides[0],
            dq_stride_s=dq_strides[1],
            dq_stride_k=dq_strides[2],
            dq_stride_h=dq_strides[3],
            dk2_stride_b=dk2_strides[0],
            dk2_stride_s=dk2_strides[1],
            dk2_stride_k=dk2_strides[2],
            dk2_stride_h=dk2_strides[3],
            dv2_stride_b=dv2_strides[0],
            dv2_stride_s=dv2_strides[1],
            dv2_stride_k=dv2_strides[2],
            dv2_stride_h=dv2_strides[3],
            # Compile-time constants
            HEAD_DIM=ctx.head_dim,
            SM_SCALE=ctx.sm_scale,
            K2_BIAS=ctx.k2_bias,
            V2_BIAS=ctx.v2_bias,
            USE_ALIBI3D=ctx.use_alibi3d,
            ALIBI_ALPHA=ctx.alibi_alpha,
            CAUSAL=ctx.causal,
            IS_SECOND_PASS=False,
            DATA_DTYPE=data_dtype,
            COMPUTE_DTYPE=compute_dtype,
            GEMM_DTYPE=gemm_dtype,
        )

        # Second pass: Process "odd" tiles and accumulate with first pass results
        two_simplicial_attn_bwd_kv2q_kernel[grid_kv2q](
            # Input tensors
            Q_ptr=q,
            K1_ptr=k1,
            K2_ptr=k2,
            V1_ptr=v1,
            V2_ptr=v2,
            dO_ptr=grad_output,
            M_ptr=m,
            D_ptr=d,
            SLOPES_ptr=alibi_slopes,
            # Output gradient tensors
            dQ_ptr=grad_q,
            dK2_ptr=grad_k2,
            dV2_ptr=grad_v2,
            # Shape parameters
            bs=ctx.batch_size,
            seq_len=ctx.seq_len,
            num_heads=ctx.num_heads,
            head_dim=ctx.head_dim,
            w1=ctx.w1,
            w2=ctx.w2,
            # Stride parameters
            q_stride_b=q_strides[0],
            q_stride_s=q_strides[1],
            q_stride_k=q_strides[2],
            q_stride_h=q_strides[3],
            k1_stride_b=k1_strides[0],
            k1_stride_s=k1_strides[1],
            k1_stride_k=k1_strides[2],
            k1_stride_h=k1_strides[3],
            k2_stride_b=k2_strides[0],
            k2_stride_s=k2_strides[1],
            k2_stride_k=k2_strides[2],
            k2_stride_h=k2_strides[3],
            v1_stride_b=v1_strides[0],
            v1_stride_s=v1_strides[1],
            v1_stride_k=v1_strides[2],
            v1_stride_h=v1_strides[3],
            v2_stride_b=v2_strides[0],
            v2_stride_s=v2_strides[1],
            v2_stride_k=v2_strides[2],
            v2_stride_h=v2_strides[3],
            dO_stride_b=do_strides[0],
            dO_stride_s=do_strides[1],
            dO_stride_k=do_strides[2],
            dO_stride_h=do_strides[3],
            m_stride_b=m_strides[0],
            m_stride_k=m_strides[1],
            m_stride_s=m_strides[2],
            d_stride_b=d_strides[0],
            d_stride_k=d_strides[1],
            d_stride_s=d_strides[2],
            slopes_stride_h=alibi_slopes.stride(0),
            dq_stride_b=dq_strides[0],
            dq_stride_s=dq_strides[1],
            dq_stride_k=dq_strides[2],
            dq_stride_h=dq_strides[3],
            dk2_stride_b=dk2_strides[0],
            dk2_stride_s=dk2_strides[1],
            dk2_stride_k=dk2_strides[2],
            dk2_stride_h=dk2_strides[3],
            dv2_stride_b=dv2_strides[0],
            dv2_stride_s=dv2_strides[1],
            dv2_stride_k=dv2_strides[2],
            dv2_stride_h=dv2_strides[3],
            # Compile-time constants
            HEAD_DIM=ctx.head_dim,
            SM_SCALE=ctx.sm_scale,
            K2_BIAS=ctx.k2_bias,
            V2_BIAS=ctx.v2_bias,
            USE_ALIBI3D=ctx.use_alibi3d,
            ALIBI_ALPHA=ctx.alibi_alpha,
            CAUSAL=ctx.causal,
            IS_SECOND_PASS=True,
            DATA_DTYPE=data_dtype,
            COMPUTE_DTYPE=compute_dtype,
            GEMM_DTYPE=gemm_dtype,
        )

        # ====================================================================
        # APPLY PRESCALE GRADIENT ADJUSTMENT
        # ====================================================================
        # If prescaling was applied in forward pass, we need to scale gradients
        # by the same factor to get gradients w.r.t. original inputs (chain rule)
        # Forward: q_original * scale_factor = q_prescaled
        # Backward: grad_q_prescaled * scale_factor = grad_q_original
        if ctx.prescale:
            scale_factor = ctx.head_dim ** -0.1666666667  # d^(-1/6)
            
            # Scale gradients for Q, K1, K2 (which were prescaled in forward)
            grad_q = grad_q * scale_factor
            grad_k1 = grad_k1 * scale_factor
            grad_k2 = grad_k2 * scale_factor
            # Note: V1 and V2 were not prescaled, so no adjustment needed

        # Return gradients (None for non-tensor parameters)
        # Order: q, k1, k2, v1, v2, w1, w2, alibi_slopes, alibi_alpha,
        #        k2_bias, v2_bias, sm_scale, prescale, causal,
        #        data_dtype, compute_dtype, gemm_dtype
        return (
            grad_q,
            grad_k1,
            grad_k2,
            grad_v1,
            grad_v2,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )

