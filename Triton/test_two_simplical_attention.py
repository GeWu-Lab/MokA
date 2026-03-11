"""Unit tests for two_simplicial_attention_pytorch."""

import torch
import pytest

from two_simplicial_attention_pytorch import two_simplicial_attention_pytorch


# ── Naive reference (triple loop, no tricks) ─────────────────────────────────

def _naive_two_simplicial(q, k1, k2, v1, v2, w1, w2, k2_bias, v2_bias, sm_scale):
    """Dead-simple loop implementation for correctness checking."""
    B, S, H, D = q.shape
    k2 = k2 + k2_bias
    v2 = v2 + v2_bias
    out = torch.zeros_like(q, dtype=torch.float32)

    for b in range(B):
        for h in range(H):
            for i in range(S):
                logits = []
                vals = []
                for j in range(S):
                    if abs(i - j) >= w1:
                        continue
                    for k in range(S):
                        if abs(i - k) >= w2:
                            continue
                        # score = (q[i] * k1[j]) . k2[k] * scale
                        score = (q[b, i, h] * k1[b, j, h] * k2[b, k, h]).sum() * sm_scale
                        logits.append(score)
                        vals.append(v1[b, j, h] * v2[b, k, h])  # [D]

                if len(logits) == 0:
                    continue

                logits_t = torch.stack(logits)  # [N]
                vals_t = torch.stack(vals)       # [N, D]
                attn = torch.softmax(logits_t.float(), dim=0)
                out[b, i, h] = (attn.unsqueeze(-1) * vals_t.float()).sum(0)

    return out.to(q.dtype)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return request.param


def _make_inputs(B, S, H, D, device, dtype=torch.float32):
    torch.manual_seed(42)
    tensors = [torch.randn(B, S, H, D, device=device, dtype=dtype) for _ in range(5)]
    return tensors  # q, k1, k2, v1, v2


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestCorrectnessVsNaive:
    """Compare vectorized einsum implementation against naive triple loop."""

    def test_full_attention(self, device):
        B, S, H, D = 1, 16, 2, 8
        q, k1, k2, v1, v2 = _make_inputs(B, S, H, D, device)
        sm_scale = D ** -0.5

        out = two_simplicial_attention_pytorch(q, k1, k2, v1, v2)
        ref = _naive_two_simplicial(q, k1, k2, v1, v2, S, S, 0.0, 0.0, sm_scale)

        torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)

    def test_with_window(self, device):
        B, S, H, D = 1, 16, 2, 8
        q, k1, k2, v1, v2 = _make_inputs(B, S, H, D, device)
        w1, w2 = 4, 6
        sm_scale = D ** -0.5

        out = two_simplicial_attention_pytorch(q, k1, k2, v1, v2, w1=w1, w2=w2)
        ref = _naive_two_simplicial(q, k1, k2, v1, v2, w1, w2, 0.0, 0.0, sm_scale)

        torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)

    def test_with_biases(self, device):
        B, S, H, D = 1, 16, 2, 8
        q, k1, k2, v1, v2 = _make_inputs(B, S, H, D, device)
        sm_scale = D ** -0.5

        out = two_simplicial_attention_pytorch(
            q, k1, k2, v1, v2, k2_bias=0.5, v2_bias=-0.3,
        )
        ref = _naive_two_simplicial(q, k1, k2, v1, v2, S, S, 0.5, -0.3, sm_scale)

        torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)

    def test_prescale(self, device):
        B, S, H, D = 1, 16, 2, 8
        q, k1, k2, v1, v2 = _make_inputs(B, S, H, D, device)
        sf = D ** (-1.0 / 6.0)

        out = two_simplicial_attention_pytorch(q, k1, k2, v1, v2, prescale=True)
        ref = _naive_two_simplicial(
            q * sf, k1 * sf, k2 * sf, v1, v2, S, S, 0.0, 0.0, 1.0,
        )

        torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)


class TestShape:
    def test_output_shape(self, device):
        B, S, H, D = 2, 32, 4, 16
        q, k1, k2, v1, v2 = _make_inputs(B, S, H, D, device)
        out = two_simplicial_attention_pytorch(q, k1, k2, v1, v2)
        assert out.shape == (B, S, H, D)

    def test_output_dtype_matches_input(self, device):
        B, S, H, D = 1, 16, 2, 8
        q, k1, k2, v1, v2 = _make_inputs(B, S, H, D, device)
        out = two_simplicial_attention_pytorch(q, k1, k2, v1, v2)
        assert out.dtype == q.dtype


class TestGradients:
    def test_all_inputs_get_gradients(self, device):
        B, S, H, D = 1, 12, 2, 8
        inputs = [t.requires_grad_(True) for t in _make_inputs(B, S, H, D, device)]
        out = two_simplicial_attention_pytorch(*inputs)
        out.sum().backward()

        for name, t in zip(["q", "k1", "k2", "v1", "v2"], inputs):
            assert t.grad is not None, f"No gradient for {name}"
            assert t.grad.shape == t.shape, f"Wrong grad shape for {name}"
            assert t.grad.isfinite().all(), f"Non-finite gradient for {name}"


class TestWindowMasking:
    def test_window_changes_output(self, device):
        """Windowed attention should differ from full attention."""
        B, S, H, D = 1, 32, 2, 8
        q, k1, k2, v1, v2 = _make_inputs(B, S, H, D, device)

        out_full = two_simplicial_attention_pytorch(q, k1, k2, v1, v2)
        out_win = two_simplicial_attention_pytorch(q, k1, k2, v1, v2, w1=4, w2=4)

        assert not torch.allclose(out_full, out_win, atol=1e-5), \
            "Windowed output should differ from full attention"

    def test_window_equals_full_when_large(self, device):
        """Window >= seq_len should give same result as full attention."""
        B, S, H, D = 1, 16, 2, 8
        q, k1, k2, v1, v2 = _make_inputs(B, S, H, D, device)

        out_full = two_simplicial_attention_pytorch(q, k1, k2, v1, v2)
        out_win = two_simplicial_attention_pytorch(q, k1, k2, v1, v2, w1=S, w2=S)

        torch.testing.assert_close(out_full, out_win, atol=1e-6, rtol=1e-6)


class TestAttentionProperties:
    def test_uniform_inputs_give_uniform_attention(self, device):
        """When Q=K1=K2=constant, all positions should get ~same output."""
        B, S, H, D = 1, 16, 2, 8
        const = torch.ones(B, S, H, D, device=device)
        v1 = torch.randn(B, S, H, D, device=device)
        v2 = torch.ones(B, S, H, D, device=device)

        out = two_simplicial_attention_pytorch(const, const, const, v1, v2)

        # Each position should see identical attention weights,
        # so output[i] = mean(v1) * 1 for all i
        expected = v1.mean(dim=1, keepdim=True).expand_as(out)
        torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-5)
