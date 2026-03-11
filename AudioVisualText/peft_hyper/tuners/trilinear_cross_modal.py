"""
Trilinear Cross-Modal LoRA Attention.

Implements the two-simplicial attention where TEXT tokens attend to (VIDEO, AUDIO) pairs.
Video and audio tokens pass through unchanged; only text is updated.

Formula:
    score[i,j,k] = (text_i ⊙ video_j)^T audio_k / sqrt(r)
    weights[i,j,k] = softmax over (j,k) of score[i,j,k]
    text_updated[i] = text[i] + alpha * sum_{j,k} weights[i,j,k] * (video_j ⊙ audio_k)
"""

import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
triton_dir = Path(__file__).resolve().parents[3] / "Triton"
if str(triton_dir) not in sys.path:
    sys.path.insert(0, str(triton_dir))

from two_simplical_attention import TwoSimplicialAttentionFunction


def trilinear_text_update(
    text: torch.Tensor,
    video: torch.Tensor,
    audio: torch.Tensor,
    alpha: float = 1.0,
    scale: Optional[float] = None,
    prescale: bool = True,
) -> torch.Tensor:
    """Update text tokens with the Triton two-simplicial attention kernel."""

    if text.device.type != "cuda":
        raise RuntimeError("trilinear_text_update requires CUDA because it uses the Triton kernel directly.")

    rank = text.size(-1)
    if scale is None:
        scale = rank ** -0.5

    # Triton's tl.dot path requires the feature dimension to be >= 16. (check if this is how we should handle this)
    # Current solution: Zero-padding preserves the original trilinear scores and values.
    padded_rank = max(rank, 16)
    if padded_rank != rank:
        pad = (0, padded_rank - rank) # zero-pad the feature dimension to 16
        text = F.pad(text, pad)
        video = F.pad(video, pad)
        audio = F.pad(audio, pad) # zero-pad the feature dimension to 16

    q = text.unsqueeze(2).contiguous()
    k1 = video.unsqueeze(2).contiguous()
    k2 = audio.unsqueeze(2).contiguous()
    v1 = video.unsqueeze(2).contiguous()
    v2 = audio.unsqueeze(2).contiguous()
    seq_len = q.size(1)

    out = TwoSimplicialAttentionFunction.apply(
        q,
        k1,
        k2,
        v1,
        v2,
        seq_len,
        seq_len,
        None,
        None,
        0.0,
        0.0,
        scale,
        prescale,
        False,
    )
    text_updated = text + alpha * out[:, :, 0, :]
    return text_updated[..., :rank]

# Currently using this function to update the text tokens. Takes ~16 hrs to train compared to 34 hrs for the original function.
def trilinear_text_update_packed(
    text: torch.Tensor,
    video: torch.Tensor,
    audio: torch.Tensor,
    text_mask: torch.Tensor,
    video_mask: torch.Tensor,
    audio_mask: torch.Tensor,
    alpha: float = 1.0,
    scale: Optional[float] = None,
    prescale: bool = True,
) -> torch.Tensor:
    """Run trilinear attention on compacted active tokens, then scatter back."""

    if text.device.type != "cuda":
        raise RuntimeError("trilinear_text_update_packed requires CUDA because it uses the Triton kernel directly.")

    updated = text.clone()
    batch_size = text.size(0)

    for batch_idx in range(batch_size):
        text_idx = torch.where(text_mask[batch_idx].squeeze(-1) > 0)[0]
        video_idx = torch.where(video_mask[batch_idx].squeeze(-1) > 0)[0]
        audio_idx = torch.where(audio_mask[batch_idx].squeeze(-1) > 0)[0]

        if text_idx.numel() == 0 or video_idx.numel() == 0 or audio_idx.numel() == 0:
            continue

        packed_len = max(int(text_idx.numel()), int(video_idx.numel()), int(audio_idx.numel()))

        packed_text = text.new_zeros((1, packed_len, text.size(-1)))
        packed_video = video.new_zeros((1, packed_len, video.size(-1)))
        packed_audio = audio.new_zeros((1, packed_len, audio.size(-1)))

        packed_text[:, : text_idx.numel(), :] = text[batch_idx : batch_idx + 1].index_select(1, text_idx)
        packed_video[:, : video_idx.numel(), :] = video[batch_idx : batch_idx + 1].index_select(1, video_idx)
        packed_audio[:, : audio_idx.numel(), :] = audio[batch_idx : batch_idx + 1].index_select(1, audio_idx)

        packed_updated = trilinear_text_update(
            text=packed_text,
            video=packed_video,
            audio=packed_audio,
            alpha=alpha,
            scale=scale,
            prescale=prescale,
        )
        updated[batch_idx, text_idx, :] = packed_updated[0, : text_idx.numel(), :]

    return updated
