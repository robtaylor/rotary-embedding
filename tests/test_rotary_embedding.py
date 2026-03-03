"""Tests for Metal rotary embedding kernel.

Validates correctness against a pure-PyTorch reference implementation
for both NeoX (Llama/Mistral) and GPT-J rotation styles.
"""

import pytest
import torch

import rotary_embedding as ops


def _is_mps_available() -> bool:
    return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()


if _is_mps_available():
    DEVICES = ["mps"]
else:
    DEVICES = [f"cuda:{i}" for i in range(max(1, torch.cuda.device_count()))]

DTYPES = [torch.float32, torch.float16, torch.bfloat16]
HEAD_SIZES = [64, 128, 256]
NUM_HEADS = [8, 32]
NUM_KV_HEADS = [1, 8]  # GQA and MHA
IS_NEOX = [True, False]
NUM_TOKENS = [1, 7, 32]
MAX_POSITION = 8192
ROTARY_DIM_FRACTIONS = [1.0]  # Full rotation; 0.5 for partial


def _build_cos_sin_cache(
    max_position: int,
    rotary_dim: int,
    dtype: torch.dtype,
    device: str,
    base: float = 10000.0,
) -> torch.Tensor:
    """Build a cos/sin cache matching vLLM's convention."""
    inv_freq = 1.0 / (
        base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim)
    )
    t = torch.arange(max_position, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)  # [max_position, rotary_dim/2]
    cos_vals = freqs.cos()
    sin_vals = freqs.sin()
    cache = torch.cat([cos_vals, sin_vals], dim=-1)  # [max_position, rotary_dim]
    return cache.to(dtype=dtype, device=device)


def _ref_rotary_embedding(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor | None,
    head_size: int,
    cos_sin_cache: torch.Tensor,
    is_neox: bool,
) -> None:
    """Pure-PyTorch reference implementation of rotary embedding.

    Uses Python scalars (.item()) rather than 0-dim tensor operations
    to avoid a PyTorch bug where .float() on float32 tensors returns the
    same view, causing aliasing issues in tight in-place update loops.
    """
    rot_dim = cos_sin_cache.shape[-1]
    embed_dim = rot_dim // 2

    num_tokens = positions.numel()
    positions_flat = positions.reshape(-1)

    for t in range(num_tokens):
        pos = positions_flat[t].item()

        # Apply to query heads.
        num_heads = query.shape[-2]
        for h in range(num_heads):
            for d in range(embed_dim):
                if is_neox:
                    x_idx, y_idx = d, embed_dim + d
                else:
                    x_idx, y_idx = 2 * d, 2 * d + 1

                x = query[t, h, x_idx].item()
                y = query[t, h, y_idx].item()
                c = cos_sin_cache[pos, d].item()
                s = cos_sin_cache[pos, embed_dim + d].item()
                query[t, h, x_idx] = x * c - y * s
                query[t, h, y_idx] = y * c + x * s

        # Apply to key heads.
        if key is not None:
            num_kv_heads = key.shape[-2]
            for h in range(num_kv_heads):
                for d in range(embed_dim):
                    if is_neox:
                        x_idx, y_idx = d, embed_dim + d
                    else:
                        x_idx, y_idx = 2 * d, 2 * d + 1

                    x = key[t, h, x_idx].item()
                    y = key[t, h, y_idx].item()
                    c = cos_sin_cache[pos, d].item()
                    s = cos_sin_cache[pos, embed_dim + d].item()
                    key[t, h, x_idx] = x * c - y * s
                    key[t, h, y_idx] = y * c + x * s


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("num_kv_heads", NUM_KV_HEADS)
@pytest.mark.parametrize("is_neox", IS_NEOX)
@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@torch.inference_mode()
def test_rotary_embedding(
    device: str,
    dtype: torch.dtype,
    head_size: int,
    num_heads: int,
    num_kv_heads: int,
    is_neox: bool,
    num_tokens: int,
) -> None:
    # Skip invalid GQA configs.
    if num_heads % num_kv_heads != 0:
        pytest.skip("num_heads must be divisible by num_kv_heads")

    rotary_dim = head_size  # Full rotation
    cos_sin_cache = _build_cos_sin_cache(
        MAX_POSITION, rotary_dim, dtype, device
    )

    # Random positions (arbitrary, non-contiguous to test flexibility).
    positions = torch.randint(0, MAX_POSITION, (num_tokens,), device=device)

    # Random query and key tensors.
    query = torch.randn(num_tokens, num_heads, head_size, dtype=dtype, device=device)
    key = torch.randn(num_tokens, num_kv_heads, head_size, dtype=dtype, device=device)

    # Clone for reference.
    query_ref = query.clone()
    key_ref = key.clone()

    # Run kernel.
    ops.rotary_embedding(positions, query, key, head_size, cos_sin_cache, is_neox)

    # Run reference on CPU copies (ref modifies in-place, so we must capture them).
    query_ref_cpu = query_ref.cpu()
    key_ref_cpu = key_ref.cpu()
    _ref_rotary_embedding(
        positions.cpu(),
        query_ref_cpu,
        key_ref_cpu,
        head_size,
        cos_sin_cache.cpu(),
        is_neox,
    )

    # Compare. Use relaxed tolerances for fp16/bf16.
    if dtype == torch.float32:
        atol, rtol = 1e-5, 1e-5
    elif dtype == torch.float16:
        atol, rtol = 1e-3, 1e-3
    else:  # bfloat16
        atol, rtol = 2e-2, 2e-2

    torch.testing.assert_close(query.cpu(), query_ref_cpu, atol=atol, rtol=rtol)
    torch.testing.assert_close(key.cpu(), key_ref_cpu, atol=atol, rtol=rtol)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.parametrize("is_neox", [True])
@torch.inference_mode()
def test_rotary_embedding_no_key(
    device: str,
    dtype: torch.dtype,
    is_neox: bool,
) -> None:
    """Test that passing key=None works correctly."""
    head_size = 128
    num_heads = 8
    num_tokens = 4
    rotary_dim = head_size
    cos_sin_cache = _build_cos_sin_cache(
        MAX_POSITION, rotary_dim, dtype, device
    )
    positions = torch.randint(0, MAX_POSITION, (num_tokens,), device=device)
    query = torch.randn(num_tokens, num_heads, head_size, dtype=dtype, device=device)

    query_ref = query.clone()

    # Run kernel with key=None.
    ops.rotary_embedding(positions, query, None, head_size, cos_sin_cache, is_neox)

    # Run reference with key=None on CPU copy.
    query_ref_cpu = query_ref.cpu()
    _ref_rotary_embedding(
        positions.cpu(),
        query_ref_cpu,
        None,
        head_size,
        cos_sin_cache.cpu(),
        is_neox,
    )

    torch.testing.assert_close(query.cpu(), query_ref_cpu, atol=1e-5, rtol=1e-5)
