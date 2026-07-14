from pathlib import Path
from typing import Optional, Tuple, Union

import mlx.core as mx


PathLike = Union[str, Path]


def load_sparse_linear(path: PathLike) -> Tuple[mx.array, mx.array]:
    """Load a converted sparse linear head from an MLX-readable file.

    The file must contain ``weight`` and ``bias`` tensors. For BGE-M3 the
    expected weight shape is ``(1, hidden_size)`` or ``(hidden_size, 1)`` and
    the bias is scalar or shape ``(1,)``.

    ``mx.load`` supports safetensors and MLX npz files. PyTorch ``.pt`` files
    are intentionally not loaded here so that mlx-embeddings does not require
    PyTorch at runtime.
    """

    state = mx.load(str(path))
    weight = state["weight"]
    bias = state["bias"]

    if weight.ndim != 2 or 1 not in weight.shape:
        raise ValueError(
            "Sparse weight must have shape (1, hidden_size) or "
            f"(hidden_size, 1), received {weight.shape}"
        )

    if weight.shape[0] == 1:
        weight = weight.reshape(-1, 1)

    if bias.size != 1:
        raise ValueError(f"Sparse bias must contain one value, received {bias.shape}")

    bias = bias.reshape(())
    mx.eval(weight, bias)
    return weight, bias


def sparse_token_weights(
    hidden_states: mx.array,
    weight: mx.array,
    bias: mx.array,
    attention_mask: Optional[mx.array] = None,
) -> mx.array:
    """Project token hidden states into non-negative lexical weights.

    Args:
        hidden_states: Tensor with shape ``(batch, sequence, hidden_size)``.
        weight: Sparse projection weight with shape ``(hidden_size, 1)``.
        bias: Scalar sparse projection bias.
        attention_mask: Optional mask with shape ``(batch, sequence)``. Masked
            token weights are set to zero.

    Returns:
        Tensor with shape ``(batch, sequence)``.
    """

    if hidden_states.ndim != 3:
        raise ValueError(
            "hidden_states must have shape (batch, sequence, hidden_size), "
            f"received {hidden_states.shape}"
        )

    batch_size, sequence_length, hidden_size = hidden_states.shape

    if weight.shape != (hidden_size, 1):
        raise ValueError(
            f"weight must have shape ({hidden_size}, 1), received {weight.shape}"
        )

    scores = (
        hidden_states.reshape(batch_size * sequence_length, hidden_size) @ weight
    ).reshape(batch_size, sequence_length)
    scores = mx.maximum(scores + bias, 0)

    if attention_mask is not None:
        if attention_mask.shape != (batch_size, sequence_length):
            raise ValueError(
                "attention_mask must match the first two hidden-state dimensions, "
                f"received {attention_mask.shape}"
            )
        scores = scores * attention_mask.astype(scores.dtype)

    return scores
