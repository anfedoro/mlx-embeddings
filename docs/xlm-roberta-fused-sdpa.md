# XLM-RoBERTa fused attention

The XLM-RoBERTa implementation uses `mx.fast.scaled_dot_product_attention` for the normal inference path instead of materializing the attention score and probability tensors explicitly.

The original implementation performed these operations separately in every encoder layer:

```python
scores = queries @ keys.swapaxes(-1, -2)
probs = softmax(scores)
context = probs @ values
```

The fused MLX SDPA kernel reduces intermediate memory traffic and improves throughput, especially for longer sequences. The manual implementation is retained when `output_attentions=True` or a `head_mask` is supplied, because those modes require access to the attention probabilities.

## Measured result

Measured on an Apple M4 Max with `mlx-community/bge-m3-mlx-8bit`, batch size 32:

| Sequence length | Previous | Fused SDPA | Speedup |
| ---: | ---: | ---: | ---: |
| 99 tokens | 134.7 chunks/s | 169.6 chunks/s | 1.26x |
| 584 tokens | 17.6 chunks/s | 26.4 chunks/s | 1.50x |

These numbers are workload-specific rather than a general performance guarantee. The gain grows with sequence length because attention accounts for a larger fraction of total encoder work.

## Padding masks

When an attention mask is not supplied, the model now derives it from `pad_token_id`. The additive mask is converted to the same dtype as the embedding output before it is passed to fused SDPA. This is required by MLX and prevents padding tokens from contributing to attention and pooling.

## BGE-M3 sparse output

The MLX community conversion of BGE-M3 currently contains the XLM-RoBERTa backbone and pooling weights but may not include the learned sparse head from the original Hugging Face model. The official BGE-M3 repository distributes that head separately as `sparse_linear.pt`.

The sparse projection is inexpensive once `last_hidden_state` has been computed:

```python
sparse = mx.maximum(
    hidden_states.reshape(-1, hidden_size)
    @ weight.reshape(hidden_size, 1),
    0,
).reshape(batch_size, sequence_length)
```

The sparse head should not be embedded directly into the generic XLM-RoBERTa architecture because it is model-specific. A separate BGE-M3 helper can load converted MLX/safetensors weights and derive dense and sparse representations from one backbone forward. Converting `sparse_linear.pt` should remain an optional workflow so that `mlx-embeddings` does not acquire a mandatory PyTorch dependency.
