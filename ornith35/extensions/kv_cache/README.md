# Ornith-35 Linear K/V Extension

This MLX 0.32 extension aliases a BF16 cache's Metal buffer and writes a
contiguous update into a previously unused token range. It exists to remove
the full-prefix `concatenate` allocation and copy from advancing decode.

Build it with:

```sh
ornith35/build_extensions.sh
```

`append_bf16` is intentionally not a general functional MLX operation. Its
output and input share storage, so evaluating the output mutates every retained
snapshot of the input. Only `TextLinearDecodeSession` may use it. That session
has one owner, advances eagerly, has fixed capacity, and provides no rollback
or branching contract. The ordinary immutable decode session remains the
authoritative fallback.
