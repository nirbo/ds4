# Ornith-35 Linear K/V Extension

This MLX 0.32 extension aliases cache Metal buffers and writes a contiguous
update into a previously unused token range. It supports paired BF16 K/V and
the Ornith-35 packed spherical-MSE K/V payloads plus BF16 norms. It exists to remove
the full-prefix `concatenate` allocation and copy from advancing decode.

Build it with:

```sh
ornith35/build_extensions.sh
```

These append operations are intentionally not general functional MLX
operations. Each output and input share storage, so evaluating an output
mutates every retained snapshot of its input. Only single-owner linear decode
sessions may use them. Such sessions advance eagerly, have fixed capacity, and
provide no rollback or branching contract. Ordinary immutable decode sessions
remain the authoritative fallback.
