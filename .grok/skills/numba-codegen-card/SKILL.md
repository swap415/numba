---
name: numba-codegen-card
description: Use Numba inspect_codegen / codegen_card instead of dumping inspect_llvm or inspect_asm when reviewing JIT performance, SIMD, FMA, or cross-CPU cost. Triggers on inspect_asm, inspect_llvm, codegen, llvm-mca, capstone, why is this kernel slow, vectorization, /codegen-card.
---

# Numba codegen card

Do not dump `inspect_llvm()` or `inspect_asm()` into context. Those are 10–40k characters of wrappers + NRT. Use the card.

```python
print(fn.inspect_codegen(fn.signatures[0]))
# or
from numba.misc.codegen_card import inspect_codegen, compare_codegen
print(inspect_codegen(fn))
print(compare_codegen(ieee_fn, fastmath_fn))
```

CLI: `python -m numba.misc.codegen_card` and `--compare`.

## What the card already extracted

- First LLVM function only. Full-module IR cannot be retargeted (`NRT_decref` uses `llvm.x86.atomic.sub.cc`).
- Hottest packed SIMD loop. `llvm-mca` on the whole dump is noise.
- JIT code is in `.ltext`, not `.text`.
- ISA (AVX/NEON, FMA), cycles/iter, port bottleneck, token cost vs dumps.

## How to read it

- `no FMA` + `vmul+vadd` → try `@jit(fastmath=True)` if IEEE allows.
- MCA cycles/iter is compute-only. If wall time ≫ MCA estimate, the kernel is bandwidth-bound. Do not rewrite arithmetic.
- Same host asm + different `-mcpu` is a schedule model, not a rerun. Zen4 ymm looking worse than Alder Lake is the 256-bit model, not a regression.
- Retarget (`retarget=('apple-m1',)`) needs kernel-only IR + `llc`.

## When dumps are still allowed

Only if the user asked for the raw IR/asm, or the card says no packed loop was found.
