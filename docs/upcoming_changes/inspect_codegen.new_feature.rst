Add inspect_codegen
-------------------

Add ``Dispatcher.inspect_codegen()`` and ``python -m numba.misc.codegen_card``
to print a compact ISA + ``llvm-mca`` card (SIMD, FMA, cycles/iter, port
pressure, token cost vs ``inspect_llvm`` / ``inspect_asm``). Optional
dependencies: ``capstone`` and ``llvm-mca``.
