"""Tests for numba.misc.codegen_card."""

import struct
import unittest

from numba.tests.support import TestCase


SAMPLE_IR = """\
; ModuleID = 'saxpy'
target triple = "x86_64-unknown-linux-gnu"

define i32 @kernel(ptr %retptr) {
  ret i32 0
}

define linkonce_odr void @NRT_decref(ptr %.1) {
  %0 = tail call i8 @llvm.x86.atomic.sub.cc.i64(ptr %.1, i64 1, i32 4)
  ret void
}

declare i8 @llvm.x86.atomic.sub.cc.i64(ptr, i64, i32)

attributes #0 = { nounwind }
!0 = distinct !{!0, !1}
!1 = !{!"llvm.loop.isvectorized", i32 1}
"""

SAMPLE_ASM = """\
.globl kernel
kernel:
	testq	%rax, %rax
	jle	.LBB0_19
.LBB0_8:
	vmulpd	(%rsi,%r10,8), %ymm1, %ymm2
	vaddpd	(%rdx,%r10,8), %ymm2, %ymm2
	vmovupd	%ymm2, (%rcx,%r10,8)
	addq	$4, %r10
	cmpq	%r10, %r8
	jne	.LBB0_8
.LBB0_19:
	retq
"""


def _elf64_with_sections(sections):
    """Minimal ELF64 LE object: empty .text plus named executable sections."""
    # ehdr 64 bytes
    e_shentsize = 64
    names = b"\0" + b"".join(n.encode() + b"\0" for n, _f, _d in sections)
    # section 0 is NULL; last is .shstrtab
    shstr = b"\0.text\0.shstrtab\0" + b"".join(
        n.encode() + b"\0" for n, _f, _d in sections if n not in (".text", ".shstrtab")
    )
    # rebuild names consistently
    name_list = ["", ".text", ".shstrtab"] + [
        n for n, _f, _d in sections if n not in (".text", ".shstrtab")
    ]
    data_by_name = {".text": b"", ".shstrtab": b""}
    flags_by_name = {".text": 0x6, ".shstrtab": 0}  # ALLOC|EXEC
    for n, flags, data in sections:
        data_by_name[n] = data
        flags_by_name[n] = flags
        if n not in name_list:
            name_list.append(n)

    shstr = b""
    name_off = {}
    for n in name_list:
        name_off[n] = len(shstr)
        shstr += n.encode() + b"\0"
    data_by_name[".shstrtab"] = shstr

    # layout: ehdr | section data | shdrs
    ehdr_size = 64
    payloads = []
    offsets = {}
    cursor = ehdr_size
    for n in name_list:
        blob = data_by_name.get(n, b"")
        offsets[n] = cursor
        payloads.append(blob)
        cursor += len(blob)
    shoff = cursor
    shnum = len(name_list)

    ehdr = bytearray(64)
    ehdr[0:4] = b"\x7fELF"
    ehdr[4] = 2  # 64-bit
    ehdr[5] = 1  # LE
    ehdr[6] = 1
    struct.pack_into("<H", ehdr, 16, 1)  # ET_REL
    struct.pack_into("<H", ehdr, 18, 62)  # EM_X86_64
    struct.pack_into("<I", ehdr, 20, 1)
    struct.pack_into("<Q", ehdr, 40, shoff)
    struct.pack_into("<H", ehdr, 52, 64)  # ehsize
    struct.pack_into("<H", ehdr, 58, e_shentsize)
    struct.pack_into("<H", ehdr, 60, shnum)
    struct.pack_into("<H", ehdr, 62, name_list.index(".shstrtab"))

    shdrs = bytearray()
    for n in name_list:
        blob = data_by_name.get(n, b"")
        flags = flags_by_name.get(n, 0)
        sh_type = 3 if n == ".shstrtab" else (0 if n == "" else 1)
        rec = bytearray(64)
        struct.pack_into("<I", rec, 0, name_off[n])
        struct.pack_into("<I", rec, 4, sh_type)
        struct.pack_into("<Q", rec, 8, flags)
        struct.pack_into("<Q", rec, 24, offsets[n] if n else 0)
        struct.pack_into("<Q", rec, 32, len(blob))
        shdrs += rec

    return bytes(ehdr) + b"".join(payloads) + bytes(shdrs)


class TestExtract(TestCase):
    def test_extract_kernel_ir_drops_nrt_x86_intrinsic(self):
        from numba.misc.codegen_card import extract_kernel_ir

        kernel = extract_kernel_ir(SAMPLE_IR)
        self.assertIn("define i32 @kernel", kernel)
        self.assertNotIn("llvm.x86.atomic.sub.cc", kernel)
        self.assertNotIn("NRT_decref", kernel)
        self.assertIn("llvm.loop.isvectorized", kernel)

    def test_extract_hot_loop_picks_packed_vector_body(self):
        from numba.misc.codegen_card import extract_hot_loop

        loop = extract_hot_loop(SAMPLE_ASM)
        self.assertIn("vmulpd", loop)
        self.assertIn("jne", loop)
        self.assertNotIn("retq", loop)
        self.assertIn(".LBB0_8:", loop)

    def test_extract_hot_loop_recognizes_fma(self):
        from numba.misc.codegen_card import extract_hot_loop

        asm = """\
.LBB0_8:
	vfmadd213pd	(%rdx,%r10,8), %ymm1, %ymm2
	vmovupd	%ymm2, (%rcx,%r10,8)
	addq	$4, %r10
	jne	.LBB0_8
.LBB0_9:
	retq
"""
        loop = extract_hot_loop(asm)
        self.assertIn("vfmadd213pd", loop)

    def test_elf_prefers_ltext_over_empty_text(self):
        from numba.misc.codegen_card import elf_executable_bytes

        payload = b"\x90" * 32
        elf = _elf64_with_sections(
            [
                (".text", 0x6, b""),
                (".ltext", 0x10000006, payload),
            ]
        )
        name, data = elf_executable_bytes(elf)
        self.assertEqual(name, ".ltext")
        self.assertEqual(data, payload)

    def test_estimate_tokens_is_chars_over_four(self):
        from numba.misc.codegen_card import estimate_tokens

        self.assertEqual(estimate_tokens("abcd" * 10), 10)


def _compile_saxpy(fastmath=False):
    import numpy as np
    from numba import jit

    @jit(fastmath=fastmath)
    def saxpy(a, x, y, out):
        for i in range(x.shape[0]):
            out[i] = a * x[i] + y[i]

    n = 64
    saxpy(2.0, np.ones(n), np.ones(n), np.empty(n))
    return saxpy


class TestCard(TestCase):
    def test_card_is_much_smaller_than_inspect_dumps(self):
        from numba.misc.codegen_card import inspect_codegen

        fn = _compile_saxpy()
        card = inspect_codegen(fn)
        self.assertGreater(card.tokens.llvm_chars, 5_000)
        self.assertGreater(card.tokens.asm_chars, 1_000)
        self.assertLess(card.tokens.card_chars, card.tokens.llvm_chars / 5)
        self.assertLess(card.tokens.card_tokens, card.tokens.asm_tokens)
        self.assertGreater(card.tokens.savings_vs_dumps, 0.7)
        text = card.brief()
        self.assertIn("saxpy", text)
        self.assertNotIn("llvm.x86.atomic", text)

    def test_card_reports_avx_and_no_fma_by_default(self):
        from numba.misc.codegen_card import inspect_codegen

        card = inspect_codegen(_compile_saxpy(fastmath=False))
        self.assertFalse(card.isa.has_fma)
        self.assertIn("AVX", card.isa.simd)
        self.assertTrue(any("FMA" in n or "fma" in n.lower() for n in card.notes)
                        or any("IEEE" in n for n in card.notes))

    def test_fastmath_enables_fma(self):
        from numba.misc.codegen_card import inspect_codegen

        card = inspect_codegen(_compile_saxpy(fastmath=True))
        self.assertTrue(card.isa.has_fma)

    def test_mca_reports_positive_throughput(self):
        from numba.misc.codegen_card import has_llvm_mca, inspect_codegen

        if not has_llvm_mca():
            self.skipTest("llvm-mca not on PATH")
        card = inspect_codegen(_compile_saxpy())
        ok = [m for m in card.mca if m.ok]
        self.assertTrue(ok)
        self.assertGreater(ok[0].cycles_per_iter, 0)
        self.assertIsNotNone(ok[0].bottleneck)

    def test_dispatcher_inspect_codegen(self):
        fn = _compile_saxpy()
        card = fn.inspect_codegen(fn.signatures[0])
        self.assertTrue(hasattr(card, "brief"))
        self.assertIn("saxpy", card.brief())

    def test_compare_mentions_fma_delta(self):
        from numba.misc.codegen_card import compare_codegen

        slow = _compile_saxpy(fastmath=False)
        fast = _compile_saxpy(fastmath=True)
        report = compare_codegen(slow, fast)
        text = str(report)
        self.assertIn("FMA", text.upper())
        self.assertLess(report.tokens.card_chars,
                        report.tokens.llvm_chars + report.tokens.asm_chars)
        self.assertTrue(any("cycles/iter" in d for d in report.deltas))

    def test_cli_demo_exits_zero(self):
        from numba.misc.codegen_card import main
        from numba.tests.support import captured_stdout

        with captured_stdout() as out:
            rc = main([])
        self.assertEqual(rc, 0)
        self.assertIn("codegen card", out.getvalue())
        self.assertIn("% smaller", out.getvalue())


if __name__ == "__main__":
    unittest.main()
