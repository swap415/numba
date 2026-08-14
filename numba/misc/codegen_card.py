"""Compact codegen card: ISA + static pipeline cost, not a raw IR/asm dump.

Optional extras:
- ``capstone`` to decode the JIT object (Numba emits ``.ltext``, not ``.text``)
- ``llvm-mca`` on ``$PATH`` for cycles/iter and port pressure
- ``llc`` on ``$PATH`` only when retargeting the kernel IR to another triple

Full-module IR cannot be retargeted: ``NRT_decref`` uses
``llvm.x86.atomic.sub.cc``. Always extract the first ``define`` first.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import struct
import subprocess
from dataclasses import dataclass
from typing import Any, Optional


def estimate_tokens(text: str) -> int:
    """Cheap token estimate: characters / 4. Good enough to compare dumps."""
    return max(1, len(text) // 4) if text else 0


def extract_kernel_ir(ir: str) -> str:
    """Keep the first function plus attributes/metadata. Drop NRT / wrappers."""
    out: list[str] = []
    in_fn = False
    seen = 0
    for line in ir.splitlines():
        if line.startswith("define "):
            seen += 1
            in_fn = seen == 1
            if in_fn:
                out.append(line)
            continue
        if line.startswith("attributes #") or line.startswith("!"):
            out.append(line)
            continue
        if in_fn:
            out.append(line)
            if line.strip() == "}":
                in_fn = False
    return "\n".join(out) + ("\n" if out else "")


_LABEL = re.compile(r"^(\.LBB\d+_\d+):")
_BACK_EDGE = re.compile(r"\b(jne|je|jle|jg|ja|jb|b\.ne|b\.eq|b\.gt|b\.lt)\b")
_PACKED = re.compile(
    r"\b(v(mul|add|sub|div)p[ds]|vfmadd\d*p[ds]|vfmsub\d*p[ds]|"
    r"fmul|fadd|fmla|ldp\s+q|\.2d|\.4s|\.8h|ymm|zmm)\b"
)


def extract_hot_loop(asm: str) -> str:
    """First basic block that is a packed-SIMD loop with a back-edge."""
    lines = asm.splitlines()
    labels = [i for i, line in enumerate(lines) if _LABEL.match(line.strip() or line)]
    # also accept labels that start the line without leading whitespace
    if not labels:
        labels = [i for i, line in enumerate(lines) if _LABEL.match(line)]
    for idx, start in enumerate(labels):
        end = labels[idx + 1] if idx + 1 < len(labels) else len(lines)
        block = lines[start:end]
        text = "\n".join(block)
        m = _LABEL.match(block[0].strip()) or _LABEL.match(block[0])
        if m is None:
            continue
        label = m.group(1)
        packed = bool(_PACKED.search(text) or "vmulpd" in text or ".2d" in text)
        back = any(label in line and _BACK_EDGE.search(line) for line in block[1:])
        if packed and back:
            last = start
            for i, line in enumerate(block):
                if i and label in line:
                    last = start + i
            return "\n".join(lines[start : last + 1]) + "\n"
    raise ValueError("no packed SIMD loop found in assembly")


def elf_executable_bytes(buf: bytes) -> tuple[str, bytes]:
    """Largest executable section. Numba JIT objects put code in ``.ltext``."""
    if buf[:4] != b"\x7fELF":
        raise ValueError("not an ELF object")
    e_shoff = struct.unpack_from("<Q", buf, 40)[0]
    e_shentsize, e_shnum, e_shstrndx = struct.unpack_from("<HHH", buf, 58)

    def shdr(i: int) -> tuple[int, int, int, int]:
        off = e_shoff + i * e_shentsize
        sh_name, _t, sh_flags, _a, sh_offset, sh_size = struct.unpack_from(
            "<IIQQQQ", buf, off
        )
        return sh_name, sh_flags, sh_offset, sh_size

    _n, _f, str_off, str_size = shdr(e_shstrndx)
    strtab = buf[str_off : str_off + str_size]
    SHF_EXECINSTR = 0x4
    cands: list[tuple[str, bytes]] = []
    for i in range(e_shnum):
        sh_name, sh_flags, sh_offset, sh_size = shdr(i)
        if not (sh_flags & SHF_EXECINSTR) or sh_size == 0:
            continue
        n = strtab[sh_name : strtab.index(b"\0", sh_name)].decode()
        cands.append((n, buf[sh_offset : sh_offset + sh_size]))
    if not cands:
        raise ValueError("no executable ELF section")
    return max(cands, key=lambda kv: len(kv[1]))


def _find_tool(name: str) -> Optional[str]:
    env = os.environ.get(f"NUMBA_{name.upper().replace('-', '_')}")
    if env and os.path.isfile(env) and os.access(env, os.X_OK):
        return env
    found = shutil.which(name)
    if found:
        return found
    for root in ("/usr/lib", "/usr/lib64"):
        if not os.path.isdir(root):
            continue
        for entry in sorted(os.listdir(root), reverse=True):
            cand = os.path.join(root, entry, "bin", name)
            if os.path.isfile(cand) and os.access(cand, os.X_OK):
                return cand
    return None


def has_llvm_mca() -> bool:
    return _find_tool("llvm-mca") is not None


def host_cpu_name() -> str:
    try:
        from llvmlite import binding as llvm
        return llvm.get_host_cpu_name()
    except Exception:
        return "generic"


@dataclass
class ISASummary:
    section: str
    n_insns: int
    simd: str
    has_fma: bool
    top_mnemonics: tuple = ()
    groups: tuple = ()


@dataclass
class MCASummary:
    cpu: str
    ok: bool
    cycles_per_iter: Optional[float] = None
    ipc: Optional[float] = None
    n_insns: Optional[int] = None
    bottleneck: Optional[str] = None
    error: Optional[str] = None
    triple: Optional[str] = None


@dataclass
class TokenBudget:
    llvm_chars: int
    asm_chars: int
    card_chars: int

    @property
    def llvm_tokens(self) -> int:
        return self.llvm_chars // 4

    @property
    def asm_tokens(self) -> int:
        return self.asm_chars // 4

    @property
    def card_tokens(self) -> int:
        return self.card_chars // 4

    @property
    def savings_vs_dumps(self) -> float:
        dumps = self.llvm_tokens + self.asm_tokens
        if dumps <= 0:
            return 0.0
        return 1.0 - (self.card_tokens / dumps)


@dataclass
class CodegenCard:
    name: str
    signature: str
    isa: Optional[ISASummary]
    host_loop: str
    mca: tuple
    tokens: TokenBudget
    notes: tuple = ()

    def brief(self) -> str:
        lines = [
            f"codegen card: {self.name}{self.signature}",
        ]
        if self.isa:
            fma = "FMA" if self.isa.has_fma else "no FMA"
            lines.append(
                f"ISA: {self.isa.simd}  {fma}  "
                f"{self.isa.n_insns} insns  section={self.isa.section}"
            )
            if self.isa.top_mnemonics:
                top = ", ".join(f"{m} {c}" for m, c in self.isa.top_mnemonics[:6])
                lines.append(f"top: {top}")
        else:
            lines.append("ISA: (capstone unavailable or no object)")
        if self.mca:
            lines.append("MCA (hot loop, static; ignores cache/DRAM):")
            for m in self.mca:
                if m.ok:
                    lines.append(
                        f"  {m.cpu:14}  {m.cycles_per_iter:6.2f} cyc/iter  "
                        f"IPC {m.ipc:4.2f}  {m.n_insns} insns  "
                        f"bound {m.bottleneck}"
                    )
                else:
                    lines.append(f"  {m.cpu:14}  FAIL  {m.error}")
        for n in self.notes:
            lines.append(f"note: {n}")
        lines.append(
            f"tokens: card {self.tokens.card_tokens} vs "
            f"llvm {self.tokens.llvm_tokens} + asm {self.tokens.asm_tokens} "
            f"({100*self.tokens.savings_vs_dumps:.0f}% smaller than dumps)"
        )
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.brief()


@dataclass
class CompareReport:
    before: CodegenCard
    after: CodegenCard
    tokens: TokenBudget
    deltas: tuple = ()

    def __str__(self) -> str:
        lines = ["codegen compare", str(self.before), "", str(self.after), ""]
        lines.extend(f"delta: {d}" for d in self.deltas)
        lines.append(
            f"tokens: both cards {self.tokens.card_tokens} vs "
            f"both dumps {self.tokens.llvm_tokens + self.tokens.asm_tokens}"
        )
        return "\n".join(lines)


def summarize_isa(code: bytes, arch: str = "x86") -> ISASummary:
    try:
        from capstone import CS_ARCH_ARM64, CS_ARCH_X86, CS_MODE_ARM, CS_MODE_64, Cs
    except ImportError as e:
        raise RuntimeError("capstone package needed for ISA summary") from e

    if arch in ("x86", "x86_64"):
        md = Cs(CS_ARCH_X86, CS_MODE_64)
    elif arch in ("aarch64", "arm64"):
        md = Cs(CS_ARCH_ARM64, CS_MODE_ARM)
    else:
        raise ValueError(arch)
    md.detail = True

    from collections import Counter

    mnems: Counter = Counter()
    groups: Counter = Counter()
    widths: Counter = Counter()
    n = 0
    has_fma = False
    for insn in md.disasm(code, 0):
        n += 1
        mnems[insn.mnemonic] += 1
        for g in insn.groups:
            groups[insn.group_name(g)] += 1
        op = insn.op_str
        if "zmm" in op:
            widths["512"] += 1
        elif "ymm" in op:
            widths["256"] += 1
        elif "xmm" in op or ".2d" in op or ".4s" in op:
            widths["128"] += 1
        elif insn.mnemonic in ("ldp", "stp") and "q" in op:
            widths["128"] += 1
        m = insn.mnemonic
        if "fma" in m or m.startswith("vfmadd") or m.startswith("vfnmadd") or m == "fmla":
            has_fma = True
    gnames = set(groups)
    if widths["512"] or "avx512" in gnames:
        simd = "AVX-512"
    elif widths["256"] or "avx" in gnames:
        simd = "AVX2" if widths["256"] else "AVX"
    elif "neon" in gnames or (arch.startswith("arm") and widths["128"]):
        simd = "NEON-128"
    elif widths["128"]:
        simd = "SSE-128"
    else:
        simd = "scalar"
    return ISASummary(
        section="?",
        n_insns=n,
        simd=simd,
        has_fma=has_fma,
        top_mnemonics=tuple(mnems.most_common(8)),
        groups=tuple(groups.most_common(8)),
    )


def run_mca(asm: str, cpu: str, triple: Optional[str] = None) -> MCASummary:
    exe = _find_tool("llvm-mca")
    if exe is None:
        return MCASummary(cpu=cpu, ok=False, error="llvm-mca not found", triple=triple)
    cmd = [exe, f"-mcpu={cpu}", "--json", "-iterations=100"]
    if triple:
        cmd.append(f"-mtriple={triple}")
    try:
        proc = subprocess.run(
            cmd, input=asm, text=True, capture_output=True, timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        return MCASummary(cpu=cpu, ok=False, error=str(e), triple=triple)
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "llvm-mca failed").strip().splitlines()
        return MCASummary(cpu=cpu, ok=False, error=err[0][:200] if err else "fail",
                          triple=triple)
    try:
        data = json.loads(proc.stdout)
        region = data["CodeRegions"][0]
        summary = region["SummaryView"]
        resources = data["TargetInfo"]["Resources"]
        infos = region["ResourcePressureView"]["ResourcePressureInfo"]
        n_insns = len(region["Instructions"])
        totals = [x for x in infos if x["InstructionIndex"] == n_insns]
        if totals:
            top = max(totals, key=lambda x: x["ResourceUsage"])
            name = resources[top["ResourceIndex"]].rstrip("\x00")
            bottleneck = f"{name} {top['ResourceUsage']:.2f}"
        else:
            bottleneck = "?"
        return MCASummary(
            cpu=cpu,
            ok=True,
            cycles_per_iter=float(summary["BlockRThroughput"]),
            ipc=float(summary["IPC"]),
            n_insns=n_insns,
            bottleneck=bottleneck,
            triple=triple,
        )
    except (KeyError, ValueError, json.JSONDecodeError) as e:
        return MCASummary(cpu=cpu, ok=False, error=f"parse: {e}", triple=triple)


def _notes_for(isa: Optional[ISASummary], loop: str, mca: tuple) -> tuple:
    notes = []
    if isa is not None and not isa.has_fma:
        notes.append("no FMA (IEEE default; try @jit(fastmath=True))")
    if isa is not None and isa.has_fma:
        notes.append("FMA present")
    if "vmulpd" in loop and "vfmadd" not in loop:
        notes.append("hot loop is vmul+vadd, not FMA")
    ok = [m for m in mca if m.ok]
    if ok:
        notes.append(
            "MCA is compute-only; bandwidth-bound kernels run slower than this"
        )
    return tuple(notes)


def inspect_codegen(
    func: Any,
    signature: Any = None,
    *,
    cpus: Optional[list] = None,
    retarget: tuple = (),
) -> CodegenCard:
    """Build a compact codegen card for a compiled Numba dispatcher."""
    if signature is None:
        if not getattr(func, "signatures", None):
            raise ValueError("function has no compiled signatures; call it first")
        signature = func.signatures[0]
    ir = func.inspect_llvm(signature)
    asm = func.inspect_asm(signature)
    name = getattr(getattr(func, "py_func", None), "__name__", None) or getattr(
        func, "__name__", "func"
    )

    isa = None
    try:
        obj = func.overloads[signature].library._compiled_object
        if obj:
            sec, code = elf_executable_bytes(obj)
            isa = summarize_isa(code, "x86")
            isa = ISASummary(
                section=sec,
                n_insns=isa.n_insns,
                simd=isa.simd,
                has_fma=isa.has_fma,
                top_mnemonics=isa.top_mnemonics,
                groups=isa.groups,
            )
    except Exception:
        isa = None

    try:
        loop = extract_hot_loop(asm)
    except ValueError:
        loop = ""

    if cpus is None:
        cpus = [host_cpu_name()]
    mca_rows = []
    if loop:
        for cpu in cpus:
            mca_rows.append(run_mca(loop, cpu))
        for spec in retarget:
            if isinstance(spec, (tuple, list)):
                triple, cpu = spec
            else:
                triple, cpu = "aarch64-unknown-linux-gnu", spec
            try:
                asm_x = _retarget_loop(ir, triple, cpu)
                mca_rows.append(run_mca(asm_x, cpu, triple=triple))
            except Exception as e:
                mca_rows.append(
                    MCASummary(cpu=cpu, ok=False, error=str(e)[:200], triple=triple)
                )

    notes = _notes_for(isa, loop, tuple(mca_rows))
    # token card size from a first pass without tokens, then fill
    tmp_tokens = TokenBudget(len(ir), len(asm), 0)
    card = CodegenCard(
        name=name,
        signature=str(signature),
        isa=isa,
        host_loop=loop,
        mca=tuple(mca_rows),
        tokens=tmp_tokens,
        notes=notes,
    )
    brief = card.brief()
    card.tokens = TokenBudget(len(ir), len(asm), len(brief))
    return card


def _retarget_loop(ir: str, triple: str, cpu: str) -> str:
    llc = _find_tool("llc")
    if llc is None:
        raise RuntimeError("llc not found")
    kernel = extract_kernel_ir(ir)
    proc = subprocess.run(
        [llc, f"-mtriple={triple}", f"-mcpu={cpu}", "-o", "-"],
        input=kernel,
        text=True,
        capture_output=True,
        timeout=30,
    )
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr or "llc failed").splitlines()[0][:200])
    return extract_hot_loop(proc.stdout)


def compare_codegen(before: Any, after: Any, **kwargs) -> CompareReport:
    """Structured delta between two compiled functions."""
    a = inspect_codegen(before, **kwargs)
    b = inspect_codegen(after, **kwargs)
    deltas = []
    if a.isa and b.isa:
        if a.isa.has_fma != b.isa.has_fma:
            deltas.append(f"FMA {a.isa.has_fma} -> {b.isa.has_fma}")
        if a.isa.simd != b.isa.simd:
            deltas.append(f"SIMD {a.isa.simd} -> {b.isa.simd}")
    a_ok = [m for m in a.mca if m.ok]
    b_ok = [m for m in b.mca if m.ok]
    if a_ok and b_ok:
        da, db = a_ok[0].cycles_per_iter, b_ok[0].cycles_per_iter
        deltas.append(f"cycles/iter {da:.2f} -> {db:.2f}")
    tokens = TokenBudget(
        llvm_chars=a.tokens.llvm_chars + b.tokens.llvm_chars,
        asm_chars=a.tokens.asm_chars + b.tokens.asm_chars,
        card_chars=len(str(a)) + len(str(b)),
    )
    # rebuild so __str__ includes deltas; card_chars approx after
    report = CompareReport(before=a, after=b, tokens=tokens, deltas=tuple(deltas))
    text = str(report)
    report.tokens = TokenBudget(
        llvm_chars=tokens.llvm_chars,
        asm_chars=tokens.asm_chars,
        card_chars=len(text),
    )
    return report


def _demo_saxpy(fastmath=False):
    import numpy as np
    from numba import jit

    @jit(fastmath=fastmath)
    def saxpy(a, x, y, out):
        for i in range(x.shape[0]):
            out[i] = a * x[i] + y[i]

    n = 4096
    saxpy(2.0, np.ones(n), np.ones(n), np.empty(n))
    return saxpy


def main(argv: Optional[list] = None) -> int:
    import argparse

    p = argparse.ArgumentParser(
        prog="python -m numba.misc.codegen_card",
        description="Compact codegen card for a Numba kernel (ISA + llvm-mca).",
    )
    p.add_argument(
        "--compare",
        action="store_true",
        help="Compare IEEE saxpy vs fastmath (shows FMA / token savings)",
    )
    p.add_argument("--json", action="store_true", help="JSON on stdout")
    p.add_argument(
        "--cpu",
        action="append",
        dest="cpus",
        help="Extra CPU model for llvm-mca (repeatable)",
    )
    args = p.parse_args(argv)

    if args.compare:
        slow = _demo_saxpy(False)
        fast = _demo_saxpy(True)
        report = compare_codegen(slow, fast, cpus=args.cpus)
        print(report)
        return 0

    card = inspect_codegen(_demo_saxpy(False), cpus=args.cpus)
    if args.json:
        print(json.dumps({
            "name": card.name,
            "brief": card.brief(),
            "tokens": {
                "card": card.tokens.card_tokens,
                "llvm": card.tokens.llvm_tokens,
                "asm": card.tokens.asm_tokens,
                "savings": card.tokens.savings_vs_dumps,
            },
        }, indent=2))
    else:
        print(card)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
