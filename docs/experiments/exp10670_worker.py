"""Worker for the numba issue #10670 unroll-metadata experiment.

Runs ONE phase in a fresh process and prints a JSON result on stdout.

Usage: python exp10670_worker.py <phase>

Phases:
  baseline             - no intervention
  inner_disable        - llvm.loop.unroll.disable on the INNERMOST loop latch,
                         applied ONLY while compiling add_for
  all_runtime_disable  - llvm.loop.unroll.runtime.disable on ALL loop latches
                         of all three functions (the production-shaped policy:
                         runtime unrolling off, full unroll of constant trip
                         counts still allowed)
  all_disable          - llvm.loop.unroll.disable on ALL latches of all three
                         functions (negative control: expected to kill the
                         full unroll of the constant range(5) loop)

All experimental code lives here; the numba package is monkeypatched at
runtime only, never modified on disk.
"""
import json
import os
import re
import sys

os.environ['NUMBA_LOOP_VECTORIZE'] = '0'

import numba  # noqa: E402
from numba import njit, uint64, float64, int64  # noqa: E402
from numba.core import lowering, ir as nir  # noqa: E402
import llvmlite  # noqa: E402
import llvmlite.ir as llir  # noqa: E402
import numpy as np  # noqa: E402


# ----------------------------------------------------------------------------
# The three probe functions (issue #10670 shapes)
# ----------------------------------------------------------------------------

def make_funcs():
    @njit(cache=False)
    def add_for(arr, x):
        # inner `for j in range(x)` with a RUNTIME trip count
        for i in range(arr.size):
            for j in range(x):
                arr[i] += 1.0

    @njit(cache=False)
    def add_while_uint(arr, x):
        # uint64 counter while-loop; uint64+int64 unifies to float64 so LLVM
        # will not runtime-unroll it (the accidental "fast" variant)
        for i in range(arr.size):
            j = uint64(0)
            while j < x:
                arr[i] += 1.0
                j += 1

    @njit(cache=False)
    def add_for_const(arr):
        # compile-time-constant trip count: already optimal (full unroll)
        for i in range(arr.size):
            for j in range(5):
                arr[i] += 1.0

    return add_for, add_while_uint, add_for_const


SIG2 = (float64[:], int64)
SIG1 = (float64[:],)


# ----------------------------------------------------------------------------
# Latch tagging: monkeypatch of numba's lowering (approach "a")
# ----------------------------------------------------------------------------

class _TagState:
    active = False
    mode = None    # 'disable' | 'runtime'
    scope = None   # 'innermost' | 'all'
    tagged = []    # (src_offset, tgt_offset, metadata_name) actually applied


def _backedges(blocks):
    """Backedges in numba IR: terminator targets at an offset <= the source
    block's offset (python loops always branch backward to their header)."""
    edges = set()
    for off, blk in blocks.items():
        term = blk.body[-1] if blk.body else None
        if term is None or not hasattr(term, 'get_targets'):
            continue
        for t in term.get_targets():
            if t <= off:
                edges.add((off, t))
    return edges


_orig_lower_block = lowering.Lower.lower_block
_orig_lower_inst = lowering.Lower.lower_inst


def _patched_lower_block(self, block):
    for off, blk in self.blocks.items():
        if blk is block:
            self._exp10670_offset = off
            break
    return _orig_lower_block(self, block)


def _patched_lower_inst(self, inst):
    ret = _orig_lower_inst(self, inst)
    if not _TagState.active:
        return ret
    if not isinstance(inst, (nir.Jump, nir.Branch)):
        return ret
    src = getattr(self, '_exp10670_offset', None)
    if src is None:
        return ret
    back_tgts = [t for t in inst.get_targets() if t <= src]
    if not back_tgts:
        return ret
    if _TagState.scope == 'innermost':
        edges = _backedges(dict(self.blocks.items()))
        if not edges:
            return ret
        # innermost loop = backedge with the highest (latest) header offset
        innermost_tgt = max(t for (_s, t) in edges)
        if innermost_tgt not in back_tgts:
            return ret
    term = self.builder.block.terminator
    if term is None:
        return ret
    mod = self.builder.module
    mdname = ('llvm.loop.unroll.disable' if _TagState.mode == 'disable'
              else 'llvm.loop.unroll.runtime.disable')
    opt = mod.add_metadata([mdname])
    # LLVM's Loop::getLoopID() requires the loop-ID node's first operand to
    # be the node itself (self-referential); otherwise the ID is ignored.
    # llvmlite's add_metadata() cannot express that (and its uniquing cache
    # would choke on a cyclic operand tuple), so create the MDValue directly
    # and splice in the self-reference afterwards.  A cyclic uniqued node is
    # implicitly treated as distinct by LLVM when parsed.
    loop_md = llir.values.MDValue(mod, [opt], name=str(len(mod.metadata)))
    loop_md.operands = (loop_md, opt)
    term.set_metadata('llvm.loop', loop_md)
    _TagState.tagged.append((src, min(back_tgts), mdname))
    return ret


lowering.Lower.lower_block = _patched_lower_block
lowering.Lower.lower_inst = _patched_lower_inst


class tagging:
    def __init__(self, mode, scope):
        self.mode, self.scope = mode, scope

    def __enter__(self):
        _TagState.active = True
        _TagState.mode = self.mode
        _TagState.scope = self.scope

    def __exit__(self, *exc):
        _TagState.active = False


# ----------------------------------------------------------------------------
# IR analysis (static, deterministic metric)
# ----------------------------------------------------------------------------

def extract_functions(ir_text):
    """Return {name: body} for every define in the module."""
    funcs = {}
    pat = re.compile(r'^define[^\n]*@"?([^"(\s]+)"?\([^\n]*\{$(.*?)^\}',
                     re.M | re.S)
    for m in pat.finditer(ir_text):
        funcs[m.group(1)] = m.group(2)
    return funcs


def hot_function(ir_text):
    """The jitted function itself (not the cpython/cfunc wrappers)."""
    for name, body in extract_functions(ir_text).items():
        if name.startswith('_ZN8__main__'):
            return name, body
    raise RuntimeError('hot function not found; defines = %s'
                       % list(extract_functions(ir_text)))


def analyze(ir_text):
    name, body = hot_function(ir_text)
    cond_br = len(re.findall(r'^\s*br i1 ', body, re.M))
    uncond_br = len(re.findall(r'^\s*br label ', body, re.M))
    # The inner `range(x)` loop's trip count is %arg.x, so guards derived
    # from arg.x belong to the INNER loop; masks/compares on %arg.arr.2
    # (arr.size) belong to the OUTER loop's own runtime unroll.
    guards = {
        # guard 1: empty-range check (semantically required, not unroll-made)
        'inner_empty_range': bool(
            re.search(r'icmp slt i64 %[^,]*arg\.x[^,]*, 1\b', body)),
        # guard 2: unroll-remainder computation + dispatch (inner, factor 8)
        'inner_remainder_mask7': bool(
            re.search(r'and i64 %[^,]*arg\.x[^,]*, 7\b', body)),
        'xtraiter_eq0_dispatch': bool(
            re.search(r'icmp eq i64 %xtraiter(\.epil)?, 0', body)),
        # guard 3: main unrolled-loop entry (inner trip count >= factor 8)
        'inner_main_entry_lt8': bool(
            re.search(r'icmp [su]lt i64 %[^,]*arg\.x[^,]*, 8\b', body)),
        # outer-loop runtime unroll artifacts (mask on arr.size)
        'outer_remainder_mask': bool(
            re.search(r'and i64 %[^,]*arg\.arr\.2[^,]*, \d+\b', body)),
    }
    guards['any_xtraiter'] = '%xtraiter' in body
    unroll_blocks = len(re.findall(r'^[^\s:]*(?:\.unr|\.epil|\.prol)[^\s:]*:',
                                   body, re.M))
    fadd = len(re.findall(r'\bfadd\b', body))
    # longest run of consecutive fadd lines = observed unroll factor of the
    # straight-line unrolled body
    run = best = 0
    for line in body.splitlines():
        if re.search(r'= fadd\b', line):
            run += 1
            best = max(best, run)
        elif line.strip():
            run = 0
    lines = len(body.strip().splitlines())
    return {
        'function': name,
        'cond_br': cond_br,
        'uncond_br': uncond_br,
        'total_br': cond_br + uncond_br,
        'guards': guards,
        'unroll_named_blocks': unroll_blocks,
        'fadd_count': fadd,
        'max_fadd_run': best,
        'body_lines': lines,
        'norm_hash': norm_hash(body),
    }


def norm_hash(body):
    """Hash of the body with run-varying bits (embedded pointers, metadata
    ids) stripped, to detect 'IR unchanged' across processes."""
    t = re.sub(r'\b\d{8,}\b', '<ADDR>', body)
    t = re.sub(r'!\d+', '!N', t)
    t = re.sub(r',?\s*!llvm\.loop !N', '', t)
    import hashlib
    return hashlib.sha256(t.encode()).hexdigest()[:16]


# ----------------------------------------------------------------------------
# Phase driver
# ----------------------------------------------------------------------------

def compile_all(phase):
    add_for, add_while_uint, add_for_const = make_funcs()
    _TagState.tagged = []
    if phase == 'baseline':
        add_for.compile(SIG2)
        add_while_uint.compile(SIG2)
        add_for_const.compile(SIG1)
    elif phase == 'inner_disable':
        with tagging('disable', 'innermost'):
            add_for.compile(SIG2)
        add_while_uint.compile(SIG2)
        add_for_const.compile(SIG1)
    elif phase == 'all_runtime_disable':
        with tagging('runtime', 'all'):
            add_for.compile(SIG2)
            add_while_uint.compile(SIG2)
            add_for_const.compile(SIG1)
    elif phase == 'all_disable':
        with tagging('disable', 'all'):
            add_for.compile(SIG2)
            add_while_uint.compile(SIG2)
            add_for_const.compile(SIG1)
    else:
        raise ValueError(phase)
    return add_for, add_while_uint, add_for_const


def correctness_check(add_for, add_while_uint, add_for_const):
    a1 = np.zeros(7); a2 = np.zeros(7); a3 = np.zeros(7)
    add_for(a1, 5)
    add_while_uint(a2, 5)
    add_for_const(a3)
    assert np.all(a1 == 5.0) and np.all(a2 == 5.0) and np.all(a3 == 5.0)
    # empty inner loop must still work
    b = np.zeros(3)
    add_for(b, 0)
    add_while_uint(b, 0)
    assert np.all(b == 0.0)
    return True


def main():
    phase = sys.argv[1]
    add_for, add_while_uint, add_for_const = compile_all(phase)
    out = {
        'phase': phase,
        'versions': {
            'numba': numba.__version__,
            'llvmlite': llvmlite.__version__,
            'llvm': '.'.join(map(str, __import__(
                'llvmlite.binding', fromlist=['x']).llvm_version_info)),
            'numpy': np.__version__,
        },
        'tagged_latches': _TagState.tagged,
        'correct': correctness_check(add_for, add_while_uint, add_for_const),
        'results': {
            'add_for': analyze(add_for.inspect_llvm(SIG2)),
            'add_while_uint': analyze(add_while_uint.inspect_llvm(SIG2)),
            'add_for_const': analyze(add_for_const.inspect_llvm(SIG1)),
        },
    }
    ir_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          'exp10670_ir')
    os.makedirs(ir_dir, exist_ok=True)
    for fname, disp, sig in (('add_for', add_for, SIG2),
                             ('add_while_uint', add_while_uint, SIG2),
                             ('add_for_const', add_for_const, SIG1)):
        path = os.path.join(ir_dir, f'{phase}.{fname}.ll')
        with open(path, 'w') as f:
            f.write(disp.inspect_llvm(sig))
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
