# Experiment: targeted `llvm.loop` unroll-disable metadata for issue #10670

**Issue:** [numba#10670](https://github.com/numba/numba/issues/10670) — "`for j in
range()` generates more branches than it needs to".

**Date:** 2026-07-20 · **Branch:** `experiment/10670-unroll-metadata` · No product
code was modified; all experimental code lives in `docs/experiments/` and patches
numba only at runtime (monkeypatch).

## Background (established previously)

- `for j in range(x)` with a **runtime** trip count compiles to a loop that LLVM
  runtime-unrolls (factor 8), adding dispatch/guard branches. On a nested
  100M-element benchmark this measured ~800M retired CPU branches vs ~525M for an
  equivalent `uint64` while-loop.
- The extra branches are **not** numba's range lowering. `numba/cpython/rangeobj.py`
  `iternext` (~line 147, confirmed in this source tree) emits a plain
  `count > 0` / decrement loop with one conditional branch — the guards appear only
  after LLVM's optimizer runs. A plain `while int64` loop shows the same overhead.
- The `uint64` while-loop only avoids the guards by accident: `uint64 + int64`
  type-unifies to `float64` and LLVM won't runtime-unroll a float-counter loop.
- A compile-time-constant `range(5)` is already optimal (fully unrolled).
- Globally disabling unrolling (llvmlite `PipelineTuningOptions.loop_unrolling=False`)
  is a bad trade: it regresses the `uint64` path and never reaches the floor.

## Hypothesis

A **targeted** suppression — attaching `llvm.loop` metadata with an unroll-disable
option to just the relevant loop latch at lowering time — removes the guard
branches for `for j in range(x)` **without** affecting the `uint64` while path or
compile-time-constant ranges.

Two metadata options were tested:

- `llvm.loop.unroll.disable` — disables *all* unrolling of the tagged loop
  (what the runbook proposed).
- `llvm.loop.unroll.runtime.disable` — disables only *runtime* unrolling; full
  unrolling of compile-time-constant trip counts stays allowed. (Added here as
  the production-shaped candidate.)

## Environment

- numba built editable from this source tree (`0+untagged.1885.g63b7565`,
  swap415/numba@63b75658).
- llvmlite **0.49.0dev0**, bundling **LLVM 22.1.0**.
- numpy 2.2.6 (2.5.x fails: this source still references `np.row_stack`,
  removed in numpy 2.5), CPython 3.12.3, Linux x86_64.
- `NUMBA_LOOP_VECTORIZE=0` for every measurement (set before importing numba).

Setup commands:

```bash
uv venv --python 3.12 .venv
# Intended (per runbook):
#   uv pip install --python .venv/bin/python \
#     --extra-index-url https://pypi.anaconda.org/numba/label/dev/simple \
#     --index-strategy unsafe-best-match --prerelease=allow -e . numpy
# DEVIATION: this sandbox's network policy rejects CONNECT to
# pypi.anaconda.org (proxy gateway answers 403), so the dev wheel index was
# unreachable:
#   error: Failed to fetch: https://pypi.anaconda.org/numba/label/dev/simple/setuptools/
#   Caused by: tunnel error: unsuccessful
# conda.anaconda.org IS reachable, and hosts the identical dev build, so the
# same llvmlite 0.49.0dev0 was installed by extracting the conda package:
curl -O https://conda.anaconda.org/numba/label/dev/linux-64/llvmlite-0.49.0dev0-py312hbe36340_0.conda
# (a .conda file is a zip; the inner pkg-*.tar.zst holds lib/python3.12/site-packages/llvmlite)
# extract and copy site-packages/llvmlite + dist-info into .venv/lib/python3.12/site-packages/
uv pip install --python .venv/bin/python "numpy<2.3" setuptools
uv pip install --python .venv/bin/python --no-deps --no-build-isolation -e .
```

This is *not* a fallback to released numba/llvmlite — it is the same
0.49.0dev0 dev build, obtained from the channel's conda packaging instead of
the wheel index. Verified in-process: `llvmlite.__version__ == '0.49.0dev0'`,
`llvm_version_info == (22, 1, 0)`.

Run everything from a directory that does **not** contain a `numba/` folder
(otherwise the source dir shadows the installed package and import fails with
`cannot import name '_typeconv'`).

## Method

Scripts (all in `docs/experiments/`):

- `exp10670_worker.py` — defines the three probe functions, the runtime
  monkeypatch that tags loop latches, and the static IR analysis. Runs one
  phase per process, prints JSON, dumps optimized IR to `exp10670_ir/`.
- `exp10670_run.py` — runs the four phases in fresh subprocesses, writes
  `exp10670_results.json`, prints the comparison table.
- `exp10670_bench.py` — indicative wall-clock only (no PMU in this sandbox).

Probe functions (signatures `(float64[:], int64)` and `(float64[:],)`):

```python
def add_for(arr, x):            # inner `for j in range(x)`, runtime trip count
    for i in range(arr.size):
        for j in range(x):
            arr[i] += 1.0

def add_while_uint(arr, x):     # accidental "fast" variant (float-unified counter)
    for i in range(arr.size):
        j = uint64(0)
        while j < x:
            arr[i] += 1.0
            j += 1

def add_for_const(arr):         # compile-time-constant range(5): already optimal
    for i in range(arr.size):
        for j in range(5):
            arr[i] += 1.0
```

**Metric (static, deterministic — no perf counters available):** for each
function, take `dispatcher.inspect_llvm(sig)`, isolate the jitted function
(`@_ZN8__main__...`, excluding the cpython/cfunc wrappers), and count
conditional (`br i1`) and unconditional (`br label`) branches; detect the
guard patterns; record the longest straight-line run of `fadd`s (observed
unroll factor); and compare a normalized body hash across phases (embedded
runtime pointers and metadata ids stripped) to prove "unchanged".

**Intervention (approach (a) of the runbook — lowering hook, not IR string
post-processing):** `numba.core.lowering.Lower.lower_block`/`lower_inst` are
monkeypatched. Python loops always branch backward, so any `ir.Jump`/`ir.Branch`
whose target offset ≤ the current block's offset is a loop backedge and its
just-emitted LLVM `br` is that loop's latch terminator. The hook attaches
`!llvm.loop` metadata there. Scope is selectable: `innermost` (largest backedge
target = innermost header) or `all` latches.

### Finding: the loop ID node must be self-referential, and llvmlite cannot express that

The first attempt attached `!llvm.loop !{!dummy, !{!"llvm.loop.unroll.disable"}}`.
The metadata verifiably reached LLVM (visible in the pre-optimization module)
yet had **zero effect** — optimized IR byte-identical to baseline. Reason:
`llvm::Loop::getLoopID()` discards a loop-ID node whose first operand is not
the node itself. llvmlite's `module.add_metadata()` cannot create a
self-referential node (its uniquing cache would need to hash a cyclic operand
tuple). Workaround used here:

```python
loop_md = llvmlite.ir.values.MDValue(mod, [opt], name=str(len(mod.metadata)))
loop_md.operands = (loop_md, opt)      # splice in the self-reference
term.set_metadata('llvm.loop', loop_md)
```

which prints as `!5 = !{!5, !3}` (LLVM treats parsed cyclic uniqued nodes as
distinct). With the self-reference in place the metadata took effect
immediately. **Any real implementation needs a small llvmlite API for
self-referential/distinct loop metadata.**

## Results

Phases: `baseline` (no tagging); `inner_disable` (`unroll.disable`, innermost
latch of `add_for` only — the other two functions compiled untouched);
`all_runtime_disable` (`unroll.runtime.disable`, every latch, all three
functions); `all_disable` (`unroll.disable`, every latch — negative control).

All phases pass functional correctness checks (results verified for x=5 and
the empty x=0 loop).

### Branch counts of the jitted function (optimized IR)

| phase | function | `br i1` | total `br` | fadds | max fadd run | unroll blocks |
|---|---|---:|---:|---:|---:|---:|
| baseline | add_for | **17** | 28 | 27 | 8 | 17 |
| baseline | add_while_uint | 11 | 18 | 10 | 2 | 5 |
| baseline | add_for_const | 5 | 7 | 25 | 5 | 3 |
| inner_disable | add_for | **11** | 18 | 5 | 1 | 5 |
| inner_disable | add_while_uint | 11 | 18 | 10 | 2 | 5 |
| inner_disable | add_for_const | 5 | 7 | 25 | 5 | 3 |
| all_runtime_disable | add_for | **4** | 6 | 1 | 1 | 0 |
| all_runtime_disable | add_while_uint | 4 | 6 | 2 | 2 | 0 |
| all_runtime_disable | add_for_const | 2 | 3 | 5 | **5** | 0 |
| all_disable | add_for | 4 | 6 | 1 | 1 | 0 |
| all_disable | add_while_uint | 4 | 6 | 2 | 2 | 0 |
| all_disable | add_for_const | 3 | 5 | 1 | **1** ⚠ | 0 |

"max fadd run" = longest straight-line sequence of `fadd`s ≈ observed unroll
factor. "unroll blocks" = basic blocks named `*.unr`/`*.prol`/`*.epil`.

### Guard branches in `add_for`

Baseline (LLVM 22.1.0) reproduces the reported guard structure, with two
updates versus the original (older-LLVM) observation: the main-entry compare
is emitted as `slt %x, 8` rather than `ult`, and LLVM 22 *additionally*
runtime-unrolls the **outer** loop (×2 here, ×4 in the other functions), which
duplicates the inner guards per unrolled outer body:

```llvm
; baseline add_for — inner-loop runtime-unroll guards
  %.230126.not = icmp slt i64 %arg.x, 1          ; guard 1: empty range
  ...
  %xtraiter = and i64 %.arg.x, 7                 ; guard 2: remainder count
  %1 = icmp eq i64 %xtraiter, 0                  ;          remainder dispatch
  br i1 %1, label %B78.us.prol.loopexit, label %B78.us.prol.preheader
  ...
B78.us.prol:                                     ; remainder (prologue) loop
  ...
  %4 = icmp slt i64 %arg.x, 8                    ; guard 3: main x8 loop entry
  br i1 %4, label %B74...crit_edge.us, label %B78.us.preheader
B78.us:                                          ; main body: 8 fadds per branch
```

After `all_runtime_disable`, the entire nest collapses to the minimal form —
the only surviving guard is the semantically required empty-range check, and
LLVM hoists even that out of the outer loop (checked once per call, not per
element):

```llvm
; all_runtime_disable add_for — complete hot path
B0.endif:
  %.109137.not = icmp slt i64 %arg.arr.2, 1      ; arr empty?
  br i1 %.109137.not, label %B112, label %B50.endif.lr.ph
B50.endif.lr.ph:
  %.230126.not = icmp slt i64 %arg.x, 1          ; range empty? (hoisted, 1x per call)
  br i1 %.230126.not, label %B112, label %B50.endif.us.preheader
B78.us:                                          ; inner loop: 1 branch / iteration
  %.300.us = fadd double %.300130.us, 1.000000e+00
  %.230.us = icmp sgt i64 %lsr.iv.next, 1
  br i1 %.230.us, label %B78.us, label %B74.B46.loopexit_crit_edge.us, !llvm.loop !0
B74.B46.loopexit_crit_edge.us:                   ; outer latch: 1 branch / element
  %exitcond.not = icmp eq i64 %.122.us, %arg.arr.2
  br i1 %exitcond.not, label %B112, label %B50.endif.us, !llvm.loop !2
```

`xtraiter`, the remainder dispatch, the `slt 8` main-entry guard, and all
`*.prol`/`*.epil`/`*.unr` blocks are gone.

### Success criteria

1. **`add_for` guards vanish — yes.** In `inner_disable`, the inner-loop
   unroll guards (`and %arg.x, 7`, its `xtraiter` dispatch, `slt %arg.x, 8`)
   disappear and `br i1` drops 17 → 11 (the remaining unroll artifacts belong
   to the outer loop, which that phase deliberately left untagged). In
   `all_runtime_disable`, `br i1` drops 17 → 4 and the only remaining guard is
   the required empty-range check.
2. **`add_while_uint` and `add_for_const` unchanged — yes, in the targeted
   phase.** In `inner_disable` both control functions are byte-identical to
   baseline (normalized-hash equal). In `all_runtime_disable` they change only
   in that their *outer* loops also stop being runtime-unrolled (11 → 4 and
   5 → 2 `br i1`) — the same intended effect, applied policy-wide; their hot
   inner bodies are unchanged (`add_while_uint`: same 2-fadd/`fcmp`/`br`
   iteration; `add_for_const`: full ×5 unroll intact, max fadd run still 5).
3. **Does it reach the `uint64` level, not just the un-unrolled level —
   reaches it, and slightly beats it.** Per inner iteration, tagged `add_for`
   executes 1 conditional branch (`fadd`/`icmp`/`br`), identical to the
   `uint64` while-loop's hot block (which spends an extra `fadd` maintaining
   its float counter). Statically, tagged `add_for` (4 × `br i1`) matches
   tagged `add_while_uint` (4) and beats baseline `add_while_uint` (11).
   The runbook's "~6 branches/iter un-unrolled level" does not appear:
   suppressing runtime unrolling still leaves LSR/rotation to reduce the
   range loop to a single-branch countdown.
4. **Negative control confirms the knob choice.** With blanket
   `llvm.loop.unroll.disable`, the constant `range(5)` loop **loses its full
   unroll** (max fadd run 5 → 1) — exactly the regression the issue wants to
   avoid. `llvm.loop.unroll.runtime.disable` keeps it (run stays 5). The
   right lever is `runtime.disable`.

### Wall-clock (indicative only)

20M-element array, x=5 (≈100M inner iterations), best of 7 runs, no PMU
access in this sandbox:

| | add_for | add_while_uint | add_for_const |
|---|---:|---:|---:|
| baseline | 82.9 ms | 78.9 ms | 43.4 ms |
| tagged (runtime.disable all) | 83.9 ms | 82.7 ms | 43.7 ms |

Differences are within run-to-run noise; this workload is memory-bound
(streaming 160 MB), so the branch reduction is not expected to show in wall
time here. The original issue's 800M-vs-525M figures are *retired-branch
counter* measurements and must be re-validated with PMUs (see next steps).

## Verdict

**The hypothesis is confirmed, with one refinement.** Targeted `llvm.loop`
unroll metadata attached at numba's lowering stage cleanly removes the
runtime-unroll guard branches of `for j in range(x)` loops:

- It is precise: tagging only the inner loop changed only the inner loop;
  untagged functions were byte-identical.
- The refinement: use **`llvm.loop.unroll.runtime.disable`**, not
  `llvm.loop.unroll.disable`. Runtime-disable removes every guard the issue
  complains about while preserving full unrolling of compile-time-constant
  trip counts; full disable regresses the constant case.
- With the metadata, the `for range(x)` loop reaches (statically, slightly
  beats) the `uint64` while-loop's branch structure — the accidental
  workaround becomes unnecessary.
- Implementation cost is small: numba already knows every loop backedge at
  lowering time; the only missing piece is an llvmlite API for
  self-referential (distinct) loop-ID metadata, since `Loop::getLoopID()`
  silently ignores non-self-referential nodes (this experiment's first
  attempt failed exactly there).

Caveats: whether suppressing runtime unrolling ever *hurts* (large trip
counts where ×8 unrolling amortizes loop overhead; interaction with
vectorization — all measurements here used `NUMBA_LOOP_VECTORIZE=0`, and
`llvm.loop` nodes also carry vectorizer hints) is not answered by the static
metric and needs hardware measurement.

## Next steps

1. **PMU validation** on a Linux box with perf-counter access
   (`kernel.perf_event_paranoid <= 1`; e.g. `py-perf-event` or `perf stat`):
   confirm retired-branch counts drop toward the 525M floor on the original
   100M-element nested benchmark, and quantify wall-time effects on
   compute-bound (not memory-bound) variants.
2. **Trip-count sweep** (x = 1…1000) with counters, to check the large-x
   regime where runtime unrolling might genuinely pay for itself; decide
   between always-on `runtime.disable` for range loops vs a heuristic or an
   `@njit` option.
3. **Vectorization interaction**: repeat with `NUMBA_LOOP_VECTORIZE=1`; make
   sure the emitted loop ID composes with (does not clobber) vectorizer
   metadata numba/LLVM may attach.
4. **Upstream llvmlite**: propose a small API for distinct/self-referential
   loop metadata (e.g. `builder.set_loop_metadata(term, ["llvm.loop.unroll.runtime.disable"])`),
   which this experiment had to hand-roll via `MDValue` operand splicing.
5. If 1–3 hold up, prototype the real numba change: emit the metadata on
   range-loop latches in lowering (the backedge is already explicit in numba
   IR), behind a config flag for A/B testing.

## Artifacts

- `exp10670_worker.py`, `exp10670_run.py`, `exp10670_bench.py` — scripts.
- `exp10670_results.json` — full per-phase measurements.
- `exp10670_ir/<phase>.<function>.ll` — complete optimized IR dumps for every
  phase × function.

Reproduce with:

```bash
cd docs/experiments && /path/to/.venv/bin/python exp10670_run.py
```
