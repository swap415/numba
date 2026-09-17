# Capstone inspection

Problem: inspect compiled CPU instructions through an optional Python dependency.
Expected behavior: `dispatcher.inspect_disasm(signature)` returns disassembly;
omitting the signature returns a dictionary, following existing inspection APIs.
Scope: text sections of the compiled object. Preserve `inspect_asm` and the
radare2-based `inspect_disasm_cfg`; do not add profiling, CFG recovery, source
mapping, or live-memory inspection.

Baseline: `0fac3436b548976a6d5fed32473e1f6099c85ee3`.
Branch: `feature/capstone-inspection`. Patch revision: uncommitted.
Investigation date: 2026-09-14. Host: macOS arm64.
Run commands from repository root, `/Users/mac357/dev/numba`, unless stated.
This directory is a local investigation record, separate from the upstream patch.

## 01 — Establish precedent

Hypothesis: existing inspection tools establish a small integration boundary.
Commands: `git log --all --oneline --grep='capstone\|disassem\|inspect_asm\|inspect_cfg' -i`
and `gh api 'search/issues?q=repo:numba/numba+capstone&per_page=100'`.
Read `numba/misc/inspection.py`, `numba/core/codegen.py`, dispatcher inspection
methods, their documentation, and the discussions below.

| Evidence | Relationship and consequence |
| --- | --- |
| [Numba #5028](https://github.com/numba/numba/issues/5028) | Earlier Capstone prototype mapped JIT ELF through assembly and IR to Python source for profiling. Related research, not a reusable released integration. Label: `feature_request`; not `good first issue`. |
| [PR #5212 meeting decision](https://github.com/numba/numba/pull/5212#issuecomment-584756226) | Maintainers requested inspection implementation in `numba.misc` and installation documentation. |
| [PR #5212 review](https://github.com/numba/numba/pull/5212#pullrequestreview-371745451) | Reviewer could not readily discover the dispatcher method; include a small runnable reference example. |
| [PR #7315](https://github.com/numba/numba/pull/7315) | Linking objects as DSOs improved radare2's relocation and DWARF analysis. Capstone decoding alone does not replace these features. |
| [PR #7315 follow-up](https://github.com/numba/numba/pull/7315#issuecomment-976627540) | Long symbols exposed radare2's 61-character matching limit. Avoid heuristic symbol matching. Local follow-up commit: `9b33a7ecf159c73c3080da44461b70476b38ee12`. |
| [PR #7074 → #7116](https://github.com/numba/numba/pull/7074#issuecomment-862415476) | DWARF language change moved into a tested successor; the abandoned label did not reject the design. Review considered both GDB and radare2. |
| [llvmlite object interface](https://llvmlite.readthedocs.io/en/latest/user-guide/binding/object-file.html) | Reuse `ObjectFileRef.from_data` and section data/address/type access instead of introducing an object parser. |
| [Capstone Python API](https://www.capstone-engine.org/lang_python.html) | `disasm_lite` yields address, size, mnemonic, and operands. Decoding stops at invalid instructions; partial output must be visible. |
| [Capstone releases](https://www.capstone-engine.org/) | Website lists stable 5.0.9 and prerelease 6.0.0-Alpha9. Use a stable version for initial verification. |

Observed: GitHub issue searches returned one Capstone result in Numba, zero in
llvmlite; CPython code and issue searches returned zero. This is search evidence,
not proof that no other prototype exists. None of the relevant issues inspected
has a `good first issue` label. GitHub API access initially failed in the sandbox;
approved read-only access succeeded. Web search alone missed #5028.
Conclusion: add an optional helper in `misc`, following dispatcher conventions.

## 02 — Establish a runnable baseline

Hypothesis: an existing environment can import and compile this checkout.
Rerun command: `python -c 'import numba, llvmlite; print(numba.__version__, llvmlite.__version__)'`.
Expected: successful import before attempting disassembly. On dependency failure,
find the supported development wheel source before changing code.

Reported observations: default environment has llvmlite 0.48; the baseline
requires at least 0.50. Local `.venv` has 0.46; the local llvmlite checkout is
0.47. None satisfies this checkout. Running inside the `numba/` package also
causes its `types` package to shadow Python's standard-library module.
Conclusion: run from repository root and create a compatible test environment.
Default import error: `ImportError: Numba requires at least version 0.50.0 of
llvmlite. Installed version is 0.48.0dev0. Please update llvmlite.`
The local checkout probe used `PYTHONPATH=/Users/mac357/dev/llvmlite` and
reported llvmlite `0.47.0dev0-55-g4fcfc39e`, LLVM `20.1.8` from repository root.
Running that probe inside the package first failed with a partially initialized
`enum` module because local `types` shadowed the standard library.

## 03 — Resolve development dependencies

Hypothesis: Numba's own wheel workflow identifies the correct development index.
Command: `rg -n 'pypi.anaconda.org|label/dev' .github/workflows/numba_osx-arm64_wheel_builder.yml`.
Observed: following the user's direction, workflow lines 28 and 105 identified
`https://pypi.anaconda.org/numba/label/dev/simple`.
Environment setup completed with
`conda create -y -n numba-capstone python=3.13 numpy pip cffi jinja2 setuptools`.
Use `conda run -n numba-capstone` for subsequent commands, as requested by the
user. An unused uv `.venv` was created before this preference was clarified;
it is excluded from the patch. Install llvmlite using pip's `--pre` option and
`-i https://pypi.anaconda.org/numba/label/dev/simple` in this environment.
The first install request was interrupted; `conda run -n numba-capstone python
-m pip show llvmlite capstone` confirmed neither package had been installed.
Completed commands, from repository root:

```sh
conda run --no-capture-output -n numba-capstone python -m pip install --pre -i https://pypi.anaconda.org/numba/label/dev/simple 'llvmlite>=0.50.0dev0,<0.51'
conda run --no-capture-output -n numba-capstone python -m pip install 'capstone>=5,<6'
```

Verified checkout import path: `/Users/mac357/dev/numba/numba/__init__.py`.
Python 3.13.15; Numba `0.67.0dev1+389.g0fac3436b`; llvmlite `0.50.0rc3`;
NumPy 2.5.2. Installed Capstone distribution: 5.0.9; its Python binding reports
5.0.7. Do not confuse the binding's stale version string with pip's package version.

## 04 — Reproduce and isolate

Hypothesis: existing compiled-object access plus llvmlite's text-section reader
is sufficient; a new binary parser is unnecessary.

```sh
conda run --no-capture-output -n numba-capstone python -m numba.investigations.capstone.01_repro
conda run --no-capture-output -n numba-capstone python -m numba.investigations.capstone.02_sections
conda run --no-capture-output -n numba-capstone python -m numba.runtests numba.tests.test_inspection
```

Before implementation, `01_repro` failed with `AttributeError: 'CPUDispatcher'
object has no attribute 'inspect_disasm'`; `02_sections` passed: `__text`,
328 bytes, 82 instructions. Regression baseline: 7 tests, 13 missing-API errors
(including seven architecture/format subcases), 0.063 seconds.
After implementation, `01_repro` passed; 7 regression tests passed in 0.074 seconds.
These are test-run durations, not performance claims.

## 05 — Challenge decoding assumptions

```sh
conda run --no-capture-output -n numba-capstone python -m numba.investigations.capstone.03_decode_cases
conda run --no-capture-output -n numba-capstone python -m numba.investigations.capstone.04_cross_target
```

Both scripts assert complete text-byte consumption. Native ARM64 trig,
conditional reduction, and parallel reduction: 360/360, 460/460, 2124/2124 bytes.
Constant/switch cases: x86 ELF 15/15 and 49/49; AArch64 ELF 24/24 and 40/40;
arm64 Mach-O 24/24 and 32/32; x86 COFF 15/15 and 55/55.
LLVM can emit empty `.text` beside populated `.ltext`; use `is_text()`.
Direct script-path execution initially could not import the uninstalled checkout;
module execution from repository root works without modifying scripts' `sys.path`.

Code inspection corrected an initial cache assumption: `_object_getbuffer_hook`
clears `_compiled_object` on loading. The integration explicitly rejects cached
code instead of changing cache retention. A serialization/restoration test
verifies this limitation. These probes support the selected scope, not all
instructions or native execution on other platforms.

## 06 — Verify surrounding behavior

```sh
PYTHONPATH=/Users/mac357/dev/numba conda run --no-capture-output -n numba-capstone python -m numba.runtests numba.tests.test_inspection numba.tests.test_dispatcher numba.tests.test_codegen numba.tests.test_caching
```

Initial run without `PYTHONPATH`, inside the sandbox: 106 tests, 2 failures,
3 errors, 5 skips, 1 expected failure. Cache fallback/zip tests lacked permission
to write `/Users/mac357/Library/Caches/numba`; a subprocess reported
`ModuleNotFoundError: No module named 'numba'`. Rerunning the command above with
approved cache access passed: 106 tests, 5 skips, 1 expected failure, 14.773 seconds.
No implementation changes were needed for these environment failures.

Lint initially found one new line exceeding 80 columns, now wrapped. Explicitly
passing `.pyi` to Flake8 also produced existing stub-style errors; normal CI
discovery does not select stubs. Pyrefly initially lacked `types-cffi`; install
the project's requirements before interpreting that result as a code defect.

Final check commands (repository root):

```sh
conda run --no-capture-output -n numba-capstone python -m pip install -r maint/requirements_pyrefly.txt -r maint/stubtest/requirements_stubtest.txt
conda run --no-capture-output -n numba-capstone flake8 --extend-exclude=.venv,.claude numba
conda run --no-capture-output -n numba-capstone pyrefly check --no-progress-bar --summary=full
PYTHONPATH=/Users/mac357/dev/numba conda run --no-capture-output -n numba-capstone python maint/stubtest.py
```

Flake8 passes after excluding local environments and correcting one experiment's
continuation indent. Its initial recursive scan entered existing `.venv`
dependencies; those diagnostics were not project source failures. Pyrefly 1.3.0:
0 errors (16 suppressed, 1 warning not shown). Stubtest: no issues in 752 modules;
its initial sandbox run could not open the mypy cache database, so the successful
rerun used approved cache access. Final focused tests: 7 passed in 0.242 seconds,
including JIT compilation and `inspect_asm` while Capstone is unavailable.
`git diff --check` passes. Independent final review found no blocking issues.

## Decision and handoff

No change leaves LLVM assembly and radare2 CFG inspection available. Reusing the
existing compiled-object cache and llvmlite section reader adds Capstone with a
smaller dependency and maintenance surface than a new object parser. The chosen
first slice is optional `inspect_disasm` section-level output with lazy dependency
loading, public stubs, reference/installation docs, and regression tests.

Failure modes to verify: missing Capstone, unsupported target architecture,
incorrect endianness or mode, undecodable bytes, empty text sections, multiple
signatures, and cached compilations. Default Numba import and compilation must
continue without Capstone. Existing inspection behavior must remain intact.

Object sections can include wrappers, helper functions, padding, and embedded
data. Their addresses and operands describe a relocatable object, not live JIT
addresses. Function boundaries, symbol resolution, relocation application, DWARF,
and CFG reconstruction are outside this slice. Architecture support must be
explicit; local arm64 success cannot establish x86 or other platform support.

Lint/type/stub checks pass. No performance claim is made, so no benchmark is
needed. Native execution was tested on macOS ARM64; other targets
were tested by emitting and decoding objects, not by executing those objects.

Follow the [Numba AI policy](https://numba.readthedocs.io/en/stable/reference/ai_tools_policy.html):
human review before sharing, preserve the PR template's AI declaration, and
attribute assistance, for example `Assisted-by: Codex`.
