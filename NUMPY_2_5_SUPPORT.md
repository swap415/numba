# NumPy 2.5 support for Numba

Working notes for the `numpy2.5-support` branch: every change in the
[NumPy 2.5.0rc1 release](https://github.com/numpy/numpy/releases/tag/v2.5.0rc1),
its implication for Numba, and how it needs to be handled here.

## Environment used for this work

- **NumPy** `2.5.0rc1` (`pip install --pre`). NumPy 2.5 **dropped Python 3.11**
  (`requires-python >=3.12`; wheels are cp312/cp313/cp314 only), so this branch
  is built and tested under **Python 3.12**.
- **llvmlite** `0.48.0rc1` (`--pre`). Numba `main` already targets
  `llvmlite >=0.48.0dev0,<0.49`, so no pin change is needed.
- Numba already version-gates a lot of NumPy behaviour via
  `numba.np.numpy_support.numpy_version`; the same mechanism is used for all
  source/test changes below.
- **Caveat:** `scipy` was not installed in the test venv, so the BLAS/LAPACK
  paths in `test_linalg` (incl. `eig`/`eigvals`) were **skipped**. The
  `linalg.eig` item below is from source analysis + NumPy behaviour and still
  needs validation with `scipy` present.

## Status legend

- ✅ **Done** — handled on this branch.
- 🔧 **TODO (source)** — needs a change to Numba's implementation to match 2.5.
- 🧪 **TODO (test)** — only test adaptation needed.
- 🔎 **Investigate** — needs verification / decision.
- ➖ **No impact** — does not affect Numba (with reason).

---

## 1. Expired deprecations / removals (BREAKING)

These remove symbols/behaviour outright and break at import or call time.

| # | NumPy change (gh) | Numba impact | Action | Status |
|---|---|---|---|---|
| 1 | `numpy.row_stack` removed (gh-30463) | `arrayobj.py` registered `overload(np.row_stack)` unconditionally → `AttributeError` at `import numba`. Tests in `test_dyn_array` also call it. | Cap the overload registration to `(2,0) <= ver < (2,5)`; gate the `np_row_stack` usecase in `test_dyn_array.test_vstack`. | ✅ Done |
| 2 | `numpy.cross` drops 2D vectors (gh-30461) | Numba's `np.cross` accepted 2-element vectors (implicit z=0); NumPy now raises. | On `ver >= (2,5)`, `np_cross` rejects 2-element inputs with `ValueError` + `cross2d` hint, matching NumPy. `test_cross*` updated. | ✅ Done |
| 3 | Data type alias `'a'` removed (gh-30613) | `np.dtype('a11')` is unconstructible. Only used in `test_numpy_support.test_string_types`; Numba source emits `'S'`, never `'a'`. | Gate the `'a11'` round-trip behind `ver < (2,5)` (`'S'` form still covered). | ✅ Done |
| 4 | `bincount` raises `TypeError` for non-integer (gh-30610) | Numba's `np_bincount` already returns no impl for non-integer dtype (→ `TypingError`), i.e. already rejects non-integer input. | Confirm parity; no change required. Optionally add a `ver >= (2,5)` test asserting rejection. | ➖ No impact (verify) |
| 5 | `maximum_sctype` removed (gh-30462) | No references in Numba source/tests. | None. | ➖ No impact |
| 6 | `get_array_wrap` removed (gh-30463) | No references in Numba. | None. | ➖ No impact |
| 7 | `numpy.chararray` re-export removed (gh-30604) | No `chararray` / `np.char.*` usage in Numba source. | None. | ➖ No impact |
| 8 | `recfromtxt`/`recfromcsv` removed (gh-30467) | File IO; not implemented in nopython. | None. | ➖ No impact |
| 9 | `numpy.lib.math` alias removed (gh-30612) | Not imported by Numba. | None. | ➖ No impact |
| 10 | `_add_newdoc_ufunc` removed (gh-30614) | Not used by Numba. | None. | ➖ No impact |
| 11 | `np.finfo(None)` → `TypeError` (gh-30460) | Numba only calls `np.finfo` with explicit dtypes (`mathimpl`, `arraymath`); it also *overloads* `np.finfo`. Never called with `None`. | None. | ➖ No impact |
| 12 | `numpy.distutils` removed (gh-30340) | `setup.py` / C-ext build does **not** use `numpy.distutils`. The only reference is a string in a `pycc/platform.py` error message (`pycc` is itself deprecated). | None for the build; optionally clean up the stale message. | ➖ No impact (verify) |

---

## 2. Compatibility notes (behaviour changes Numba reimplements)

| # | NumPy change (gh) | Numba impact | Action | Status |
|---|---|---|---|---|
| 1 | `linalg.eig` / `eigvals` **always return complex** (gh-30411) | **Significant.** Numba's `eig_impl`/`eigvals_impl` use `real_eig_impl` for real input: they return **real** arrays when all eigenvalues are real, and **raise** when any eigenvalue is complex (Numba can't change return type at runtime). NumPy 2.5 removes the dynamic typing — eig is now statically complex. | On `ver >= (2,5)`, make Numba's real-input `eig`/`eigvals` **always return complex** arrays (assemble `wr + 1j*wi`, complex eigenvectors). This both matches NumPy 2.5 **and** removes the long-standing "raises on complex eigenvalues of a real matrix" limitation. Update `test_linalg` accordingly. | 🔧 TODO (source) |
| 2 | BTPE binomial Stirling-series fix (gh-31238); `Generator.binomial` / `Generator.multinomial` streams change. Legacy `RandomState` is intentionally **unchanged**. | Numba's `np/random/distributions.py::random_binomial_btpe` is a port of NumPy's *old* (buggy) BTPE and drives `Generator.binomial` (Numba does **not** implement `Generator.multinomial`). The Stirling acceptance test (case 52, ~L682–691) shows the inherited bug: the `w` term divides by `66320.` while the other three use `166320.`; per gh-31238 the leading coefficient should be `13860` (not `13680`) and the 3rd/4th terms' signs are corrected. Legacy `np.random.binomial` lives separately in `cpython/randomimpl.py` and must stay as-is. | On `ver >= (2,5)`, reconcile `random_binomial_btpe` with NumPy 2.5's corrected Stirling series so `Generator.binomial` matches NumPy for the same `BitGenerator` seed. **Leave the legacy path untouched.** Current spot-checks pass (the error only bites specific `n`), so add a targeted parity test for an affected `n`. | 🔧 TODO (source) + 🔎 |
| 3 | `datetime64`/`timedelta64` overflow → `OverflowError` (gh-31378) | Numba deliberately uses C wraparound for **all** integer (incl. timedelta) arithmetic and does no overflow checking (performance); this is a documented, pervasive divergence, not datetime-specific. Object-mode tests defer to NumPy and now see the raise. | Keep Numba's wraparound semantics. Tests: skip the overflowing `astype` in `test_comparisons`; only run the overflow-wraparound `test_mul` case where it still applies (nopython still wraps). Document the divergence. | ✅ Done (test) — design decision: do **not** add overflow checks |
| 4 | `np.where` no longer truncates Python ints → `OverflowError` (gh-30803) | Numba's `np.where` overloads operate on typed values; out-of-range Python-int truncation isn't really expressible the same way. Low risk. | Verify `np.where` with scalar branches on 2.5; adapt a test only if one trips. | 🔎 Investigate |
| 5 | `from_dlpack` raises `BufferError` (was `RuntimeError`) (gh-30937) | Affects Numba's DLPack interop (CUDA / `__dlpack__`). Error-type only. | Check Numba's dlpack import paths/tests for `RuntimeError` expectations; relax to `BufferError` where 2.5 is in play (likely CUDA-only, untested here). | 🔎 Investigate (CUDA) |
| 6 | Default memory allocator → `PyMem_RawMalloc/Free` (gh-30846, gh-31503) | NumPy arrays handed to Numba use a different allocator; Numba's NRT manages its own memory. Should be transparent. | None expected; watch for NRT/`tracemalloc`-related test assumptions. | ➖ No impact (verify) |
| 7 | MSVC ≥ 19.35 required (gh-30489) | Windows build toolchain only. | Ensure Windows CI uses VS 2022 ≥ 17.5; no code change. | ➖ Build infra |
| 8 | Cython ≥ 3.0 required (gh-30770) | Numba does not use Cython. | None. | ➖ No impact |

---

## 3. Deprecations (warnings now, errors later — fix before they expire)

| # | NumPy change (gh) | Numba impact | Action | Status |
|---|---|---|---|---|
| 1 | `generic` unit for `timedelta64` deprecated; incl. implicit bare-int conversion (gh-29619) | **Broad.** Numba's datetime/timedelta support models the unit-less ("generic") `timedelta64`/`datetime64`, and the test suite uses `TD(1)` / `DT(0)` (no unit) heavily. Running on 2.5 emits many `DeprecationWarning`s; the unit will eventually become an error. | Audit `np/npdatetime*.py` and datetime tests for unit-less construction. Short term: ensure warnings don't fail tests. Medium term: move tests to explicit units and decide whether Numba keeps modelling the generic unit (it must while NumPy still supports it for back-compat). | 🔎 Investigate (broad) |
| 2 | Non-integer inputs to `tri`/`triu_indices`/`tril_indices`(+`_from`) deprecated (gh-30869) | Numba reimplements `np.tri`, `np.tril_indices(_from)`, `np.triu_indices(_from)`. If Numba's impls accept/produce float `N/M/k`, they should mirror the deprecation. | Verify Numba's impls require integer `N/M/k`; align with NumPy (reject/deprecate non-integer). Also pick up "**`triu_indices` now accepts unsigned ints**" (Changes §5). | 🔎 Investigate |
| 3 | `numpy.take` casting-rule fix for `out=` (gh-30615) | Numba's `np.take` (`numpy_take(a, indices, axis=None)`) has **no `out=`**, so the rule change is moot. | None. | ➖ No impact |
| 4 | `numpy.fix` deprecated in favour of `numpy.trunc` (gh-30644) | No `overload(np.fix)` in Numba source. Tests may call `np.fix`. | None for source; gate any test that calls `np.fix` on 2.5 if it warns. | ➖ No impact (verify) |
| 5 | Setting `dtype` attribute deprecated (gh-29244) | Can't set `.dtype` in nopython; host-side Numba code doesn't do `arr.dtype = ...`. | None. | ➖ No impact |
| 6 | Setting `shape` attribute / in-place `resize` deprecated (gh-29536, gh-30181) | Numba doesn't expose `arr.shape = ...` assignment or in-place `resize` in nopython. | None. | ➖ No impact |
| 7 | `numpy.char.chararray` / `numpy.char.[as]array` deprecated (gh-30605, gh-30802) | No `np.char` usage in Numba. | None. | ➖ No impact |
| 8 | `numpy.ma.round_` deprecated (gh-30738) | Masked arrays unsupported in nopython. | None. | ➖ No impact |
| 9 | `numpy.typename` deprecated (gh-30774) | Not used by Numba. | None. | ➖ No impact |
| 10 | Custom dtype property / `__array_finalize__` deprecations (gh-31293) | Numba's host arrays are plain `ndarray`s; no `arr.view(dtype=...)` subclass machinery. | None. | ➖ No impact |

---

## 4. New features (signature / feature gaps — adopt to "match more of NumPy")

| # | NumPy change (gh) | Numba impact | Action | Status |
|---|---|---|---|---|
| 1 | `descending=True` for `np.sort` / `np.argsort` (gh-31345) | Numba's `impl_np_sort(a)` takes only `a`; `argsort` supports `kind` but not `descending`. Signature now lags NumPy. | Add a `descending` keyword to Numba's `np.sort` / `np.argsort` (and `ndarray.sort`/`argsort`) overloads, NaNs-to-end in both directions, to match 2.5. | 🔧 TODO (source, optional) |
| 2 | N-D polynomial eval: `polyvalnd`, `chebvalnd`, `legvalnd`, `hermvalnd`, `hermevalnd`, `lagvalnd` (gh-30857) | New functions Numba doesn't overload — pure feature gap. | Optionally add `@overload`s in `np/polynomial`. Not required for compatibility. | 🔧 TODO (source, optional) |
| 3 | `register_dlpack_dtype` for user dtypes (gh-31256) | Optional interop feature. | None required. | ➖ Optional |
| 4 | `ndarray` structural pattern matching (`Py_TPFLAGS_SEQUENCE`) (gh-30653) | Numba doesn't lower `match`/`case` over arrays. | None. | ➖ No impact |
| 5 | Pixi package definitions (gh-30381) | NumPy's own build infra. | None. | ➖ No impact |

---

## 5. C API changes

| # | NumPy change (gh) | Numba impact | Action | Status |
|---|---|---|---|---|
| 1 | New dtype accessors `PyDataType_TYPE/KIND/BYTEORDER/TYPEOBJ` (gh-30994) | Additive. Numba's C extensions already use accessor macros for the (opaque since 1.20) `PyArray_Descr`. | None required; optionally adopt the new macros. | ➖ No impact |
| 2 | `PyArray_DescrFromScalar` keeps parametric params (gh-31067) | Numba doesn't depend on the old parameter-discarding behaviour. | None. | ➖ No impact |
| 3 | `"real"`/`"imag"` ArrayMethods registration (gh-30984) | Numba doesn't register NumPy ArrayMethods. | None. | ➖ No impact |
| 4 | Free-threaded stable ABI (PEP 803) (gh-31091) | Relevant to a future free-threaded Numba build, not this branch. | Track separately. | ➖ Future |

---

## 6. Performance / internal changes (mostly informational)

| # | NumPy change (gh) | Numba impact | Action | Status |
|---|---|---|---|---|
| 1 | **`searchsorted` batched binary search, up to 20x** (gh-30517) | **Root cause** of Numba's `searchsorted` test divergence. For **sorted** input (the only behaviour NumPy *defines*) Numba still matches bit-for-bit (0 divergences in fuzzing). NumPy changed only the traversal for **unsorted** input, which NumPy documents as undefined. | No source change: Numba's `searchsorted` is correct for defined inputs. Tests only cross-check unsorted `a` against NumPy on `ver < (2,5)`; sorted input is always validated. | ✅ Done (test) |
| 2 | `np.sign` timedelta loop changed `m->m` → `m->d` (float64, NaT→NaN) — *(listed in 2.5 as part of ufunc/typing work; surfaced via the loop-types tests)* | Numba registered only `ufunc_db[np.sign]['m->m']` (returns timedelta64); the new `m->d` loop was unimplemented → typing error. | Added `npdatetime.timedelta_sign_to_float_impl` and registered `ufunc_db[np.sign]['m->d']` for `ver >= (2,5)`. Numba now matches NumPy for arrays and scalars (incl. NaT). | ✅ Done (source) |
| 3 | Structured-array `memcpy` copy; padding bytes may be copied; `NPY_NOT_TRIVIALLY_COPYABLE` (gh-29270) | Could affect record-array equality assertions that previously ignored padding. `test_record_dtype`/`test_recarray_usecases` passed on this branch. | Watch record-dtype tests; no change needed currently. | ➖ No impact (verify) |
| 4 | `meshgrid` always returns tuple (gh-30707) | Numba does not overload `np.meshgrid`. | None. | ➖ No impact |
| 5 | `ctypeslib.as_ctypes` scalar restriction (gh-30538) | Not on a Numba nopython path. | None. | ➖ No impact |
| 6 | `__array_interface__` on scalars now read-only (gh-30538) | Numba reads array interface from objects, not scalar `__ref`. | None expected; verify if a buffer/array-adaptor test trips. | ➖ No impact (verify) |
| 7 | `.real`/`.imag` for object dtype (gh-30984) | Object-dtype arrays unsupported in nopython. | None. | ➖ No impact |
| 8 | Faster contiguous reductions; free-threading ufunc scaling (gh-31274, gh-30846) | NumPy-internal; Numba has its own reductions/ufunc machinery. | None. | ➖ No impact |
| 9 | Static-typing (`.pyi`) improvements: `linalg`, `ma`, `fft`, shape-typing (gh-30480, gh-30566, gh-31172, gh-31226) | Numba doesn't consume NumPy's type stubs at runtime. | None. | ➖ No impact |
| 10 | f2py `intent(inplace)` / allocatable changes (gh-29929, gh-30965) | Numba doesn't use f2py. | None. | ➖ No impact |

---

## Summary

**Already handled on this branch (✅):**
- `np.row_stack` removal (source + tests)
- `np.cross` 2D removal (source + tests)
- `np.sign(timedelta64)` `m->d` loop (source + tests)
- `'a'` dtype alias removal (test)
- `searchsorted` undefined-unsorted divergence (test)
- datetime/timedelta overflow now raising (test; deliberate keep-wraparound)

**Remaining genuine compatibility work (🔧):**
1. **`linalg.eig`/`eigvals` → always complex** on `ver >= (2,5)` (highest value: also lifts a real-matrix limitation).
2. **`Generator.binomial` BTPE Stirling-series fix** to match NumPy 2.5's corrected stream (leave legacy `RandomState` untouched).

**Optional feature parity (🔧, not required to pass):**
- `descending=` for `np.sort`/`np.argsort`.
- N-D polynomial eval functions.

**To investigate (🔎):**
- `timedelta64` generic-unit deprecation warnings across datetime tests (broad, future-proofing).
- `tri`/`triu_indices`/`tril_indices` non-integer deprecation + unsigned-int support.
- `np.where` Python-int overflow, `from_dlpack` `BufferError` (CUDA), allocator/`tracemalloc`, record-dtype padding — verify, adapt only if a test trips.
