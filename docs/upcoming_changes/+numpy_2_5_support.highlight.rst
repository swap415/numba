Support for NumPy 2.5
---------------------

Numba now supports NumPy 2.5, with the following notable changes:

- ``np.row_stack`` has been removed in NumPy 2.5. Use ``np.vstack`` instead
  (``np.row_stack`` was an alias of ``np.vstack``).

- ``np.cross`` no longer supports 2-element (2D) input vectors in NumPy 2.5.
  Numba's ``np.cross`` now rejects them as well; use ``cross2d`` from
  ``numba.np.extensions`` for 2D cross products.

- ``np.linalg.eig`` and ``np.linalg.eigvals`` now always return complex
  results for real input arrays, matching NumPy 2.5. As a result, Numba no
  longer raises for real matrices with complex eigenvalues on NumPy 2.5.

- ``np.sign`` of a ``timedelta64`` now returns a ``float64`` (``NaT`` maps to
  ``NaN``), matching the new NumPy 2.5 ufunc loop.

- ``np.searchsorted`` results for input arrays that are not sorted (an
  undefined case in NumPy) may differ from NumPy 2.5 due to NumPy's new batched
  binary search. Results for correctly sorted input are unaffected.
