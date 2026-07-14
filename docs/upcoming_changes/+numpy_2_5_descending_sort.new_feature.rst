Support for the ``descending`` keyword in sorting
-------------------------------------------------

``np.sort``, ``np.argsort`` and the ``ndarray.sort`` / ``ndarray.argsort``
methods now accept the ``descending`` keyword argument added in NumPy 2.5.
When ``descending=True`` the array is sorted in descending order, with any
NaNs placed at the end (matching NumPy). ``descending`` may be a runtime
boolean.
