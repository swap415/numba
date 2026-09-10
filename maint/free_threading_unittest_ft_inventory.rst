Free-threading inventory (Python 3.15t + unittest-ft)
=====================================================

This is a work-order, not a changelog. Goal: Numba's unittest suite
survives ``unittest-ft`` on a free-threading interpreter so the remaining
failures can be fixed one subsystem at a time.

Environment
-----------

* Branch synced to ``numba/numba`` main (includes Python 3.15 support,
  ``max_python_version = "3.16"`` exclusive).
* CPython 3.15.0rc2 free-threading (``python3.15t``, ``sys._is_gil_enabled()
  is False``), ``PYTHON_GIL=0``.
* NumPy 2.5.2 cp315t, llvmlite 0.50.0rc3 cp315t.
* In-tree Numba 0.68.0dev0 built with gcc/g++.
* Runner: ``unittest-ft -j 4`` per test module, 240s timeout (longer for
  known-slow modules). CUDA and gdb suites skipped (no device / no gdb).

``unittest-ft`` loads every test in a module and runs them concurrently on a
thread pool in one process. That is a harsher workload than Numba's own
``-m`` multiprocess runner, which respects ``_numba_parallel_test_ = False``.

Headline
--------

221 modules discovered (CPU tests only).

* 145 OK
* 72 FAIL
* 2 TIMEOUT (no progress until killed)
* 1 SIGSEGV
* 1 deadlock then SIGABRT

Sequential ``numba.runtests`` of ``test_threadsafety`` is green on 3.15t.
The SIGSEGV only appears when those tests *overlap* under ``unittest-ft``.
A large fraction of FAILs are one test-infra global (NRT leak-check stats).
The rest are product data races.

Suggested fix order
-------------------

Do these in order. Later items stay racy until earlier shared mutable
state is locked or made illegal to share.

1. Typing / impl registry loaders
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``numba.core.utils.stream_list`` yields a shared generator;
``BaseRegistryLoader.new_registrations`` does ``next(self._registrations[name])``.
Two threads in ``typingctx.refresh()`` / ``targetctx.refresh()`` hit:

* ``ValueError: generator already executing``
  (``test_array_analysis``, ``test_help``, ``test_looplifting``)
* Missing attributes that never got installed, e.g.
  ``Unknown attribute 'flat' of type array(...)`` (``test_remove_dead``)

Fix: make the incremental loader thread-safe (mutex around ``next()``, or
replace the generator with an index + list snapshot). Same pattern exists
for both typing and impl registries.

2. NRT stats enable flag + MemoryLeakMixin
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``NRT_MemSys_enable_stats`` / ``disable_stats`` flip a process-global bool.
``MemoryLeakMixin`` snapshots process-global alloc/free counters.

Under ``unittest-ft`` this is the dominant FAIL mode:

* ``RuntimeError: NRT stats are disabled.`` (42 modules)
* ``AssertionError: <alloc> != <free>`` (31 modules)

Not a leak in the compiler. Concurrent tests enable/disable/read the same
counters.

Fix: refcount the enable flag; keep counters atomic (they already are);
either skip leak checks when another test holds stats, or give the mixin
thread-local / scoped stats. Unblocks most of the 72 FAILs so real races
stop hiding in the noise.

3. Dispatcher C++ overload table
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``Dispatcher::addDefinition`` ``push_back``s into ``std::vector`` with no
mutex. ``Dispatcher::resolve`` / ``Dispatcher_call`` read ``functions.size()``
and iterate the same vectors. ``compile()`` holds ``global_compiler_lock``
on the Python side, but *calls* of already-compiled overloads do not, so a
compile on thread B races with a call on thread A.

Fix: mutex (or RCU/immutable snapshot) around the C++ overload table.
This is on the call hot path; keep the uncontended case cheap.

4. Type intern cache
~~~~~~~~~~~~~~~~~~~~

``numba.core.types.abstract._typecache`` and ``_typecodes = itertools.count()``
are interned without a lock. Duplicate type codes or a torn intern break
``_dispatcher`` matching.

``_typeof.cpp`` also keeps process-global ``typecache`` / ``ndarray_typecache``
dicts (CPython dicts are internally locked; the check-then-act around them
is not).

5. workqueue / parallel gufunc
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``numba/np/ufunc/workqueue.c`` says it up front: *"This module is not
thread-safe."* ``_nesting_level`` and the task queue are DSO globals.

Empirical: ``test_threadsafety`` sequential OK; ``unittest-ft`` SIGSEGV in
``GUFunc.__call__`` → ``self.ufunc(*args)`` while several
``test_concurrent_*`` methods run at once.

``npyufunc.test_update_inplace`` hung for 240s with **zero** tests finished
(likely the same queue / nest-level deadlock).

Fix: serialize ``add_task`` / pool lifetime, make nesting atomic, or refuse
overlapping parallel regions from multiple Python threads with a defined
error instead of a hang/crash.

6. Dynamic (g)ufunc compile path
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``GUFunc.__call__`` / DUFunc ``build_ufunc`` is check-then-act on
``self.ufunc`` and the ewise cache. Overlapping ``@guvectorize`` compiles
from several tests share that path.

Also: ``npyufunc.test_caching`` almost all FAILs from concurrent cache-dir
/ module import (see item 12).

7. StencilFunc IR mutation
~~~~~~~~~~~~~~~~~~~~~~~~~~

``StencilFunc.id_counter += 1`` is racy. Worse: stencil lowering mutates
``kernel_ir`` / neighborhood state.

Empirical: ``test_stencils`` under ``unittest-ft`` prints
``unsupported operand type(s) for -: 'NoneType' and 'float'`` on many
``TestManyStencils`` cases, then deadlocks. Faulthandler dump showed the
main thread in ``concurrent.futures.wait`` and a worker in
``_PyMutex_LockTimed`` / ``_PyParkingLot_Park``. Sequential stencils
progresses.

8. Temporary dispatcher map
~~~~~~~~~~~~~~~~~~~~~~~~~~~

``numba.core.typeinfer._temporary_dispatcher_map`` is a process-global
``{py_func.__name__: dispatcher}``. Two compiles of different functions
with the same ``__name__`` clobber each other. Used for recursive
``@jit`` resolution.

9. Lowering / codegen caches
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Unlocked dicts, check-then-act:

* ``numba.np.arrayobj._sorts``
* ``numba.core.cgutils._struct_proxy_cache``
* ``numba.experimental.jitclass.boxing._cache_specialized_box``
* ``numba.core.typing.cffi_utils._ool_func_types`` / ``_ool_func_ptr``
* CUDA ``_ptx_cache`` etc. (not exercised here)

``test_serialize`` compiled a pickled/renamed module and hit
``NotImplementedError: No definition for lowering <built-in method <lambda>
of _dynfunc._Closure ...>``. That is a lowering-map miss consistent with a
racy install or a clobbered closure environment.

10. Compiler event bus
~~~~~~~~~~~~~~~~~~~~~~

``numba.core.event.broadcast`` walks a global listener list. Listeners hold
mutable instance state (``TimingListener._ts``).

Empirical: ``test_event`` asserts see the wrong dispatcher; ``test_operators``
failed in ``compiler_lock.release`` → ``TimingListener.on_end`` with
``AttributeError: 'TimingListener' object has no attribute '_ts'``.

The RLock itself is fine. The callbacks are not.

11. WeakValueDictionary memos
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``_MemoMixin._memo``, ``environment.Environment._memo``, cloudpickle
trackers. ``WeakValueDictionary`` is not free-threading-safe. Deserialize
/ rebuild of dispatchers can drop or duplicate instances.

12. Test isolation (do after 1–2 so the suite is readable)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These are not product bugs but they make ``unittest-ft`` unusable:

* ``import_dynamic`` + shared module names
  (``dispatcher_caching_test_fodder`` ``KeyError`` in importlib;
  ``test_caching``, ``test_parfors_caching``).
* CFFI out-of-line module: concurrent writes to the same ``.so``
  (``ImportError: ...so: file too short``).
* ``captured_stdout`` swapping ``sys.stdout`` process-wide.
* Shared ``CACHE_DIR`` / trashcan temp dirs.
* ``test_event`` listeners registered globally in ``setUp``.

Crashes and hangs (must-fix product)
------------------------------------

=============================================  ========  ============================================
Module                                         Result    Notes
=============================================  ========  ============================================
``numba.tests.test_threadsafety``              SIGSEGV   gufunc ``__call__``; OK if tests run serially
``numba.tests.test_stencils``                  deadlock  NoneType neighborhood math, then PyMutex park
``numba.tests.npyufunc.test_update_inplace``   TIMEOUT   240s, 0 tests completed
``numba.tests.test_parfors_passes``            TIMEOUT   11 tests FAIL then hang
=============================================  ========  ============================================

Already skipped on free-threading
---------------------------------

``skip_if_freethreading`` covers ``test_pycc``, GIL-reacquire deadlock tests
in ``test_gil`` / ``test_parallel_ufunc_issues``. Those stay skipped until
pycc and GIL-reacquire semantics are defined for no-GIL.

What already worked concurrently
--------------------------------

Not an all-clear, but these ran to completion under ``unittest-ft``: 
``test_dispatcher``, ``test_parfors`` (293 tests), ``test_parallel_backend``,
``test_num_threads``, ``test_nrt``, ``test_gil`` (minus skipped cases),
``test_random``, ``test_typeof``, most doc examples.

That means ``global_compiler_lock`` is doing *some* of the job for
single-function compile. The failures above are the places it does not
cover: C++ tables, registries, NRT stats, workqueue, stencil IR, events.

How to re-run
-------------

::

    PYTHON_GIL=0 PYTHONFAULTHANDLER=1 PYTHONPATH=. \
      python -m unittest_ft -v -j 4 numba.tests.test_threadsafety

Per-module sweep with timeouts lives in the agent notes; do not add a
permanent harness until the first few items above are fixed.
