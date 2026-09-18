==========================
Notes on Target Extensions
==========================

.. warning:: All features and APIs described in this page are in-development and
             may change at any time without deprecation notices being issued.


Inheriting compiler flags from the caller
=========================================

Compiler flags, i.e. options such as ``fastmath``, ``nrt`` in
``@jit(nrt=True, fastmath=True))`` are specified per-function but their
effects are not well-defined---some flags affect the entire callgraph, some
flags affect only the current function. Sometimes it is necessary for callees
to inherit flags from the caller; for example the ``fastmath`` flag should be
infectious.

To address the problem, the following are needed:

1. Better definitions for the semantics of compiler flags. Preferably, all flags should
   limit their effect to the current function. (TODO)
2. Allow compiler flags to be inherited from the caller. (Done)
3. Consider compiler flags in function resolution. (Done for CPU dispatchers)

:class:`numba.core.targetconfig.ConfigStack` is used to propagate the compiler flags
throughout the compiler. At the start of the compilation, the flags are pushed
into the ``ConfigStack``, which maintains a thread-local stack for the
compilation. Thus, callees can check the flags in the caller.

CPU dispatchers select separate compiled versions when inherited ``fastmath``,
``error_model``, ``forceinline``, or ``_nrt`` options differ. Explicit callee
options take precedence over the caller's options. Calls from Python use the
callee's own options and defaults. Inherited versions are private dispatchers;
the public overload table and inspection methods describe the default version.
Disk cache entries distinguish effective compiler flags.

Targets opt into this selection by listing their inherited user-facing option
names in ``TargetOptions.inheritable``. The corresponding option mappings must
accept the resolved flag values. Targets without this declaration retain their
existing compilation behavior.

This selection applies to calls that retain a dispatcher boundary. Numba IR
inlining, such as ``inline='always'``, copies the callee's body into the caller
before lowering and continues to use the caller's flags for that body.

.. autoclass:: numba.core.targetconfig.ConfigStack
    :members:

Compiler flags
--------------

`Compiler flags`_ are defined as a subclass of ``TargetConfig``:

.. _Compiler flags: https://github.com/numba/numba/blob/7e8538140ce3f8d01a5273a39233b5481d8b20b1/numba/core/compiler.py#L39

.. autoclass:: numba.core.targetconfig.TargetConfig
    :members:


These are internal compiler flags and they are different from the user-facing
options used in the jit decorators.

Internally, `the user-facing options are mapped to the internal compiler flags <https://github.com/numba/numba/blob/7e8538140ce3f8d01a5273a39233b5481d8b20b1/numba/core/options.py#L72>`_
by :class:`numba.core.options.TargetOptions`. Each target can override the
default compiler flags and control the flag inheritance in
``TargetOptions.finalize``. `The CPU target overrides it.
<https://github.com/numba/numba/blob/7e8538140ce3f8d01a5273a39233b5481d8b20b1/numba/core/cpu.py#L259>`_

.. autoclass:: numba.core.options.TargetOptions
    :members: finalize


In :meth:`numba.core.options.TargetOptions.finalize`,
use :meth:`numba.core.targetconfig.TargetConfig.inherit_if_not_set`
to request a compiler flag from the caller if it is not set for the current
function.
