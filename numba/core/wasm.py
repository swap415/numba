"""
WASM target support for Numba.

This module provides codegen classes for compiling Numba functions to WebAssembly.
The target infrastructure is created lazily to avoid circular imports.
"""

import re

import llvmlite.binding as ll
import llvmlite.ir as llvmir


# WASM target triple and data layout
WASM32_TRIPLE = "wasm32-unknown-unknown"
WASM32_DATA_LAYOUT = "e-m:e-p:32:32-i64:64-n32:64-S128"


class _WASMCFunc:
    """
    A Python callable wrapper for a WASM function.
    This mimics the interface expected by numba's dispatcher.
    """
    __slots__ = ('_wasm_func', '_fndesc', '_wrapper')

    def __init__(self, wasm_func, fndesc):
        self._wasm_func = wasm_func
        self._fndesc = fndesc
        # Create a simple wrapper function
        def wrapper(*args):
            # For now, assume we can call the WASM function directly
            # This works because wasmtime handles the basic type conversions
            # The WASM function uses CPU calling convention (retptr, excinfo, args)
            # but for simple numeric types, we can call it more directly
            return wasm_func(*args)
        self._wrapper = wrapper

    def __call__(self, *args):
        return self._wrapper(*args)


def _create_wasm_cfunc_wrapper(wasm_func, fndesc):
    """Create a Python callable wrapper for a WASM function."""
    return _WASMCFunc(wasm_func, fndesc)


def _normalize_ir_text(text):
    """Normalize IR text for cross-platform compatibility."""
    return re.sub(r'[\x00]', '', text).replace('\r\n', '\n')


class WASMCodeLibrary:
    """
    A code library that compiles to WebAssembly using llvmlite's WasmExecutionEngine.
    """

    def __init__(self, codegen, name):
        self._codegen = codegen
        self.name = name
        self._finalized = False
        self._final_module = ll.parse_assembly(
            str(self._codegen._create_empty_module(self.name)))
        self._final_module.name = _normalize_ir_text(self.name)
        self._wasm_engine = None
        self._wasm_functions = {}
        self._dynamic_globals = []
        self._reload_init = set()
        self._entry_name = None
        self.recorded_timings = None

    def _raise_if_finalized(self):
        if self._finalized:
            raise RuntimeError("CodeLibrary already finalized")

    def _ensure_finalized(self):
        if not self._finalized:
            self.finalize()

    def add_ir_module(self, ir_module):
        self._raise_if_finalized()
        assert isinstance(ir_module, llvmir.Module)
        ir = _normalize_ir_text(str(ir_module))
        # Fix common linkage for WASM - convert "common global" to "internal global"
        # WASM doesn't support common symbols
        ir = ir.replace(' = common global ', ' = internal global ')
        ll_module = ll.parse_assembly(ir)
        ll_module.name = ir_module.name
        ll_module.verify()
        self.add_llvm_module(ll_module)

    def add_llvm_module(self, ll_module):
        self._final_module.link_in(ll_module)

    def finalize(self):
        self._raise_if_finalized()
        self._final_module.verify()
        self._finalize_final_module()

    def _finalize_final_module(self):
        self._finalize_dynamic_globals()
        # Create WASM engine from the module
        self._wasm_engine = ll.create_wasm_engine(self._final_module)
        self._wasm_engine.finalize_object()
        self._finalized = True

    def _finalize_dynamic_globals(self):
        # Scan for dynamic globals
        for gv in self._final_module.global_variables:
            if gv.name.startswith('numba.dynamic.globals'):
                self._dynamic_globals.append(gv.name)

    def get_wasm_function(self, name):
        """
        Get a callable WASM function by name.

        Returns a WasmFunction that can be called with Python arguments.
        """
        self._ensure_finalized()
        if name not in self._wasm_functions:
            self._wasm_functions[name] = self._wasm_engine.get_function(name)
        return self._wasm_functions[name]

    def get_wasm_bytes(self):
        """
        Get the compiled WASM binary.

        Returns the raw .wasm bytes that can be loaded in a browser.
        """
        self._ensure_finalized()
        return self._wasm_engine.get_wasm_bytes()

    def get_pointer_to_function(self, name):
        """
        Not supported for WASM - use get_wasm_function() instead.
        """
        raise NotImplementedError(
            "WASM does not use native pointers. Use get_wasm_function() instead."
        )

    def enable_object_caching(self):
        """Enable object caching (no-op for WASM)."""
        pass

    def add_linking_library(self, library):
        """Add a library to link with."""
        if hasattr(library, '_final_module'):
            self.add_llvm_module(library._final_module)

    def get_llvm_str(self):
        """Get the LLVM IR as a string."""
        return str(self._final_module)

    def create_ir_module(self, name):
        """Create a new IR module for lowering."""
        return self._codegen._create_empty_module(name)


class WASMCodegen:
    """
    Codegen for WebAssembly target.
    """

    _library_class = WASMCodeLibrary

    def __init__(self, module_name):
        # Initialize LLVM targets
        ll.initialize_all_targets()
        ll.initialize_all_asmprinters()

        self._data_layout = WASM32_DATA_LAYOUT
        self._llvm_module = ll.parse_assembly(
            str(self._create_empty_module(module_name)))
        self._llvm_module.name = "wasm_codegen_module"
        self._init(self._llvm_module)

    def _init(self, llvm_module):
        # Create target machine for wasm32
        target = ll.Target.from_triple(WASM32_TRIPLE)
        self._tm = target.create_target_machine(
            opt=2,
            reloc='default',
            codemodel='default',
        )
        self._target_data = self._tm.target_data
        self._data_layout = str(self._target_data)

    def _create_empty_module(self, name):
        ir_module = llvmir.Module(_normalize_ir_text(name))
        ir_module.triple = WASM32_TRIPLE
        ir_module.data_layout = self._data_layout
        return ir_module

    def _add_module(self, module):
        # Not needed for WASM - each library manages its own engine
        pass

    def create_library(self, name):
        """Create a WASMCodeLibrary for use with this codegen."""
        return self._library_class(self, name)

    @property
    def target_data(self):
        """The LLVM target data object."""
        return self._target_data

    def magic_tuple(self):
        """Return a tuple describing the codegen target."""
        return (WASM32_TRIPLE, "generic", "")


# -----------------------------------------------------------------------------
# Lazy target infrastructure to avoid circular imports
# -----------------------------------------------------------------------------

_wasm_target = None
_WASMContext = None
_WASMDispatcher = None


def _create_target_classes():
    """Create WASM target classes lazily to avoid circular imports."""
    global _WASMContext, _WASMDispatcher

    if _WASMContext is not None:
        return

    from numba.core.base import BaseContext
    from numba.core.compiler_lock import global_compiler_lock
    from numba.core.descriptors import TargetDescriptor
    from numba.core import dispatcher, typing
    from numba.core.options import TargetOptions, include_default_options
    from numba.core.utils import threadsafe_cached_property as cached_property

    # Minimal set of options for WASM target
    _wasm_options_mixin = include_default_options(
        "nopython",
        "forceobj",
        "_nrt",
        "debug",
        "boundscheck",
        "no_rewrites",
        "no_cpython_wrapper",
        "no_cfunc_wrapper",
        "fastmath",
        "error_model",
    )

    class WASMTargetOptions(_wasm_options_mixin, TargetOptions):
        """Target options for WASM compilation."""
        def finalize(self, flags, options):
            # WASM always uses nopython mode
            if not flags.is_set("enable_pyobject"):
                flags.enable_pyobject = False
            # Disable NRT for WASM (different memory model)
            flags.inherit_if_not_set("nrt", default=False)

    class WASMContext(BaseContext):
        """
        Target context for WebAssembly compilation.
        """
        # WASM doesn't support dynamic globals (runtime addresses)
        allow_dynamic_globals = False

        # Disable NRT for now - WASM has different memory model
        enable_nrt = False

        def __init__(self, typingctx, target='wasm'):
            super().__init__(typingctx, target)

        @global_compiler_lock
        def init(self):
            self._internal_codegen = WASMCodegen("numba.wasm")

        @property
        def call_conv(self):
            from numba.core import callconv
            return callconv.CPUCallConv(self)

        def create_module(self, name):
            return self._internal_codegen._create_empty_module(name)

        def codegen(self):
            return self._internal_codegen

        @property
        def target_data(self):
            """Get the LLVM target data for WASM."""
            return self._internal_codegen.target_data

        def get_executable(self, library, fndesc, env):
            """Get an executable wrapper for the compiled function."""
            # Finalize the library if not already done
            library._ensure_finalized()
            # Get the WASM function
            wasm_func = library.get_wasm_function(fndesc.llvm_func_name)
            # Create a Python callable wrapper
            return _create_wasm_cfunc_wrapper(wasm_func, fndesc)

    class WASMTarget(TargetDescriptor):
        """Target descriptor for WASM."""
        options = WASMTargetOptions

        @cached_property
        def _toplevel_target_context(self):
            return WASMContext(self.typing_context, self._target_name)

        @cached_property
        def _toplevel_typing_context(self):
            return typing.Context()

        @property
        def target_context(self):
            return self._toplevel_target_context

        @property
        def typing_context(self):
            return self._toplevel_typing_context

    class WASMDispatcher(dispatcher.Dispatcher):
        """Dispatcher for WASM target."""
        pass

    _WASMContext = WASMContext
    _WASMDispatcher = WASMDispatcher

    return WASMTarget, WASMTargetOptions


def get_wasm_target():
    """Get or create the global WASM target instance."""
    global _wasm_target
    if _wasm_target is None:
        WASMTarget, _ = _create_target_classes()
        _wasm_target = WASMTarget('wasm')
        _WASMDispatcher.targetdescr = _wasm_target
    return _wasm_target


# -----------------------------------------------------------------------------
# WASM JIT Decorator
# -----------------------------------------------------------------------------

class WASMCompiledFunction:
    """
    A compiled WASM function that can be called from Python.
    Handles the CPU calling convention (retptr, excinfo, args).
    """
    # Fixed offsets in WASM linear memory for calling convention
    RETPTR_OFFSET = 0       # Return value at offset 0
    EXCINFO_OFFSET = 16     # Exception info at offset 16

    def __init__(self, py_func, library, fndesc, signature):
        self._py_func = py_func
        self._library = library
        self._fndesc = fndesc
        self._signature = signature
        self._wasm_func = library.get_wasm_function(fndesc.llvm_func_name)
        self._wasm_engine = library._wasm_engine

    def __call__(self, *args):
        """Call the compiled WASM function."""
        import struct
        # Call with CPU calling convention: (retptr, excinfo, *args)
        # retptr and excinfo are WASM memory offsets (pointers in WASM are i32)
        status = self._wasm_func(
            self.RETPTR_OFFSET,
            self.EXCINFO_OFFSET,
            *args
        )

        # Check status (0 = RETCODE_OK in CPU calling convention)
        if status != 0:
            raise RuntimeError(f"WASM function returned error status: {status}")

        # Read result from retptr (8 bytes for i64)
        result_bytes = self._wasm_engine.read_memory(self.RETPTR_OFFSET, 8)
        result = struct.unpack('<q', result_bytes)[0]  # Little-endian i64
        return result

    def get_wasm_bytes(self):
        """Get the compiled WASM binary."""
        return self._library.get_wasm_bytes()

    def get_llvm_ir(self):
        """Get the LLVM IR."""
        return self._library.get_llvm_str()


class WASMJitCompiler:
    """
    Simple compiler for WASM that bypasses the complex dispatcher.
    """
    def __init__(self, py_func, signature=None):
        self._py_func = py_func
        self._signature = signature
        self._overloads = {}

    def compile(self, argtypes):
        """Compile the function for the given argument types."""
        from numba.core import compiler, sigutils
        from numba.core.compiler import Flags

        # Get the WASM target context
        target = get_wasm_target()
        typingctx = target.typing_context
        targetctx = target.target_context

        # Create compilation flags for nopython mode
        flags = Flags()
        flags.no_cpython_wrapper = True
        flags.no_cfunc_wrapper = True
        flags.error_model = 'numpy'
        flags.enable_pyobject = False
        flags.force_pyobject = False
        flags.nrt = False  # WASM doesn't use NRT

        # Create library
        library = targetctx.codegen().create_library(self._py_func.__name__)

        # Compile
        cres = compiler.compile_extra(
            typingctx, targetctx, self._py_func,
            args=argtypes, return_type=None,
            flags=flags, locals={},
            library=library,
        )

        # Finalize the library if not already finalized
        if not library._finalized:
            library.finalize()

        # Create compiled function
        compiled = WASMCompiledFunction(
            self._py_func, library, cres.fndesc, cres.signature
        )
        self._overloads[argtypes] = compiled
        return compiled

    def __call__(self, *args):
        """Compile and call with the given arguments."""
        from numba import typeof

        # Get argument types
        argtypes = tuple(typeof(arg) for arg in args)

        # Check if already compiled
        if argtypes not in self._overloads:
            self.compile(argtypes)

        return self._overloads[argtypes](*args)

    def get_wasm_bytes(self):
        """Get WASM bytes for the first (or only) overload."""
        if not self._overloads:
            raise RuntimeError("Function not yet compiled")
        return next(iter(self._overloads.values())).get_wasm_bytes()


def wasm_jit(signature_or_function=None, **options):
    """
    Decorator to compile a function to WebAssembly.

    Usage:
        @wasm_jit
        def add(a, b):
            return a + b

        # Or with signature:
        @wasm_jit('int32(int32, int32)')
        def add(a, b):
            return a + b

    The compiled function can be called normally, and the WASM binary
    can be retrieved via the .get_wasm_bytes() method.
    """
    # Ensure WASM target is registered
    _register_wasm_target()

    def decorator(func):
        return WASMJitCompiler(func, signature_or_function)

    if signature_or_function is None:
        return decorator
    elif callable(signature_or_function):
        # @wasm_jit without arguments
        return WASMJitCompiler(signature_or_function)
    else:
        # @wasm_jit('signature')
        return decorator


# -----------------------------------------------------------------------------
# Target Registration
# -----------------------------------------------------------------------------

# WASM target class for the registry (inherits from Target, not TargetDescriptor)
_WASM_target_class = None


def _get_wasm_target_class():
    """Get or create the WASM target class for the registry."""
    global _WASM_target_class
    if _WASM_target_class is None:
        from numba.core.target_extension import Generic

        class WASM(Generic):
            """Mark the target as WASM (WebAssembly)."""
            pass

        _WASM_target_class = WASM
    return _WASM_target_class


def _register_wasm_target():
    """Register WASM target with numba's target registry."""
    from numba.core.target_extension import (
        target_registry, dispatcher_registry, jit_registry
    )

    # Only register if not already registered
    if 'wasm' not in target_registry:
        # Ensure target classes are created (includes dispatcher with targetdescr)
        get_wasm_target()

        WASM = _get_wasm_target_class()
        target_registry['wasm'] = WASM
        dispatcher_registry[WASM] = _WASMDispatcher
        jit_registry[WASM] = wasm_jit


# Don't auto-register on import to avoid circular imports
# Users should call _register_wasm_target() or use wasm_jit() which will trigger registration
