"""
WASM target support for Numba.

This module provides codegen classes for compiling Numba functions to WebAssembly.
"""

import re

import llvmlite.binding as ll
import llvmlite.ir as llvmir

from numba.core import cgutils
from numba.core.base import BaseContext
from numba.core.codegen import Codegen, CodeLibrary, initialize_llvm
from numba.core.compiler_lock import global_compiler_lock
from numba.core.descriptors import TargetDescriptor
from numba.core import dispatcher, typing
from numba.core.options import TargetOptions
from numba.core.utils import threadsafe_cached_property as cached_property


# WASM target triple and data layout
WASM32_TRIPLE = "wasm32-unknown-unknown"
WASM32_DATA_LAYOUT = "e-m:e-p:32:32-i64:64-n32:64-S128"


class WASMCodeLibrary(CodeLibrary):
    """
    A code library that compiles to WebAssembly using llvmlite's WasmExecutionEngine.
    """

    def __init__(self, codegen, name):
        super().__init__(codegen, name)
        self._final_module = ll.parse_assembly(
            str(self._codegen._create_empty_module(self.name)))
        self._final_module.name = cgutils.normalize_ir_text(self.name)
        self._wasm_engine = None
        self._wasm_functions = {}

    def add_ir_module(self, ir_module):
        self._raise_if_finalized()
        assert isinstance(ir_module, llvmir.Module)
        ir = cgutils.normalize_ir_text(str(ir_module))
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


class WASMCodegen(Codegen):
    """
    Codegen for WebAssembly target.
    """

    _library_class = WASMCodeLibrary

    def __init__(self, module_name):
        initialize_llvm()
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
        ir_module = llvmir.Module(cgutils.normalize_ir_text(name))
        ir_module.triple = WASM32_TRIPLE
        ir_module.data_layout = self._data_layout
        return ir_module

    def _add_module(self, module):
        # Not needed for WASM - each library manages its own engine
        pass

    @property
    def target_data(self):
        """The LLVM target data object."""
        return self._target_data

    def magic_tuple(self):
        """Return a tuple describing the codegen target."""
        return (WASM32_TRIPLE, "generic", "")


# -----------------------------------------------------------------------------
# WASM Target Context
# -----------------------------------------------------------------------------

class WASMTargetOptions(TargetOptions):
    """Target options for WASM compilation."""
    pass


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

    def create_module(self, name):
        return self._internal_codegen._create_empty_module(name)

    def codegen(self):
        return self._internal_codegen


# -----------------------------------------------------------------------------
# WASM Target Descriptor and Dispatcher
# -----------------------------------------------------------------------------

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


# Global WASM target instance
wasm_target = WASMTarget('wasm')


class WASMDispatcher(dispatcher.Dispatcher):
    """Dispatcher for WASM target."""
    targetdescr = wasm_target


# -----------------------------------------------------------------------------
# WASM JIT Decorator
# -----------------------------------------------------------------------------

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
    from numba.core.decorators import jit

    # Force WASM target and nopython mode
    options['target'] = 'wasm'
    options['nopython'] = True

    return jit(signature_or_function, **options)


# -----------------------------------------------------------------------------
# Target Registration (called when module is imported)
# -----------------------------------------------------------------------------

def _register_wasm_target():
    """Register WASM target with numba's target registry."""
    from numba.core.target_extension import (
        target_registry, dispatcher_registry, jit_registry
    )

    # Only register if not already registered
    if 'wasm' not in target_registry:
        target_registry['wasm'] = wasm_target
        dispatcher_registry[wasm_target] = WASMDispatcher
        jit_registry[wasm_target] = wasm_jit


# Auto-register on import
_register_wasm_target()
