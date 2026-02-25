"""
Tests for WASM target support.
"""
import unittest
import llvmlite.binding as ll
import llvmlite.ir as llvmir

from numba.core.wasm import WASMCodegen, WASMCodeLibrary, WASM32_TRIPLE


class TestWASMCodegen(unittest.TestCase):
    """Test basic WASM codegen functionality."""

    def test_create_codegen(self):
        """Test that WASMCodegen can be created."""
        cg = WASMCodegen("test_module")
        self.assertIsNotNone(cg)
        self.assertEqual(cg._llvm_module.triple, WASM32_TRIPLE)

    def test_create_library(self):
        """Test that a WASMCodeLibrary can be created."""
        cg = WASMCodegen("test_module")
        lib = cg.create_library("test_lib")
        self.assertIsInstance(lib, WASMCodeLibrary)

    def test_compile_simple_function(self):
        """Test compiling a simple add function to WASM."""
        # Create IR module with a simple add function
        ir_module = llvmir.Module(name="test_add")
        ir_module.triple = WASM32_TRIPLE

        # Define function: i32 @add(i32 %a, i32 %b)
        i32 = llvmir.IntType(32)
        fnty = llvmir.FunctionType(i32, [i32, i32])
        fn = llvmir.Function(ir_module, fnty, name="add")
        fn.attributes.add("wasm-export-name")
        fn.attributes.target_dependent["wasm-export-name"] = "add"

        # Build function body
        block = fn.append_basic_block(name="entry")
        builder = llvmir.IRBuilder(block)
        a, b = fn.args
        result = builder.add(a, b, name="result")
        builder.ret(result)

        # Compile to WASM
        cg = WASMCodegen("test_module")
        lib = cg.create_library("test_lib")
        lib.add_ir_module(ir_module)
        lib.finalize()

        # Get WASM bytes
        wasm_bytes = lib.get_wasm_bytes()
        self.assertTrue(wasm_bytes.startswith(b'\x00asm'))

        # Get and call function
        add_fn = lib.get_wasm_function("add")
        result = add_fn(3, 4)
        self.assertEqual(result, 7)

    def test_compile_multiply(self):
        """Test compiling a multiply function."""
        ir_module = llvmir.Module(name="test_mul")
        ir_module.triple = WASM32_TRIPLE

        i32 = llvmir.IntType(32)
        fnty = llvmir.FunctionType(i32, [i32, i32])
        fn = llvmir.Function(ir_module, fnty, name="multiply")
        fn.attributes.add("wasm-export-name")
        fn.attributes.target_dependent["wasm-export-name"] = "multiply"

        block = fn.append_basic_block(name="entry")
        builder = llvmir.IRBuilder(block)
        a, b = fn.args
        result = builder.mul(a, b, name="result")
        builder.ret(result)

        cg = WASMCodegen("test_module")
        lib = cg.create_library("test_lib")
        lib.add_ir_module(ir_module)
        lib.finalize()

        mul_fn = lib.get_wasm_function("multiply")
        self.assertEqual(mul_fn(6, 7), 42)


if __name__ == "__main__":
    unittest.main()
