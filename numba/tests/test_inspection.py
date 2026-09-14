import sys
import unittest
from unittest.mock import Mock, patch

import llvmlite.binding as ll

from numba import jit
from numba.core.codegen import JITCPUCodegen
from numba.core.compiler_lock import global_compiler_lock
from numba.misc import inspection
from numba.tests.support import TestCase


try:
    import capstone
except ImportError:
    capstone = None


class TestInspectionDependencies(TestCase):
    def test_missing_capstone(self):
        @jit
        def increment(x):
            return x + 1

        with patch.dict(sys.modules, {'capstone': None}):
            self.assertEqual(increment(1), 2)
            self.assertTrue(increment.inspect_asm(increment.signatures[0]))
            with self.assertRaisesRegex(RuntimeError, 'capstone'):
                increment.inspect_disasm(increment.signatures[0])


@unittest.skipIf(capstone is None, 'requires capstone')
class TestObjectDisassembly(TestCase):
    @global_compiler_lock
    def emit_object(self, triple, source):
        ll.initialize_all_targets()
        ll.initialize_all_asmprinters()
        target = ll.Target.from_triple(triple)
        with target.create_target_machine() as machine:
            with ll.parse_assembly(source) as module:
                module.triple = triple
                module.data_layout = str(machine.target_data)
                return machine.emit_object(module)

    def test_object_formats(self):
        triples = (
            'x86_64-unknown-linux-gnu',
            'i386-unknown-linux-gnu',
            'i686-unknown-linux-gnu',
            'aarch64-unknown-linux-gnu',
            'x86_64-apple-darwin',
            'arm64-apple-darwin',
            'x86_64-pc-windows-msvc',
        )
        source = 'define i32 @answer() { ret i32 42 }'
        for triple in triples:
            with self.subTest(triple=triple):
                data = self.emit_object(triple, source)
                result = inspection.disassemble_object(data, triple)
                self.assertIn('text', result)
                self.assertRegex(result, r'0x[0-9a-f]+\s+\w+')
                self.assertRegex(result, r'\bret\b')

    def test_data_only(self):
        triple = 'x86_64-unknown-linux-gnu'
        data = self.emit_object(triple, '@value = global i32 42')
        self.assertEqual(inspection.disassemble_object(data, triple), '')

    def test_unsupported_architecture(self):
        with self.assertRaisesRegex(ValueError, 'riscv64'):
            inspection.disassemble_object(b'', 'riscv64-unknown-linux-gnu')

    def test_incomplete_instruction(self):
        section = Mock()
        section.is_text.return_value = True
        section.name.return_value = b'.text'
        section.address.return_value = 0x1000
        section.data.return_value = b'\xc3\x0f'
        section.size.return_value = 2
        with patch.object(ll.ObjectFileRef, 'from_data') as from_data:
            obj = from_data.return_value.__enter__.return_value
            obj.sections.return_value = [section]
            with self.assertRaisesRegex(ValueError, r'\.text.*0x(?:100)?1\b'):
                inspection.disassemble_object(b'', 'x86_64-unknown-linux-gnu')


@unittest.skipIf(capstone is None, 'requires capstone')
class TestDispatcherDisassembly(TestCase):
    def setUp(self):
        super().setUp()
        arch = ll.get_process_triple().split('-')[0]
        if arch not in {'x86_64', 'i386', 'i686', 'aarch64', 'arm64'}:
            self.skipTest(f'unsupported host architecture: {arch}')

    def test_signatures(self):
        @jit
        def increment(x):
            return x + 1

        self.assertEqual(increment.inspect_disasm(), {})
        self.assertEqual(increment(1), 2)
        self.assertEqual(increment(1.5), 2.5)
        results = increment.inspect_disasm()
        self.assertEqual(set(results), set(increment.signatures))
        self.assertEqual(len(results), 2)
        for signature, result in results.items():
            with self.subTest(signature=signature):
                self.assertIsInstance(result, str)
                self.assertIn('text', result)
                self.assertRegex(result, r'\bret\b')
                self.assertEqual(result, increment.inspect_disasm(signature))
                library = increment.overloads[signature].library
                self.assertEqual(result, library.get_disasm_str())

    @global_compiler_lock
    def test_cached_code(self):
        @jit
        def increment(x):
            return x + 1

        self.assertEqual(increment(1), 2)
        library = increment.overloads[increment.signatures[0]].library
        state = library.serialize_using_object_code()
        codegen = JITCPUCodegen('disassembly_cache_test')
        restored = codegen.unserialize_library(state)
        with self.assertRaisesRegex(RuntimeError, 'cached code'):
            restored.get_disasm_str()


if __name__ == '__main__':
    unittest.main()
