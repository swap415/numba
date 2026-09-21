import os
import unittest

import numpy as np

from numba import jit
from numba.tests.support import TestCase, override_config


class TestLLVMPassRemarks(TestCase):
    @TestCase.run_test_in_subprocess(envvars={
        'NUMBA_OPT': '3',
        'NUMBA_LOOP_VECTORIZE': '1',
    })
    def test_vectorization_remarks(self):
        @jit(debug=True)
        def add(left, right, out):
            for i in range(out.size):
                out[i] = left[i] + right[i]

        size = 64
        left = np.arange(size, dtype=np.float64)
        right = left + 1
        out = np.empty_like(left)
        with override_config('LLVM_PASS_REMARKS', 'loop-vectorize'):
            add(left, right, out)

        np.testing.assert_array_equal(out, left + right)
        metadata = add.get_metadata(add.signatures[0])
        remarks_by_stage = metadata['llvm_pass_remarks']
        remarks = '\n'.join(remarks_by_stage.values())
        self.assertIn('--- !Passed', remarks)
        self.assertIn('Pass:            loop-vectorize', remarks)
        self.assertIn('Name:            Vectorized', remarks)
        self.assertIn('DebugLoc:', remarks)
        self.assertIn(os.path.basename(__file__), remarks)

    def test_filter_excludes_remarks(self):
        @jit
        def add_one(value):
            return value + 1

        with override_config('LLVM_PASS_REMARKS', 'not-a-real-pass'):
            self.assertEqual(add_one(1), 2)

        metadata = add_one.get_metadata(add_one.signatures[0])
        self.assertEqual(metadata['llvm_pass_remarks'], {})

    def test_disabled(self):
        @jit
        def add_one(value):
            return value + 1

        with override_config('LLVM_PASS_REMARKS', None):
            self.assertEqual(add_one(1), 2)

        metadata = add_one.get_metadata(add_one.signatures[0])
        self.assertEqual(metadata['llvm_pass_remarks'], {})

    def test_invalid_filter(self):
        @jit
        def add_one(value):
            return value + 1

        with override_config('LLVM_PASS_REMARKS', '('):
            with self.assertRaisesRegex(RuntimeError,
                                        'parentheses not balanced'):
                add_one(1)


if __name__ == '__main__':
    unittest.main()
