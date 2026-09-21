import gc
import math
import pickle
import re
import weakref

from numba import jit, njit
from numba.core.dispatcher import Dispatcher
from numba.core.extending import overload
from numba.core.targetconfig import ConfigStack
from numba.core.compiler import Compiler, Flags, DEFAULT_FLAGS
from numba.core import errors, event, types
from numba.core.funcdesc import default_mangler

from numba.tests.support import TestCase, unittest


class TestCompilerFlags(TestCase):
    def test_setting_invalid_attribute(self):
        flags = Flags()
        msg = "'Flags' object has no attribute 'this_really_does_not_exist'"
        with self.assertRaisesRegex(AttributeError, msg):
            flags.this_really_does_not_exist = True


class TestCompilerFlagCachedOverload(TestCase):
    def test_fastmath_in_overload(self):
        def fastmath_status():
            pass

        @overload(fastmath_status)
        def ov_fastmath_status():
            flags = ConfigStack().top()
            val = "Has fastmath" if flags.fastmath else "No fastmath"

            def codegen():
                return val

            return codegen

        @njit(fastmath=True)
        def set_fastmath():
            return fastmath_status()

        @njit()
        def foo():
            a = fastmath_status()
            b = set_fastmath()
            return (a, b)

        a, b = foo()
        self.assertEqual(a, "No fastmath")
        self.assertEqual(b, "Has fastmath")


class TestDispatcherFlagInheritance(TestCase):
    def make_caller(self, callee, **options):
        @jit(**options)
        def caller(a, b):
            return callee(a, b)

        return caller

    def test_fastmath_compile_order(self):
        for order in ((False, True), (True, False)):
            with self.subTest(order=order):
                @jit
                def callee(a, b):
                    return (a - b) + b

                callers = {flag: self.make_caller(callee, fastmath=flag)
                           for flag in order}
                for flag in order * 2:
                    expected = 0.5 if flag else 0.0
                    self.assertEqual(callers[flag](0.5, 1e16), expected)

    def test_fastmath_in_same_caller(self):
        for order in ((False, True), (True, False)):
            with self.subTest(order=order):
                @jit
                def callee(a, b):
                    return (a - b) + b

                first = self.make_caller(callee, fastmath=order[0])
                second = self.make_caller(callee, fastmath=order[1])

                @jit
                def caller(a, b):
                    return first(a, b), second(a, b)

                expected = tuple(0.5 if flag else 0.0 for flag in order)
                self.assertEqual(caller(0.5, 1e16), expected)

    def test_explicit_fastmath(self):
        for explicit in (False, True):
            for order in ((False, True), (True, False)):
                with self.subTest(explicit=explicit, order=order):
                    @jit(fastmath=explicit)
                    def callee(a, b):
                        return (a - b) + b

                    expected = 0.5 if explicit else 0.0
                    for flag in order:
                        caller = self.make_caller(callee, fastmath=flag)
                        self.assertEqual(caller(0.5, 1e16), expected)

    def test_python_call_after_fastmath(self):
        @jit
        def callee(a, b):
            return (a - b) + b

        caller = self.make_caller(callee, fastmath=True)
        self.assertEqual(caller(0.5, 1e16), 0.5)
        self.assertEqual(callee(0.5, 1e16), 0.0)
        self.assertEqual(caller(0.5, 1e16), 0.5)

    def test_nested_fastmath(self):
        for order in ((False, True), (True, False)):
            with self.subTest(order=order):
                @jit
                def callee(a, b):
                    return (a - b) + b

                middle = self.make_caller(callee)
                for flag in order:
                    caller = self.make_caller(middle, fastmath=flag)
                    expected = 0.5 if flag else 0.0
                    self.assertEqual(caller(0.5, 1e16), expected)

    def test_fastmath_subsets(self):
        for order in ((False, True), (True, False)):
            with self.subTest(order=order):
                @jit
                def callee(a, b):
                    return (a - b) + b

                for nsz in order:
                    flags = {'reassoc', 'nsz'} if nsz else {'reassoc'}
                    caller = self.make_caller(callee, fastmath=flags)
                    result = caller(0.5, math.inf)
                    if nsz:
                        self.assertEqual(result, 0.5)
                    else:
                        self.assertTrue(math.isnan(result))

    def test_error_model_compile_order(self):
        for order in (('python', 'numpy'), ('numpy', 'python')):
            with self.subTest(order=order):
                @jit
                def callee(a, b):
                    return a / b

                callers = {model: self.make_caller(callee, error_model=model)
                           for model in order}
                for model in order * 2:
                    if model == 'python':
                        with self.assertRaises(ZeroDivisionError):
                            callers[model](1.0, 0.0)
                    else:
                        self.assertEqual(callers[model](1.0, 0.0), math.inf)

    def test_reuse_same_flags(self):
        @jit
        def callee(a, b):
            return (a - b) + b

        for flag in (False, True):
            self.make_caller(callee, fastmath=flag)(0.5, 1e16)

        with event.install_recorder('numba:compile') as recorder:
            for flag in (False, True):
                caller = self.make_caller(callee, fastmath=flag)
                self.assertEqual(caller(0.5, 1e16), 0.5 if flag else 0.0)

        compiles = [evt for _, evt in recorder.buffer
                    if evt.data['dispatcher'].py_func is callee.py_func]
        self.assertEqual(compiles, [])

    def test_recursive_fastmath(self):
        for order in ((False, True), (True, False)):
            with self.subTest(order=order):
                @jit
                def callee(a, b):
                    if a < 1:
                        return (a - b) + b
                    return callee(a - 1, b) + 1

                for flag in order:
                    caller = self.make_caller(callee, fastmath=flag)
                    self.assertEqual(caller(2.5, 1e16), 2.5 if flag else 2.0)

    def test_mutual_recursion_fastmath(self):
        for flag in (False, True):
            with self.subTest(fastmath=flag):
                @jit
                def first(n, a, b):
                    if n <= 0:
                        return (a - b) + b
                    return second(n - 1, a, b)

                @jit(fastmath=flag)
                def second(n, a, b):
                    if n <= 0:
                        return (a - b) + b
                    return first(n - 1, a, b)

                @jit(fastmath=not flag)
                def caller(n, a, b):
                    return first(n, a, b)

                self.assertEqual(caller(0, 0.5, 1e16), 0.0 if flag else 0.5)
                for n in (1, 2, 3):
                    self.assertEqual(caller(n, 0.5, 1e16),
                                     0.5 if flag else 0.0)
                self.assertEqual(first(0, 0.5, 1e16), 0.0)

    def test_eager_signature(self):
        @jit('float32(float64, float64)')
        def callee(a, b):
            return (a - b) + b

        for flag in (False, True):
            caller = self.make_caller(callee, fastmath=flag)
            self.assertEqual(caller(0.5, 1e16), 0.5 if flag else 0.0)
            self.assertEqual(caller.nopython_signatures[0].return_type,
                             types.float32)
            with self.assertRaises(errors.TypingError):
                caller(0.5j, 1e16)

        self.assertEqual(callee(0.5, 1e16), 0.0)
        self.assertEqual(callee.signatures, [(types.float64, types.float64)])
        with self.assertRaises(TypeError):
            callee(0.5j, 1e16)

        callee.recompile()
        caller = self.make_caller(callee, fastmath=True)
        self.assertEqual(caller(0.5, 1e16), 0.5)
        self.assertEqual(caller.nopython_signatures[0].return_type,
                         types.float32)

    def test_eager_signature_conversion(self):
        @jit('int64(int64, int64)')
        def callee(a, b):
            return a + b

        caller = self.make_caller(callee, fastmath=True)
        self.assertEqual(caller(1.9, 2.9), 3)
        with self.assertRaises(errors.TypingError):
            caller(1j, 2j)
        self.assertEqual(callee.signatures, [(types.int64, types.int64)])

    def test_failed_eager_specialization(self):
        class CustomPipeline(Compiler):
            def compile_extra(self, func):
                if (self.state.flags.fastmath and
                        self.state.args == (types.int64,)):
                    raise errors.TypingError(
                        'reject fast integer specialization')
                return super().compile_extra(func)

        @jit(['int64(int64)', 'float64(float64)'],
             pipeline_class=CustomPipeline)
        def callee(a):
            return a + a

        def caller(a):
            return callee(a)

        with self.assertRaisesRegex(errors.TypingError,
                                    'reject fast integer specialization'):
            jit(fastmath=True)(caller)(1.0)
        with self.assertRaises(errors.TypingError):
            jit(fastmath=True)(caller)(1j)
        self.assertEqual(callee.signatures, [(types.int64,), (types.float64,)])

    def test_disable_compile(self):
        @jit
        def callee(a, b):
            return a + b

        caller = self.make_caller(callee, fastmath=True)
        self.assertEqual(caller(1.0, 2.0), 3.0)
        callee.disable_compile()
        self.assertEqual(callee(1.0, 2.0), 3.0)
        with self.assertRaises(TypeError):
            callee(1j, 2j)

        for flag in (False, True, {'reassoc'}):
            caller = self.make_caller(callee, fastmath=flag)
            self.assertEqual(caller(1.0, 2.0), 3.0)
            with self.assertRaises(errors.TypingError):
                caller(1j, 2j)

        callee.disable_compile(False)
        for flag in (False, True, {'reassoc'}):
            caller = self.make_caller(callee, fastmath=flag)
            self.assertEqual(caller(1j, 2j), 3j)
        self.assertEqual(callee(1j, 2j), 3j)

    def test_recompile(self):
        offset = 1.0

        @jit
        def callee(a, b):
            return (a - b) + b + offset

        for flag in (False, True):
            caller = self.make_caller(callee, fastmath=flag)
            self.assertEqual(caller(0.5, 1e16), 1.5 if flag else 1.0)

        offset = 2.0
        callee.recompile()
        for flag in (False, True):
            caller = self.make_caller(callee, fastmath=flag)
            self.assertEqual(caller(0.5, 1e16), 2.5 if flag else 2.0)

    def test_disable_compile_existing_signatures(self):
        @jit
        def callee(a, b):
            return a + b

        caller = self.make_caller(callee, fastmath=True)
        self.assertEqual(caller(1.0, 2.0), 3.0)
        self.assertEqual(callee(1j, 2j), 3j)
        callee.disable_compile()
        self.assertEqual(caller(1j, 2j), 3j)
        self.assertEqual(callee(1.0, 2.0), 3.0)

    def test_inferred_return_types(self):
        def selected():
            pass

        @overload(selected)
        def overload_selected():
            value = 'fast' if ConfigStack().top().fastmath else 1

            def impl():
                return value

            return impl

        @jit
        def callee(a, b):
            return selected()

        fast = self.make_caller(callee, fastmath=True)
        self.assertEqual(fast(1, 2), 'fast')
        for action in (callee.disable_compile, callee.recompile):
            action()
            self.assertEqual(callee(1, 2), 1)
            self.assertEqual(fast(1, 2), 'fast')
            for flag in (False, True, {'reassoc'}):
                caller = self.make_caller(callee, fastmath=flag)
                self.assertEqual(caller(1, 2), 'fast' if flag else 1)

    def test_serialized_return_types(self):
        for declared in (False, True):
            with self.subTest(declared=declared):
                offset = 1

                def impl(a, b):
                    return a + b + offset

                signature = 'int64(int64, int64)' if declared else None
                callee = jit(signature)(impl)
                self.assertEqual(callee(1, 0), 2)
                callee.disable_compile()
                original = weakref.ref(callee)
                pickled = pickle.dumps(callee)
                del callee
                Dispatcher._recent.clear()
                gc.collect()
                self.assertIsNone(original())

                callee = pickle.loads(pickled)
                callee.py_func.__closure__[0].cell_contents = 1.5
                callee.recompile()
                expected = 2 if declared else 2.5
                self.assertEqual(callee(1, 0), expected)
                caller = self.make_caller(callee, fastmath=True)
                self.assertEqual(caller(1, 0), expected)

    def test_custom_pipeline_and_locals(self):
        compiled = []

        class CustomPipeline(Compiler):
            def compile_extra(self, func):
                compiled.append(bool(self.state.flags.fastmath))
                return super().compile_extra(func)

        @jit(pipeline_class=CustomPipeline, locals={'value': types.float32})
        def callee(a, b):
            value = a
            return value + b

        for flag in (False, True):
            caller = self.make_caller(callee, fastmath=flag)
            self.assertEqual(caller(16777217.0, 0.0), 16777216.0)
        self.assertEqual(compiled, [False, True])

    def test_function_argument_fastmath(self):
        for flag in (False, True):
            with self.subTest(fastmath=flag):
                @jit
                def callee(a, b):
                    return (a - b) + b

                @jit(fastmath=flag)
                def apply(fn, a, b):
                    return fn(a, b)

                @jit(fastmath=not flag)
                def caller(a, b):
                    return apply(callee, a, b)

                self.assertEqual(caller(0.5, 1e16), 0.5 if flag else 0.0)


class TestFlagMangling(TestCase):

    def test_demangle(self):

        def check(flags):
            mangled = flags.get_mangle_string()
            out = flags.demangle(mangled)
            # Demangle result MUST match summary()
            self.assertEqual(out, flags.summary())

        # test empty flags
        flags = Flags()
        check(flags)

        # test default
        check(DEFAULT_FLAGS)

        # test other
        flags = Flags()
        flags.no_cpython_wrapper = True
        flags.nrt = True
        flags.fastmath = True
        check(flags)

    def test_mangled_flags_is_shorter(self):
        # at least for these control cases
        flags = Flags()
        flags.nrt = True
        flags.auto_parallel = True
        self.assertLess(len(flags.get_mangle_string()), len(flags.summary()))

    def test_mangled_flags_with_fastmath_parfors_inline(self):
        # at least for these control cases
        flags = Flags()
        flags.nrt = True
        flags.auto_parallel = True
        flags.fastmath = True
        flags.inline = "always"
        self.assertLess(len(flags.get_mangle_string()), len(flags.summary()))
        demangled = flags.demangle(flags.get_mangle_string())
        # There should be no pointer value in the demangled string.
        self.assertNotIn("0x", demangled)

    def test_demangling_from_mangled_symbols(self):
        """Test demangling of flags from mangled symbol"""
        # Use default mangler to mangle the string
        fname = 'foo'
        argtypes = types.int32,
        flags = Flags()
        flags.nrt = True
        flags.inline = "always"
        name = default_mangler(
            fname, argtypes, abi_tags=[flags.get_mangle_string()],
        )
        # Find the ABI-tag. Starts with "B"
        prefix = "_Z3fooB"
        # Find the length of the ABI-tag
        m = re.match("[0-9]+", name[len(prefix):])
        size = m.group(0)
        # Extract the ABI tag
        base = len(prefix) + len(size)
        abi_mangled = name[base:base + int(size)]
        # Demangle and check
        demangled = Flags.demangle(abi_mangled)
        self.assertEqual(demangled, flags.summary())


if __name__ == "__main__":
    unittest.main()
