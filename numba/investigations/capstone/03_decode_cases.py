import math

import capstone as cs
import llvmlite.binding as ll
import numpy as np
from numba import jit


@jit
def trig(x):
    return math.sin(x) + math.pi


@jit
def reduce(xs):
    total = 0.0
    for x in xs:
        if x > 0:
            total += x * x
    return total


@jit(parallel=True)
def parallel(xs):
    return (xs * xs).sum()


arch = ll.get_process_triple().split('-')[0]
assert arch in {'aarch64', 'arm64', 'x86_64'}
decoder = (cs.Cs(cs.CS_ARCH_X86, cs.CS_MODE_64) if arch == 'x86_64'
           else cs.Cs(cs.CS_ARCH_ARM64, cs.CS_MODE_ARM))
for function, argument in [(trig, 1.0), (reduce, np.arange(16.)),
                           (parallel, np.arange(16.))]:
    function(argument)
    library = next(iter(function.overloads.values())).library
    with ll.ObjectFileRef.from_data(library._get_compiled_object()) as obj:
        for section in obj.sections():
            if section.is_text():
                code = section.data()
                instructions = decoder.disasm_lite(code, section.address())
                consumed = sum(size for _, size, _, _ in instructions)
                print(function.py_func.__name__, consumed, len(code))
                assert consumed == len(code), code[consumed:consumed + 16].hex()
