from capstone import Cs, CS_ARCH_ARM64, CS_MODE_ARM
from llvmlite import binding as llvm
from numba import jit


@jit
def increment(x):
    return x + 1


assert increment(2) == 3
library = increment.overloads[increment.signatures[0]].library
decoder = Cs(CS_ARCH_ARM64, CS_MODE_ARM)
with llvm.ObjectFileRef.from_data(library._get_compiled_object()) as obj:
    for section in obj.sections():
        if section.is_text():
            code = decoder.disasm_lite(section.data(), section.address())
            instructions = list(code)
            assert sum(row[1] for row in instructions) == section.size()
            print(section.name(), section.size(), len(instructions))
