from numba import jit


@jit
def increment(x):
    return x + 1


assert increment(2) == 3
assert increment(2.5) == 3.5
assert len(increment.inspect_asm()) == 2
assert len(increment.inspect_disasm()) == 2
assert "ret" in increment.inspect_disasm(increment.signatures[0])
