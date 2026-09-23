import capstone as cs
import llvmlite.binding as ll


ll.initialize_all_targets()
ll.initialize_all_asmprinters()
sources = [
    'define double @constant() { ret double 0x400921FB54442D18 }',
    '''define i64 @branch(i64 %x) {
        entry: switch i64 %x, label %other [
            i64 0, label %a i64 1, label %b i64 2, label %c i64 3, label %d]
        a: ret i64 5
        b: ret i64 9
        c: ret i64 15
        d: ret i64 71
        other: ret i64 42
    }''',
]
triples = ['x86_64-unknown-linux-gnu', 'aarch64-unknown-linux-gnu',
           'arm64-apple-darwin', 'x86_64-pc-windows-msvc']
for triple in triples:
    decoder = (cs.Cs(cs.CS_ARCH_X86, cs.CS_MODE_64)
               if triple.startswith('x86')
               else cs.Cs(cs.CS_ARCH_ARM64, cs.CS_MODE_ARM))
    with ll.Target.from_triple(triple).create_target_machine() as machine:
        for source in sources:
            with ll.parse_assembly(source) as module:
                module.triple = triple
                module.data_layout = str(machine.target_data)
                data = machine.emit_object(module)
                with ll.ObjectFileRef.from_data(data) as obj:
                    for section in obj.sections():
                        if section.is_text():
                            code = section.data()
                            instructions = decoder.disasm_lite(
                                code, section.address())
                            consumed = sum(i[1] for i in instructions)
                            print(triple, section.name(), consumed, len(code))
                            assert consumed == len(code), code.hex()
