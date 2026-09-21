import json
import os
from pathlib import Path
import struct
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

import llvmlite.binding as llvm
import numba

from numba.tests.support import TestCase


class TestPerfJIT(TestCase):
    @unittest.skipUnless(
        sys.platform.startswith('linux') and
        llvm.has_perf_jit_events,
        'requires an LLVM build with perf JIT events',
    )
    def test_jitdump_contains_numba_debug_info(self):
        source = """import inspect
import json
from numba import jit

@jit
def perf_jitdump_target(value):
    return value + 1

assert perf_jitdump_target(41) == 42
signature = perf_jitdump_target.signatures[0]
descriptor = perf_jitdump_target.overloads[signature].fndesc
lines, start_line = inspect.getsourcelines(perf_jitdump_target.py_func)
return_line = start_line + next(
    index for index, line in enumerate(lines) if 'return value + 1' in line
)
print(json.dumps({'name': descriptor.mangled_name, 'line': return_line}))
"""
        with TemporaryDirectory() as directory:
            source_path = Path(directory) / 'workload.py'
            source_path.write_text(source, encoding='utf-8')
            env = os.environ.copy()
            package_root = str(Path(numba.__file__).resolve().parent.parent)
            pythonpath = [package_root]
            if env.get('PYTHONPATH'):
                pythonpath.append(env['PYTHONPATH'])
            env['PYTHONPATH'] = os.pathsep.join(pythonpath)
            env['JITDUMPDIR'] = directory
            env['NUMBA_ENABLE_PROFILING'] = '1'
            env.pop('NUMBA_DEBUGINFO', None)
            env.pop('NUMBA_DISABLE_JIT', None)
            completed = subprocess.run(
                [sys.executable, str(source_path)],
                check=True,
                capture_output=True,
                env=env,
                text=True,
                timeout=60,
            )
            expected = json.loads(completed.stdout.splitlines()[-1])
            dumps = list(Path(directory).rglob('jit-*.dump'))
            self.assertEqual(len(dumps), 1)
            data = dumps[0].read_bytes()

        byte_order = '<' if sys.byteorder == 'little' else '>'
        file_header = struct.Struct(byte_order + 'IIIIIIQQ')
        record_header = struct.Struct(byte_order + 'IIQ')
        code_load = struct.Struct(byte_order + 'IIQQQQ')
        debug_info = struct.Struct(byte_order + 'QQ')
        debug_entry = struct.Struct(byte_order + 'QII')

        self.assertGreaterEqual(len(data), file_header.size)
        magic, version, header_size, *_ = file_header.unpack_from(data)
        self.assertEqual(magic, 0x4A695444)
        self.assertEqual(version, 1)
        self.assertGreaterEqual(header_size, file_header.size)
        self.assertLessEqual(header_size, len(data))

        loads = []
        debug_records = []
        offset = header_size
        while offset < len(data):
            self.assertGreaterEqual(len(data) - offset, record_header.size)
            record_id, record_size, _ = record_header.unpack_from(data, offset)
            self.assertGreaterEqual(record_size, record_header.size)
            record_end = offset + record_size
            self.assertLessEqual(record_end, len(data))
            fixed_offset = offset + record_header.size

            if record_id == 0:
                self.assertGreaterEqual(
                    record_size,
                    record_header.size + code_load.size + 1,
                )
                _, _, _, address, size, _ = code_load.unpack_from(
                    data, fixed_offset,
                )
                name_offset = fixed_offset + code_load.size
                name_end = data.index(b'\x00', name_offset, record_end)
                self.assertEqual(record_end - name_end - 1, size)
                loads.append({
                    'offset': offset,
                    'address': address,
                    'name': data[name_offset:name_end].decode(),
                })
            elif record_id == 2:
                self.assertGreaterEqual(
                    record_size,
                    record_header.size + debug_info.size,
                )
                address, count = debug_info.unpack_from(data, fixed_offset)
                entry_offset = fixed_offset + debug_info.size
                entries = []
                for _ in range(count):
                    self.assertGreaterEqual(
                        record_end - entry_offset, debug_entry.size + 1,
                    )
                    line_address, line, _ = debug_entry.unpack_from(
                        data, entry_offset,
                    )
                    entry_offset += debug_entry.size
                    filename_end = data.index(
                        b'\x00', entry_offset, record_end,
                    )
                    filename = data[entry_offset:filename_end].decode()
                    entry_offset = filename_end + 1
                    entries.append((line_address, line, filename))
                self.assertEqual(entry_offset, record_end)
                debug_records.append({
                    'offset': offset,
                    'address': address,
                    'entries': entries,
                })
            offset = record_end

        self.assertEqual(offset, len(data))
        target_loads = [
            record for record in loads if record['name'] == expected['name']
        ]
        self.assertEqual(len(target_loads), 1)
        target_load = target_loads[0]
        target_debug_records = [
            record for record in debug_records
            if record['address'] == target_load['address']
        ]
        self.assertEqual(len(target_debug_records), 1)
        target_debug = target_debug_records[0]
        self.assertLess(target_debug['offset'], target_load['offset'])
        locations = {
            (line, os.path.basename(filename))
            for _, line, filename in target_debug['entries']
        }
        self.assertIn((expected['line'], 'workload.py'), locations)


if __name__ == '__main__':
    unittest.main()
