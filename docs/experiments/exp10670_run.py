"""Runner for the numba issue #10670 unroll-metadata experiment.

Runs each phase of exp10670_worker.py in a fresh subprocess (so monkeypatch
state can never leak between phases), collects the JSON results into
exp10670_results.json next to this file, and prints a comparison table.

Usage: python exp10670_run.py
"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
WORKER = os.path.join(HERE, 'exp10670_worker.py')
PHASES = ['baseline', 'inner_disable', 'all_runtime_disable', 'all_disable']
FUNCS = ['add_for', 'add_while_uint', 'add_for_const']


def main():
    results = {}
    for phase in PHASES:
        print(f'--- running phase: {phase}', file=sys.stderr)
        proc = subprocess.run([sys.executable, WORKER, phase],
                              capture_output=True, text=True, cwd=HERE)
        if proc.returncode != 0:
            print(proc.stdout, file=sys.stderr)
            print(proc.stderr, file=sys.stderr)
            raise SystemExit(f'phase {phase} failed')
        results[phase] = json.loads(proc.stdout)

    out_path = os.path.join(HERE, 'exp10670_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'wrote {out_path}', file=sys.stderr)

    base = results['baseline']['results']
    print(f"\nversions: {results['baseline']['versions']}")
    for phase in PHASES:
        print(f'\n== {phase} (tagged latches: '
              f'{results[phase]["tagged_latches"]}) ==')
        hdr = (f'{"func":16} {"cond_br":>7} {"tot_br":>6} {"fadd":>4} '
               f'{"maxrun":>6} {"unrblk":>6}  guards / unchanged-vs-baseline')
        print(hdr)
        for fn in FUNCS:
            r = results[phase]['results'][fn]
            g = r['guards']
            gs = ','.join(k for k, v in g.items() if v)
            same = ('SAME' if r['norm_hash'] == base[fn]['norm_hash']
                    else 'DIFF')
            print(f'{fn:16} {r["cond_br"]:>7} {r["total_br"]:>6} '
                  f'{r["fadd_count"]:>4} {r["max_fadd_run"]:>6} '
                  f'{r["unroll_named_blocks"]:>6}  [{gs}] {same}')


if __name__ == '__main__':
    main()
