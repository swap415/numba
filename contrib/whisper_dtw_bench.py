"""Compare OpenAI Whisper's Numba kernels with and without JIT.

Whisper (openai-whisper 20250625) JITs two CPU functions in whisper/timing.py:

    @numba.jit(nopython=True)
    def backtrace(trace)

    @numba.jit(nopython=True, parallel=True)
    def dtw_cpu(x)

The GPU DTW path is Triton (whisper/triton_ops.py:dtw_kernel), not Numba CUDA.
It still calls backtrace() on a NumPy copy of the trace.

Both kernels run only when word_timestamps=True (add_word_timestamps ->
find_alignment -> dtw). Default transcription never calls them.

dtw_cpu is an O(N*M) sequential DP. parallel=True is a no-op on the nested
loops: they use range(), not prange(), and have loop-carried dependences.
backtrace is O(path length) ~ O(N+M).

Sizes: a 30s Whisper window is 1500 time frames (TOKENS_PER_SECOND=50).
N is the number of text tokens in the window (~32-256).

Usage:
    python contrib/whisper_dtw_bench.py
    python contrib/whisper_dtw_bench.py --skip-e2e
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np


# --- Whisper timing.py kernels, copied verbatim except the decorator is applied
# both as identity (pure Python) and as numba.jit. --------------------

def backtrace(trace):
    i = trace.shape[0] - 1
    j = trace.shape[1] - 1
    trace[0, :] = 2
    trace[:, 0] = 1

    result = []
    while i > 0 or j > 0:
        result.append((i - 1, j - 1))

        if trace[i, j] == 0:
            i -= 1
            j -= 1
        elif trace[i, j] == 1:
            i -= 1
        elif trace[i, j] == 2:
            j -= 1
        else:
            raise ValueError("Unexpected trace[i, j]")

    result = np.array(result)
    return result[::-1, :].T


def dtw_cpu(x):
    N, M = x.shape
    cost = np.ones((N + 1, M + 1), dtype=np.float32) * np.inf
    trace = -np.ones((N + 1, M + 1), dtype=np.float32)

    cost[0, 0] = 0
    for j in range(1, M + 1):
        for i in range(1, N + 1):
            c0 = cost[i - 1, j - 1]
            c1 = cost[i - 1, j]
            c2 = cost[i, j - 1]

            if c0 < c1 and c0 < c2:
                c, t = c0, 0
            elif c1 < c0 and c1 < c2:
                c, t = c1, 1
            else:
                c, t = c2, 2

            cost[i, j] = x[i - 1, j - 1] + c
            trace[i, j] = t

    return backtrace(trace)


def dtw_fill(x):
    """dtw_cpu without backtrace; returns the trace matrix."""
    N, M = x.shape
    cost = np.ones((N + 1, M + 1), dtype=np.float32) * np.inf
    trace = -np.ones((N + 1, M + 1), dtype=np.float32)

    cost[0, 0] = 0
    for j in range(1, M + 1):
        for i in range(1, N + 1):
            c0 = cost[i - 1, j - 1]
            c1 = cost[i - 1, j]
            c2 = cost[i, j - 1]

            if c0 < c1 and c0 < c2:
                c, t = c0, 0
            elif c1 < c0 and c1 < c2:
                c, t = c1, 1
            else:
                c, t = c2, 2

            cost[i, j] = x[i - 1, j - 1] + c
            trace[i, j] = t

    return trace


JFK_URL = (
    "https://raw.githubusercontent.com/openai/whisper/main/tests/jfk.flac"
)
JFK_PATH = Path("/tmp/whisper-jfk.flac")


def _ms(seconds: float) -> str:
    if seconds < 1e-6:
        return f"{seconds * 1e9:.1f} ns"
    if seconds < 1e-3:
        return f"{seconds * 1e6:.1f} us"
    if seconds < 1:
        return f"{seconds * 1e3:.2f} ms"
    return f"{seconds:.3f} s"


def _bench(fn, make_args, repeats: int, warmup: int = 1) -> dict:
    for _ in range(warmup):
        fn(*make_args())
    samples = []
    for _ in range(repeats):
        args = make_args()
        t0 = time.perf_counter()
        fn(*args)
        samples.append(time.perf_counter() - t0)
    return {
        "median": statistics.median(samples),
        "min": min(samples),
        "n": repeats,
    }


def _compile_s(fn, args) -> float:
    t0 = time.perf_counter()
    fn(*args)
    return time.perf_counter() - t0


def _whisper_jit_pair(numba):
    """Decorate in Whisper's order: backtrace first, then dtw_cpu."""
    w_backtrace = numba.jit(nopython=True)(backtrace)
    saved = dtw_cpu.__globals__["backtrace"]
    dtw_cpu.__globals__["backtrace"] = w_backtrace
    try:
        w_dtw_cpu = numba.jit(nopython=True, parallel=True)(dtw_cpu)
    finally:
        dtw_cpu.__globals__["backtrace"] = saved
    return w_backtrace, w_dtw_cpu


def run_micro() -> dict:
    import numba

    w_backtrace, w_dtw_cpu = _whisper_jit_pair(numba)
    dtw_cpu_serial_nb = numba.jit(nopython=True)(dtw_cpu)
    dtw_fill_nb = numba.jit(nopython=True)(dtw_fill)

    rng = np.random.default_rng(0)
    # (tokens, frames). 1500 frames = 30s Whisper window.
    shapes = [(32, 300), (64, 1500), (128, 1500), (256, 1500)]

    x_tiny = rng.standard_normal((8, 8), dtype=np.float32)
    compile_times = {
        "whisper_backtrace": _compile_s(w_backtrace, (dtw_fill(x_tiny),)),
        "whisper_dtw_cpu": _compile_s(w_dtw_cpu, (x_tiny,)),
        "dtw_cpu_serial": _compile_s(dtw_cpu_serial_nb, (x_tiny,)),
        "dtw_fill": _compile_s(dtw_fill_nb, (x_tiny,)),
    }
    x_warm = rng.standard_normal(shapes[-1], dtype=np.float32)
    w_dtw_cpu(x_warm)
    dtw_cpu_serial_nb(x_warm)
    dtw_fill_nb(x_warm)

    rows = []
    for shape in shapes:
        x = rng.standard_normal(shape, dtype=np.float32)
        trace0 = dtw_fill_nb(x)
        path_len = int(w_backtrace(trace0.copy()).shape[1])

        def make_x(_x=x):
            return (_x.copy(),)

        def make_trace(_t=trace0):
            return (_t.copy(),)

        py_bt = _bench(backtrace, make_trace, repeats=15, warmup=2)
        nb_bt = _bench(w_backtrace, make_trace, repeats=50, warmup=5)
        py_dtw = _bench(dtw_cpu, make_x, repeats=3, warmup=1)
        nb_dtw = _bench(w_dtw_cpu, make_x, repeats=20, warmup=3)
        nb_dtw_serial = _bench(dtw_cpu_serial_nb, make_x, repeats=20, warmup=3)
        nb_fill = _bench(dtw_fill_nb, make_x, repeats=20, warmup=3)

        # Correctness: python vs numba on this shape.
        py_path = dtw_cpu(x)
        nb_path = w_dtw_cpu(x)
        match = bool(np.array_equal(py_path, nb_path))

        rows.append(
            {
                "shape": shape,
                "cells": shape[0] * shape[1],
                "path_len": path_len,
                "python_backtrace": py_bt,
                "numba_backtrace": nb_bt,
                "python_dtw_cpu": py_dtw,
                "numba_dtw_cpu": nb_dtw,
                "numba_dtw_cpu_serial": nb_dtw_serial,
                "numba_dtw_fill": nb_fill,
                "paths_match": match,
            }
        )

    return {
        "numba": numba.__version__,
        "numpy": np.__version__,
        "compile_s": compile_times,
        "rows": rows,
    }


def print_micro(result: dict) -> None:
    print()
    print("## Microbenchmark: Whisper kernels, Python vs Numba")
    print()
    print(f"numba {result['numba']}, numpy {result['numpy']}")
    print()
    print("JIT compile (first call, tiny 8x8 input):")
    for k, v in result["compile_s"].items():
        print(f"  {k}: {_ms(v)}")
    print()
    print(
        "| shape (tokens x frames) | kernel | Python | Numba | speedup |"
    )
    print("| --- | --- | ---: | ---: | ---: |")
    for row in result["rows"]:
        n, m = row["shape"]
        label = f"{n}x{m} ({row['cells']} cells, path {row['path_len']})"
        print(
            f"| {label} | backtrace | "
            f"{_ms(row['python_backtrace']['median'])} | "
            f"{_ms(row['numba_backtrace']['median'])} | "
            f"{row['python_backtrace']['median'] / row['numba_backtrace']['median']:.1f}x |"
        )
        print(
            f"| | dtw_cpu (Whisper: parallel=True) | "
            f"{_ms(row['python_dtw_cpu']['median'])} | "
            f"{_ms(row['numba_dtw_cpu']['median'])} | "
            f"{row['python_dtw_cpu']['median'] / row['numba_dtw_cpu']['median']:.1f}x |"
        )
        print(
            f"| | dtw_cpu serial nopython | "
            f"{_ms(row['python_dtw_cpu']['median'])} | "
            f"{_ms(row['numba_dtw_cpu_serial']['median'])} | "
            f"{row['python_dtw_cpu']['median'] / row['numba_dtw_cpu_serial']['median']:.1f}x |"
        )
        fill_frac = (
            row["numba_dtw_fill"]["median"] / row["numba_dtw_cpu"]["median"]
            if row["numba_dtw_cpu"]["median"]
            else 0.0
        )
        bt_frac = 1.0 - fill_frac
        print(
            f"| | numba fill vs full dtw (backtrace share) | "
            f"{_ms(row['numba_dtw_fill']['median'])} fill | "
            f"{_ms(row['numba_dtw_cpu']['median'])} full | "
            f"backtrace ~{max(0.0, bt_frac)*100:.1f}% |"
        )
        if not row["paths_match"]:
            print(f"| | correctness | MISMATCH | | |")
    print()


def run_e2e(audio_path: str, repeats: int) -> dict:
    import torch
    import whisper
    from whisper import timing as tmod
    from whisper import transcribe as tscript

    audio = whisper.load_audio(audio_path)
    duration = audio.shape[0] / whisper.audio.SAMPLE_RATE
    model = whisper.load_model("tiny.en", device="cpu")

    dtw_calls = []
    orig_dtw = tmod.dtw

    def wrapped_dtw(x):
        t0 = time.perf_counter()
        out = orig_dtw(x)
        dtw_calls.append(
            {
                "s": time.perf_counter() - t0,
                "shape": tuple(int(v) for v in x.shape),
            }
        )
        return out

    tmod.dtw = wrapped_dtw

    aws_calls = []
    orig_aws = tmod.add_word_timestamps

    def wrapped_aws(*args, **kwargs):
        t0 = time.perf_counter()
        out = orig_aws(*args, **kwargs)
        aws_calls.append(time.perf_counter() - t0)
        return out

    tmod.add_word_timestamps = wrapped_aws
    tscript.add_word_timestamps = wrapped_aws

    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))

    jit = os.environ.get("NUMBA_DISABLE_JIT", "0") not in {"1", "true", "True"}

    def once(word_timestamps: bool) -> dict:
        dtw_calls.clear()
        aws_calls.clear()
        t0 = time.perf_counter()
        result = model.transcribe(
            audio,
            word_timestamps=word_timestamps,
            fp16=False,
            verbose=False,
            temperature=0.0,
        )
        elapsed = time.perf_counter() - t0
        text = (result.get("text") or "").strip()
        return {
            "s": elapsed,
            "n_dtw": len(dtw_calls),
            "dtw_s": sum(c["s"] for c in dtw_calls),
            "dtw_shapes": [c["shape"] for c in dtw_calls],
            "aws_s": sum(aws_calls),
            "n_segments": len(result.get("segments") or []),
            "n_words": sum(
                len(seg.get("words") or [])
                for seg in (result.get("segments") or [])
            ),
            "text_len": len(text),
        }

    # Warmup compiles Numba and settles Torch.
    once(word_timestamps=True)

    plain = [once(False) for _ in range(repeats)]
    words = [once(True) for _ in range(repeats)]

    def med(runs, key):
        return statistics.median(r[key] for r in runs)

    return {
        "jit": jit,
        "numba_disable_jit": os.environ.get("NUMBA_DISABLE_JIT"),
        "torch": torch.__version__,
        "whisper": getattr(whisper, "__version__", "unknown"),
        "device": "cpu",
        "model": "tiny.en",
        "audio": audio_path,
        "audio_s": duration,
        "repeats": repeats,
        "plain_median_s": med(plain, "s"),
        "words_median_s": med(words, "s"),
        "words_dtw_median_s": med(words, "dtw_s"),
        "words_aws_median_s": med(words, "aws_s"),
        "plain": plain,
        "words": words,
        "threads": {
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "torch_threads": torch.get_num_threads(),
        },
    }


def print_e2e(result: dict) -> None:
    mode = "Numba JIT on" if result["jit"] else "Numba JIT disabled"
    print()
    print(f"## End-to-end Whisper tiny.en CPU ({mode})")
    print()
    print(
        f"audio {result['audio_s']:.2f}s, model {result['model']}, "
        f"device {result['device']}, repeats {result['repeats']}"
    )
    print()
    print("| run | median wall | DTW (sum) | add_word_timestamps |")
    print("| --- | ---: | ---: | ---: |")
    print(
        f"| word_timestamps=False | {_ms(result['plain_median_s'])} | "
        f"0 | 0 |"
    )
    print(
        f"| word_timestamps=True | {_ms(result['words_median_s'])} | "
        f"{_ms(result['words_dtw_median_s'])} | "
        f"{_ms(result['words_aws_median_s'])} |"
    )
    if result["words"]:
        shapes = result["words"][0]["dtw_shapes"]
        print()
        print(f"DTW calls per transcribe: {result['words'][0]['n_dtw']}, shapes {shapes}")
        print(
            f"DTW share of word-timestamp run: "
            f"{100 * result['words_dtw_median_s'] / result['words_median_s']:.2f}%"
        )
        extra = result["words_median_s"] - result["plain_median_s"]
        if extra > 0:
            print(
                f"word_timestamps overhead vs plain: {_ms(extra)} "
                f"({100 * extra / result['plain_median_s']:.1f}% of plain); "
                f"DTW is {100 * result['words_dtw_median_s'] / extra:.1f}% of that overhead"
            )
    print()


def _ensure_jfk() -> str:
    if not JFK_PATH.exists():
        print(f"downloading {JFK_URL}", flush=True)
        urlretrieve(JFK_URL, JFK_PATH)
    return str(JFK_PATH)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--skip-e2e", action="store_true")
    p.add_argument("--e2e-only", action="store_true")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument(
        "--dump",
        type=str,
        default="",
        help="Write JSON results to this path",
    )
    args = p.parse_args(argv)

    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("MKL_NUM_THREADS", "4")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
    os.environ.setdefault("NUMBA_NUM_THREADS", "4")

    blob: dict = {
        "python": sys.version.split()[0],
        "NUMBA_DISABLE_JIT": os.environ.get("NUMBA_DISABLE_JIT"),
    }

    if not args.e2e_only:
        print("running microbenchmarks...", flush=True)
        blob["micro"] = run_micro()
        print_micro(blob["micro"])

    if not args.skip_e2e:
        audio = _ensure_jfk()
        if args.e2e_only:
            print("running end-to-end Whisper...", flush=True)
            blob["e2e"] = run_e2e(audio, repeats=args.repeats)
            print_e2e(blob["e2e"])
        else:
            blob["e2e"] = []
            here = str(Path(__file__).resolve())
            for disable, tag in ((False, "jit-on"), (True, "jit-off")):
                dump = f"/tmp/whisper-e2e-{tag}.json"
                env = os.environ.copy()
                if disable:
                    env["NUMBA_DISABLE_JIT"] = "1"
                else:
                    env.pop("NUMBA_DISABLE_JIT", None)
                print(f"running end-to-end Whisper ({tag})...", flush=True)
                subprocess.check_call(
                    [
                        sys.executable,
                        here,
                        "--e2e-only",
                        "--repeats",
                        str(args.repeats),
                        "--dump",
                        dump,
                    ],
                    env=env,
                )
                child = json.loads(Path(dump).read_text())
                blob["e2e"].append(child["e2e"])
            print()
            print("## E2E comparison: Numba JIT on vs disabled")
            print()
            print(
                "| config | plain transcribe | word_timestamps | DTW | "
                "add_word_timestamps |"
            )
            print("| --- | ---: | ---: | ---: | ---: |")
            for e in blob["e2e"]:
                tag = "JIT on" if e["jit"] else "JIT off"
                print(
                    f"| {tag} | {_ms(e['plain_median_s'])} | "
                    f"{_ms(e['words_median_s'])} | "
                    f"{_ms(e['words_dtw_median_s'])} | "
                    f"{_ms(e['words_aws_median_s'])} |"
                )
            on, off = blob["e2e"][0], blob["e2e"][1]
            if not on["jit"]:
                on, off = off, on
            if off["words_dtw_median_s"] > 0:
                print()
                print(
                    "DTW JIT-off / JIT-on: "
                    f"{off['words_dtw_median_s'] / on['words_dtw_median_s']:.1f}x"
                )
                print(
                    "word_timestamps wall JIT-off / JIT-on: "
                    f"{off['words_median_s'] / on['words_median_s']:.3f}x"
                )
                print(
                    "plain transcribe JIT-off / JIT-on: "
                    f"{off['plain_median_s'] / on['plain_median_s']:.3f}x"
                )
            print()

    if args.dump:
        Path(args.dump).write_text(json.dumps(blob, indent=2, default=str))
        print(f"wrote {args.dump}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
