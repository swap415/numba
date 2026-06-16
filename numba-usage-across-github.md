# Numba usage across GitHub — notable open-source projects (ranked by stars)

**Compiled:** 2026-06-16 · **Source:** GitHub Code Search API + Repositories API (live star/fork counts).

## What this is (and methodology)

[Numba](https://github.com/numba/numba) is a JIT compiler for numerical Python.
The goal was to find repositories that use Numba **in their actual source code**
(a real `import numba` / `from numba import …` / `@njit` / `@numba.jit`), and rank
them by GitHub stars and forks.

Two important caveats about "all" uses:

1. **Volume.** A single import pattern — `from numba import njit` — already matches
   **11,600+ files** on GitHub, and `import numba` matches far more. The vast majority
   are personal projects, coursework, and forks. A literal exhaustive list is neither
   feasible nor useful, so this is a curated ranking of the **most-starred, notable
   projects** with verified direct usage.
2. **Code Search can't sort by stars.** GitHub's code-search endpoint returns matches in
   relevance/index order, not by repository popularity. So the process was: identify
   candidate projects across the scientific-Python / ML / audio ecosystems, **verify each
   one actually imports numba in its source** (via `repo:owner/name from numba import` /
   `import numba`), then pull live star/fork counts from the Repositories API and sort.

Every repository in the table below was individually verified to contain a real Numba
import in its own code (file paths cited in the "Where it's used" column).

## Ranked list (verified direct Numba users)

| # | Repository | ⭐ Stars | 🍴 Forks | Domain | Where Numba is used (example) |
|---|------------|--------:|--------:|--------|-------------------------------|
| 1 | [openai/whisper](https://github.com/openai/whisper) | 102,793 | 12,539 | Speech recognition | `whisper/timing.py` (`import numba`) — DTW for word timestamps |
| 2 | [pandas-dev/pandas](https://github.com/pandas-dev/pandas) | 48,984 | 20,014 | DataFrames | `pandas/core/_numba/kernels/*.py` — the `engine="numba"` execution path |
| 3 | [shap/shap](https://github.com/shap/shap) | 25,532 | 3,729 | ML explainability | `shap/explainers/_exact.py`, `_partition.py`, `utils/_masked_model.py` (`@njit`) |
| 4 | [numba/numba](https://github.com/numba/numba) | 11,048 | 1,280 | *Numba itself* | The compiler's own source (self-hosting/test suite) |
| 5 | [yzhao062/pyod](https://github.com/yzhao062/pyod) | 9,877 | 1,482 | Anomaly detection | `pyod/models/{abod,hbos,loci,sod,rod,lmdd,qmcd}.py` (`@njit`) |
| 6 | [sktime/sktime](https://github.com/sktime/sktime) | 9,808 | 2,194 | Time-series ML | `sktime/.../_*_numba.py`, ROCKET/SFA/DTW kernels (`@njit`) |
| 7 | [vaexio/vaex](https://github.com/vaexio/vaex) | 8,504 | 603 | Out-of-core DataFrames | `vaex-core/vaex/expression.py` (Numba JIT virtual columns), `vaex-ml/cluster.py` |
| 8 | [librosa/librosa](https://github.com/librosa/librosa) | 8,466 | 1,054 | Audio/music analysis | `core/audio.py`, `filters.py`, `sequence.py`, `beat.py` (`from numba import jit`) |
| 9 | [lmcinnes/umap](https://github.com/lmcinnes/umap) | 8,205 | 862 | Dimensionality reduction | `umap/umap_.py`, `layouts.py`, `distances.py`, `sparse.py` (`import numba`) |
| 10 | [DeepLabCut/DeepLabCut](https://github.com/DeepLabCut/DeepLabCut) | 5,683 | 1,787 | Pose estimation | `deeplabcut/core/trackingutils.py` (`from numba import jit`) |
| 11 | [stumpy-dev/stumpy](https://github.com/stumpy-dev/stumpy) | 4,102 | 353 | Time-series (matrix profile) | Core STAMP/STOMP kernels are Numba-compiled throughout |
| 12 | [holoviz/datashader](https://github.com/holoviz/datashader) | 3,551 | 378 | Big-data visualization | `compiler.py`, `glyphs/*.py`, `reductions.py`, `composite.py` (`@ngjit`) |
| 13 | [tslearn-team/tslearn](https://github.com/tslearn-team/tslearn) | 3,155 | 372 | Time-series ML | `tslearn/metrics/{_dtw,_gak,soft_dtw_fast,…}.py` (`from numba import njit`) |
| 14 | [scverse/scanpy](https://github.com/scverse/scanpy) | 2,491 | 750 | Single-cell genomics | `metrics/_morans_i.py`, `_gearys_c.py`, `tools/_rank_genes_groups.py` (`@njit`) |
| 15 | [QuantEcon/QuantEcon.py](https://github.com/QuantEcon/QuantEcon.py) | 2,357 | 2,282 | Quantitative economics | `quantecon/optimize/*`, `markov/*`, `game_theory/*` (Numba pervasive) |
| 16 | [scikit-hep/awkward](https://github.com/scikit-hep/awkward) | 965 | 125 | Particle-physics arrays | `src/awkward/_connect/numba/*` — first-class Numba extension type |
| 17 | [pymc-devs/pytensor](https://github.com/pymc-devs/pytensor) | 622 | 193 | Tensor compiler (PyMC backend) | `pytensor/link/numba/**` — full Numba compilation backend |

> Star/fork counts are live as of the compile date and change continuously.

## Notable exclusions (popular, but NOT direct code users)

These show up if you search loosely for "numba," but they do **not** import Numba in their
own source — included here so the distinction is explicit:

| Repository | ⭐ Stars | Why it's excluded |
|------------|--------:|-------------------|
| [freqtrade/freqtrade](https://github.com/freqtrade/freqtrade) | 51,506 | No `import numba` anywhere in source |
| [microsoft/qlib](https://github.com/microsoft/qlib) | 44,462 | No direct Numba import in source |
| [modin-project/modin](https://github.com/modin-project/modin) | 10,388 | No direct Numba import in source |
| [facebookresearch/demucs](https://github.com/facebookresearch/demucs) | 10,214 | No direct Numba import (archived) |
| [scikit-image/scikit-image](https://github.com/scikit-image/scikit-image) | 6,529 | Uses Cython, not Numba |
| [scikit-learn-contrib/hdbscan](https://github.com/scikit-learn-contrib/hdbscan) | 3,119 | Uses Cython, not Numba |
| [PythonOT/POT](https://github.com/PythonOT/POT) | 2,809 | Uses its own numpy/torch/jax backend, not Numba |
| [RVC-Project/Retrieval-based-Voice-Conversion-WebUI](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) | ~25k | Only configures `logging.getLogger("numba")`; Numba is a transitive dep via librosa |
| [coqui-ai/TTS](https://github.com/coqui-ai/TTS) | ~38k | Numba is a transitive dependency (via librosa); no direct import |

**Related note — PyMC:** [pymc-devs/pymc](https://github.com/pymc-devs/pymc) (~9k⭐) runs
models through Numba via PyTensor's `NumbaLinker` (`mode="NUMBA"`) rather than authoring
`@njit` kernels itself — the real Numba code lives in **PyTensor** (#17 above).

## How to reproduce / extend

```text
# Verify a repo imports Numba directly (GitHub Code Search):
repo:<owner>/<name> from numba import
repo:<owner>/<name> import numba

# Get live stars/forks (Repositories API):
GET https://api.github.com/repos/<owner>/<name>   ->  stargazers_count, forks_count
```

The same approach generalizes: pick a candidate, confirm a real import in its source, then
rank by the popularity metric you care about (stars or forks).
