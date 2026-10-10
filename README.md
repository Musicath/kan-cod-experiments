# KAN CoD experiments

Code and compact result tables for *Kolmogorov--Arnold Networks: Curse of
Dimensionality and Structural Bias*. The manuscript source is maintained in
[CoD-of-KANs](https://github.com/Musicath/CoD-of-KANs).

This repository covers the paper's constructive spline-KAN to ReLU-cubed MLP
transfer, controlled ridge-profile experiments, and normal-label and
random-label MNIST experiments. The worst-case lower bound is a mathematical
proof in the paper and does not depend on experiment code.

## Setup

The reported training environment used Python 3.12.7, PyTorch 2.5.1, and the
official PyKAN repository at commit
`ecde4ec3274d3bef1ad737479cf126aed38ab530`. Dependencies are pinned in
[`requirements.txt`](requirements.txt). Use a fresh environment and run:

```bash
python -m pip install -r requirements.txt
```

The PyKAN dependency is installed from its upstream Git repository. No copy of
PyKAN or MNIST image data is redistributed here. PyTorch installation may need
a platform-specific CUDA wheel; follow the official PyTorch instructions for
your machine before installing the remaining requirements.

## Rebuild the figures from the released tables

```bash
python experiments/analysis/run_all.py
python experiments/analysis/validate_release.py
```

The first command writes the original experiment figures to
`COD_of_KAN/figures/` and derived tables to `experiments/data/processed/`.
Those generated files are ignored by Git. The second command checks table
counts, selected summary values, and the existence of the principal figures.
Neither command retrains models. The reproduced panel from Liu et al.'s paper
is not redistributed here; consult the cited original publication.

GitHub Actions runs these two commands on pushes, pull requests, version tags,
and manual dispatch using the pinned packages in
[`requirements-analysis.txt`](requirements-analysis.txt). This CPU-only check
does not rerun the training experiments or the separate GPU smoke test.

The compact inputs committed here are:

| Result | Source file |
| --- | --- |
| Constructive scaling (four main-text targets) | `experiments/diagnostics/constructive_example2/results/*.csv` |
| Controlled ridge profiles (three seeds) | `experiments/data/processed/ridge_profile_v2_24gpu_20260920/all_runs.csv.gz` |
| Sine-profile supplementary diagnostics | `experiments/data/raw/sine_ridge_v2/*.csv` |
| Standard- and random-label MNIST | `experiments/data/raw/mnist_n60000/*.csv`, `experiments/data/raw/mnist_n6000/*.csv` |

`all_runs.csv.gz` contains 174,960 successful configurations and is directly
readable by pandas. Large per-job server archives, logs, model checkpoints, and
historical exploratory scans are intentionally excluded. See
[`DATA.md`](DATA.md) for provenance and the distinction between result-table
reconstruction and full training reruns.

## Rerun experiments

- `experiments/diagnostics/constructive_example2/five_function_warm_start.py`
  constructs the target-aware spline-KAN approximants, transfers them to
  ReLU-cubed MLPs, and optionally refines the warm starts. It imports
  `construct_and_plot.py` from the same directory.
- `experiments/manifests/prepare_profile_screen_v2_cosine_schedule.py`
  recreates the controlled-ridge configuration grid. The GPU launcher is
  `experiments/manifests/launch_dense_fourier_manifest.py`, with
  `experiments/runners/run_interaction_profile_v1.py` as its runner.
- `experiments/runners/run_mnist_normal_labels.py` and
  `experiments/runners/run_mnist_random_labels_full.py` implement the two MNIST
  tasks. The normal labels are used for classification; the random-label test
  measures fitting of independently randomized training labels.

For example, generate one seed's ridge-profile manifests:

```bash
python experiments/manifests/prepare_profile_screen_v2_cosine_schedule.py \
  --output-dir experiments/manifests/generated --seed 0
```

Full sweeps are computationally expensive. The original ridge experiment
pooled 174,960 successful jobs over three seeds. The released CSV tables let
readers check and redraw the reported comparisons without repeating that cost.
To restrict a GPU run to physical device 1, pass `--gpus 1` to the launcher;
it sets `CUDA_VISIBLE_DEVICES` for the worker. Keep generated runs under
`experiments/runs/` so machine metadata and logs are not committed.

## Scope and provenance

This is a curated paper-code release, not a dump of the authors' working
directory. The figure-building script covers the original experiment panels
used in the paper; some figures in the paper are sourced from prior work and
are outside this repository. The released MNIST training entry points include
later portability and low-budget extensions relative to the exact historical
server snapshots. The committed metric tables, rather than a claim of a new
byte-identical training rerun, are the source for figure reconstruction.

Our original code and generated metric tables are MIT-licensed; third-party
packages remain under their respective licenses. If you use this repository,
please cite the paper and record the repository commit used for your analysis.
