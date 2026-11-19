# MST-CP

Micro-Scale Transport Conformal Prediction for cohort sums under distribution shift.

MST-CP builds prediction intervals for the sum of labels over a cohort of 25 to 100 units. A model trained on a source population is deployed on a shifted target population, and only a small labelled target audit (100 to 300 rows) is available. This repository holds the implementation, the experiment scripts, the stored results, and a script that verifies those results. The accompanying paper is not included.

## Method

1. A base predictor, gradient boosting by default, is trained on the source. A local ridge drift correction models its residuals on the labelled target audit.
2. A two-scale studentiser `s_k^2 = kappa * sbar^2_S(B)/k + v_ep(B) + tau^2` separates a row-noise variance that averages like `1/k` from leverage and random-effect terms that do not.
3. The studentised score is calibrated on micro-cohorts of `m0` in {4, 5, 8} target rows, so a few hundred labels give enough calibration cohorts. The conformal threshold `q` is then applied at the target cohort size `m`, with the studentiser evaluated at `m`. The interval for the cohort sum has half-width `m * q * s_m` and is capped at the source label range.

Proposition 1 of the paper shows that no deterministic, everywhere-finite threshold computed from `K` group scores alone has worst-case miscoverage below `1/(K+1)`. Calibrating directly at the cohort size therefore needs at least nine target groups for a finite 90% interval.

## Theory and results

The theoretical results (a bridging bound, a variance-inflation corollary, and a noise and drift crossover scale) hold for an idealised pivot under assumptions A1 to A3. The implemented estimator has no finite-sample distribution-free guarantee. Its plug-in studentiser and the i.i.d. assumption on micro-cohort scores (A2) fall outside the theory.

Everything reported for the implementation is empirical coverage on ten shifted workloads built from five public datasets, plus five IID workloads. With 200 audit labels, mean coverage over the ten shifted workloads is 0.92 (worst workload 0.90), and the mean Winkler score is 0.61 times that of an uncorrected labelled-target oracle. That oracle is centred at the source prediction, so a ratio below one reflects the drift correction as much as the calibration.

The method has these limitations:

- Coverage can fail when the audit labels are not a random sample of the target. With labels drawn only from the earliest 25% of the target period, MST-CP v1.0 covers 0.81 on house sales S2 at `n_T = 200` (0.84 at `n_T = 100` and 0.82 at `n_T = 300`) and 0.93 on bike S1. On the eight non-temporal shifted workloads, with labels drawn from the quarter of the target region nearest the source, mean coverage is 0.65, the lowest workload is 0.08, and 2 of 8 reach 0.85.
- In the exploratory refresh evaluation on seeds 6 to 15, with 200 labels acquired in every strategy, coverage with all labels taken from the initial window is 0.78 on house sales S2 and 0.89 on bike S1. With four blocks of 50 labels it is 0.96 and 0.92, and with two blocks of 100 labels it is 0.95 and 0.87.
- Without shift it is wider than source-calibrated intervals: on the five IID workloads at `n_T = 200` the mean Winkler ratio is 1.44 against 0.99.
- It is wider than drift-corrected oracles that use the same drift model and calibrate cohorts of the target size on the labelled target pool. At `n_T = 200` the Winkler ratio is 1.08 against the oracle DC_a and 1.39 against DC_b, on all ten shifted workloads for both, and those oracles cover 0.87 and 0.84.
- With a ridge or logistic base predictor, mean coverage reaches 0.85 on only 8 of the 10 workloads on seeds 3 to 5 (lowest 0.84) and on 8 of 10 on seeds 6 to 15 (lowest 0.84). On seeds 6 to 15 a quadratic drift basis, a post hoc variant that is not the submitted method, gives mean coverage 0.91 against 0.89 without it. The predictor-family experiments (MLP, random forest, ridge or logistic) use `n_T = 200` only.
- The temporal results rest on two data paths (house sales S2 and bike S1). The sequential-audit analysis, the refresh evaluation and the quadratic-drift variant were added after the confirmation seeds and are exploratory.

The repository supports no claim of general robustness under temporal shift or of independence from the base predictor.

## Installation

The stored results were produced with Python 3.12.15. The code uses NumPy, SciPy, scikit-learn and pandas.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` pins the versions of the environment that produced the stored results. Running experiments needs network access, because the datasets are downloaded on first use. Verifying the stored results does not.

## Verifying the stored results

```bash
python scripts/verify_results.py
```

The script runs in about a second, executes no experiment, and uses only the standard library. It prints `VERIFICATION PASSED` when every check succeeds. It checks four things, each reported under a numbered check:

- **Integrity** (`[0a]` to `[0e]`): the SHA-256 of every stored result file against `results/SHA256SUMS`, the canonical hashes of the 45 `MSTLOCKA` payloads against `LOCK_V10_FILE_HASHES.txt`, and the md5 sums in `CONFIRM2_MANIFEST.txt`.
- **Code** (`[0d]`): a fingerprint of each script's syntax tree, so that comments and formatting do not matter. The fingerprints were computed with Python 3.12, and the check is skipped on other versions.
- **Reported values** (`[1a]` to `[2f]`): the table cells and statistics in `results/reported_values.json`, taken from the numbers printed in the paper, are recomputed in plain Python from the raw result files and compared. Every entry of that file must be compared, or the check fails.
- **This README** (`[3]`): each number in the "Theory and results" section is recomputed from the stored artifacts, and the sentence containing it must appear verbatim here.

Several other stored quantities are not recomputed independently: the Berry-Esseen proxy, the v1.1 shifted-workload cost, the seed-0 diagnostics, the ordered-audit and grid tables, and the preliminary `cov_vs_m.txt` input.

The hash checks `[0c]` and `[0e]` describe the repository as released. After results are regenerated, `[0e]` fails because regenerated files differ in wall-clock fields, and `[0c]` fails if `AUX_REVISIONS.json` was regenerated. `[0a]` is the check for regenerated `MSTLOCKA` files.

## Reproducing the experiments

All scripts sit in the repository root and are run from there.

```bash
bash run_lock.sh
python scripts/verify_results.py
python lock_verify.py
python lock_summ_mst.py 3,4,5 lockfinal && shasum -a 256 MST_RESULTS_lockfinal.txt

PYTHONHASHSEED=0 python aux_revisions.py
python be_proxy_lock.py
mkdir -p out && python confirm12.py && python confirm3.py
```

`run_lock.sh` runs the full confirmation experiment for seeds 3 to 5 (about 6 minutes on 14 cores in the authors' run). Set `PY=/path/to/python` to choose the interpreter. It writes 165 files into `results/` (45 `MSTLOCKA`, 45 `MSTLOCKB`, 45 `V11RLOCK`, 30 `V11OLOCK`) and overwrites the stored ones, so run it in a fresh clone or copy `results/` first. The `MSTLOCKB` files are a second run with a different worker-pool size, which `lock_verify.py` compares with run A (it should report 45 of 45 identical and writes `LOCK_V10_FILE_HASHES.txt`). The `lock_summ_mst.py` command writes `MST_RESULTS_lockfinal.txt`, whose SHA-256 should be `b0f1ff5aa92d63dde4ff952fcfc6e3796a780b536778707757acf0f1341cc565`.

| Script | Output | Content |
|---|---|---|
| `mst.py` (`TAG=LOCKA`) | `MSTLOCKA_<dataset>_<shift>_<seed>.json` | MST-CP v1.0 and all baselines, random-sample audit labels, 10 shifted and 5 IID workloads, seeds 3 to 5 |
| `mst2.py` (`TAG=LOCK`) | `V11RLOCK_*.json` | Same draws: v1.0 path, shrinkage variant, ablations, and the labelled-target oracle O1 |
| `mst2.py` (`TAG=LOCK ORDERED=1 FWD=1`) | `V11OLOCK_*.json` | Ordered (early or near-source) audit labels on the 10 shifted workloads |
| `aux_revisions.py` | `AUX_REVISIONS.json` | Plug-in thresholds, original-scale evaluation, cap activation, drift-corrected oracles and predictor families on the same seeds and draws (about 5 minutes) |
| `confirm12.py`, `confirm3.py` | `CONFIRM2_predictor_L0_L2_seeds6-15.json`, `CONFIRM2_rolling_audit_seeds6-15.json` | Unused seeds 6 to 15: predictor sensitivity (boosting, ridge or logistic, quadratic-drift variant) and rolling temporal audits. They write to `out/` for comparison with the stored files |
| `be_proxy_lock.py` | `be_proxy_lock.json` | Lyapunov-ratio proxy for the Berry-Esseen remark (seed 0, descriptive) |

The other stored files are `LOCK_V10_FILE_HASHES.txt` (canonical digests of the 45 v1.0 payloads), `LOCK_CONFIRMATION_TABLE_v10_runA.txt` (the confirmation table from an older, independent summariser), `CONFIRM2_MANIFEST.txt` (md5 sums of the final-round artifacts), `cov_vs_m.txt` (a preliminary study with seeds 0 to 2; the script that generated it is not part of this repository), `SHA256SUMS` and `reported_values.json`. The result JSON files are stored exactly as the scripts wrote them.

## Data

`common.py` downloads the datasets into `data/` (git-ignored) on first use. The OpenML datasets are pinned by dataset id.

| Name in code | Source | Identifier | Task and target |
|---|---|---|---|
| `california` | `sklearn.datasets.fetch_california_housing` | scikit-learn download of `cal_housing_py3.pkz` | regression |
| `house_sales` | OpenML, `house_sales` version 1 | dataset id 42079 | regression on `log(price)`, ordered by `date` |
| `bike` | OpenML, `Bike_Sharing_Demand` version 2 | dataset id 42712 | regression on `log1p(count)`, ordered by time |
| `adult` | OpenML, `adult` version 2 | dataset id 1590 | classification, income above 50K as 0 or 1 |
| `wine_quality` | OpenML, `wine_quality` version 1 | dataset id 287 | regression |

The shift variable, its direction, and the IID split of each workload are defined by `SPLITS` and `make_split` in `common.py`.

## Seeds and reproducibility

- Confirmation seeds are 3, 4 and 5 (`MSTLOCKA`, `V11RLOCK`, `V11OLOCK`, `AUX_REVISIONS`). Unused seeds 6 to 15 are in `CONFIRM2_*`, seed 0 is used for the Berry-Esseen proxy, and seeds 0 to 2 for `cov_vs_m.txt`. Development seeds are not part of this repository.
- Protocol constants are miscoverage `alpha = 0.1`; audit sizes `n_T` in {100, 200, 300}; cohort sizes `m` in {25, 50, 100}; cohort mechanisms kNN, slice and decile, pooled; 20 nested label draws per seed; and a micro-cohort size `m0` in {4, 5, 8} chosen from `n_T`.
- Every random generator is seeded. The one remaining source of non-determinism is scikit-learn's brute-force nearest-neighbour search, which breaks exact distance ties differently for different OpenMP thread counts, and the house sales data contain duplicate rows. `run_lock.sh`, `aux_revisions.py` and `sbx.py` therefore set `OMP_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS`, `OPENBLAS_NUM_THREADS` and `MKL_NUM_THREADS` to 1 and `PYTHONHASHSEED` to 0. These variables must be exported before `mst.py` or `mst2.py` is called directly.
- The released scripts differ from the code that produced the stored results only in comments, formatting, removed progress printing (including a print-only pilot mode in `inv3.py`) and, in `common.py`, the OpenML dataset-id arguments, which replace a lookup by name and version that resolved to the same datasets. Computations, parameters and seeds are unchanged.
- The authors reproduced the stored results bit for bit on one machine with the pinned versions and two worker-pool sizes. In a fresh Python 3.12 environment, the released scripts reproduced the stored outputs exactly (wall-clock fields excluded) for the seed-3 jobs of `mst.py` (15 of 45 files) and of `mst2.py` (all 25 seed-3 files, random and ordered audits), all seed-3 jobs of every `aux_revisions.py` check, 60 of the 300 `confirm12.py` jobs, `confirm3.py` in full, `be_proxy_lock.py` (byte-identical output) and `lock_summ_mst.py` (run on the regenerated and stored `MSTLOCKA` files, giving the SHA-256 above). The complete `run_lock.sh` and `aux_revisions.py` runs were not repeated, and other platforms and BLAS builds were not tested.

## Layout

```
common.py, mst.py, mst2.py            data loading, splits, baselines, MST-CP v1.0 (mst.py), ablations and ordered audits (mst2.py)
aux_revisions.py, be_proxy_lock.py    auxiliary analyses on the same draws
sbx.py, inv12_full.py, inv3.py,
confirm12.py, confirm3.py             final confirmation round (seeds 6 to 15)
run_lock.sh, lock_verify.py,
lock_summ_mst.py                      full run from scratch and its checks
scripts/verify_results.py             verification of the stored results
results/                              frozen results and provenance files
```

The scripts stay in the root because they import one another by name and write to `results/` and `data/` relative to their own location.

## License

The source code is released under the [MIT License](LICENSE).

The experimental results are released under the
[Creative Commons Attribution 4.0 International License](results/LICENSE).

The datasets are not redistributed. Their sources and identifiers are listed in the Data section.
