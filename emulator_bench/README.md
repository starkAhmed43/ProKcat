# emulator_bench

This folder adds a CataPro-style TVT workflow for ProKcat while keeping feature/model logic close to the authors' notebook implementation.

## What is similar to CataPro emulator_bench

- `feature_utils.py`: reusable featurization helpers + persistent per-item cache
- `build_tvt_features.py`: build one split's features with cache reuse
- `train_prokcat_mlp_tvt.py`: train on explicit train/val/test (no CV folding)
- `run_split_benchmarks.py`: run all thresholds across split families for one value type

## What is reused from original ProKcat notebooks

The following logic is kept intentionally close to original implementation in `code/run_train_test.ipynb`:

1. Molecular graph and fingerprint construction
- based on original helpers in `code/feature_functions.py`:
  - atom dictionary lookup/update
  - bond dictionary creation
  - radius-based graph fingerprint IDs
  - adjacency with self-loop
  - 1024-bit Morgan fingerprint

2. Sequence tokenization
- same n-gram token ID behavior as `split_sequence` in `code/feature_functions.py`

3. Label transform
- same `log10(target)` convention used in original feature generation (`log10_kcat.pkl`)

4. Model architecture and MLP head
- notebook-derived classes are pulled into:
  - `original_prokcat_mlp.py`
  - `original_data_utils.py`
- source notebook block references are commented inline in those files.
- we use only the MLP path (`New_MLP_Independent` equivalent), not KAN.

5. Tensor conversion + metrics
- notebook-compatible utilities are copied into `original_data_utils.py`
- this avoids depending on baseline runner scripts.

## Input schema expected by builder

Required columns:
- sequence column (default: `sequence`)
- smiles column (default: `smiles`)
- target column (default passed by `--target_col`)

Temperature fields:
- preferred: `Temp_K_norm` and `Inv_Temp_norm`
- if missing, script will derive them from `Temp` or `Temp_K`

## Step 1: Build split features with cache

Example (train split):

```bash
python emulator_bench/build_tvt_features.py \
  --input_path /path/to/train.csv \
  --output_root /path/to/threshold_dir/prokcat_features \
  --split_name train \
  --dict_dir data/dict \
  --has_dict False \
  --target_col value \
  --sequence_col sequence \
  --smiles_col smiles \
  --cache_dir emulator_bench/.cache_embeddings
```

For val/test use `--has_dict True`.

If your target column is already in log10 space (for example `log10_value`), add:

```bash
--target_is_log10
```

Each split feature folder includes `feature_meta.json` with source/transform metadata.

## Step 2: Train on explicit TVT

```bash
python emulator_bench/train_prokcat_mlp_tvt.py \
  --data_root /path/to/threshold_dir/prokcat_features \
  --dict_dir data/dict \
  --param_dict_pkl data/hyparams/param_2.pkl \
  --out_dir /path/to/threshold_dir/prokcat_results \
  --epochs 40 \
  --batch_size 16 \
  --device cuda:0
```

## Step 3: Run all thresholds in one command

```bash
python emulator_bench/run_split_benchmarks.py \
  --base_dir /home/adhil/github/EMULaToR/data/processed/baselines/ProKcat \
  --value_type kcat \
  --target_col log10_value \
  --sequence_col sequence \
  --smiles_col smiles \
  --cache_dir emulator_bench/.cache_embeddings \
  --device cuda:0 \
  --seeds 0 1 2 3 4 \
  --primary_metric MSE
```

Expected per-threshold outputs:
- `prokcat_features/train_features/*`
- `prokcat_features/val_features/*`
- `prokcat_features/test_features/*`
- `prokcat_features/*/feature_meta.json`
- `prokcat_results/seed_<seed>/best_backbone.pth`
- `prokcat_results/seed_<seed>/best_mlp_head.pth`
- `prokcat_results/seed_<seed>/logfile.csv`
- `prokcat_results/seed_<seed>/final_results_val.csv`
- `prokcat_results/seed_<seed>/final_results_test.csv`
- `prokcat_results/seed_<seed>/pred_label_val.csv`
- `prokcat_results/seed_<seed>/pred_label_test.csv`

Summary files under `<base_dir>/<value_type>/`:
- `prokcat_summary_runs.csv` (one row per seed run)
- `prokcat_summary_thresholds.csv` (mean/variance across seeds)
- `prokcat_summary_by_split_group.csv` (enzyme/substrate aggregates)
- `prokcat_summary_ranked.csv` (ranked by primary metric)
- `prokcat_summary.csv` (backward-compatible threshold summary)

## Notes

- This workflow is cache-first and split-friendly for many thresholds.
- It keeps original ProKcat featurization/model behavior from notebook code, but wraps it in a benchmark-friendly orchestration layer.

Useful sweep options:
- `--seeds 0 1 2 3 4`
- `--ratio_tolerance 0.02`
- `--primary_metric MSE`
- `--higher_is_better` (use for metrics like PCC/R2)

## Step 4: Tune hyperparameters with Optuna

`tune_optuna.py` uses the same split discovery and feature-building flow, then runs multi-seed training inside each trial and optimizes a selected metric.

Example:

```bash
python emulator_bench/tune_optuna.py \
  --base_dir /home/adhil/github/EMULaToR/data/processed/baselines/ProKcat \
  --value_type kcat \
  --split_groups enzyme_sequence_splits substrate_splits \
  --target_col log10_value \
  --sequence_col sequence \
  --smiles_col smiles \
  --metric MSE \
  --eval_split val \
  --seeds 41 42 43 44 45 \
  --n_trials 40 \
  --max_jobs 10 \
  --epochs 80 \
  --device cuda:0
```

Generated artifacts (under `<base_dir>/<value_type>/optuna_studies`):
- `<study_name>_best_hparams.json`
- `<study_name>_trials.csv`

Notes:
- `--separate_by_split_group` runs one study per split family.
- `--storage sqlite:///...` enables persistent Optuna studies.
- `--parallel_runs_per_trial` controls concurrency across seed/split runs inside each trial.
- `--trial_parallelism` controls Optuna trial-level concurrency.

## Step 5: Re-run benchmarks with tuned hyperparameters

Pass the tuned JSON into the benchmark runner to override train settings:

```bash
python emulator_bench/run_split_benchmarks.py \
  --base_dir /home/adhil/github/EMULaToR/data/processed/baselines/ProKcat \
  --value_type kcat \
  --hparams_json /home/adhil/github/EMULaToR/data/processed/baselines/ProKcat/kcat/optuna_studies/prokcat_kcat_mse_best_hparams.json \
  --seeds 41 42 43 44 45 \
  --primary_metric MSE
```
