# Retrain ProKcat From Scratch on Custom TVT Splits

This guide is a practical end-to-end process for retraining this repository on your own `train/val/test` splits.

It assumes:
- You already have split files (`train.csv`, `val.csv`, `test.csv`)
- You will handle schema mapping and temperature columns yourself
- You want true training from scratch (no pretrained checkpoint warm start)

## 1. Environment Setup

From repo root:

```bash
cd /home/adhil/github/ProKcat
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision torchaudio
pip install numpy pandas scipy scikit-learn rdkit-pypi fair-esm
```

Notes:
- Use a CUDA-compatible PyTorch build if you train on GPU.
- `fair-esm` is required by `feature_functions.py`.

## 2. Prepare Data Files in This Repo

Place your processed files in the repository `data` directory with these names:

- `data/train_data.csv`
- `data/val_data.csv`
- `data/test_data.csv`

Also ensure these directories exist:

```bash
mkdir -p data/dict
mkdir -p data/train_features data/val_features data/test_features
mkdir -p data/my_performance
```

## 3. Generate Features for Each Split

The training loaders expect feature folders with this naming contract:
- `train_features/*.pkl`
- `val_features/*.pkl`
- `test_features/*.pkl`

Run feature generation split-by-split from the `code` directory.

### 3.1 Important toggle in `code/gen_features.py`

`gen_features.py` currently calls `get_esm_features(...)` by default.
For baseline DLTKcat training in `DLTKcat_run_train_test.py`, switch it to `get_features(...)`.

In `code/gen_features.py`, replace:

```python
get_esm_features(data_path, output_path, radius, ngram, has_dict, dict_path, has_label)
```

with:

```python
get_features(data_path, output_path, radius, ngram, has_dict, dict_path, has_label)
```

### 3.2 Run feature generation

```bash
cd code

# Train split builds dictionaries
python gen_features.py \
  --data ../data/train_data.csv \
  --output ../data/train_features \
  --radius 2 --ngram 3 \
  --has_dict False \
  --dict_path ../data/dict \
  --has_label True

# Val split reuses dictionaries
python gen_features.py \
  --data ../data/val_data.csv \
  --output ../data/val_features \
  --radius 2 --ngram 3 \
  --has_dict True \
  --dict_path ../data/dict \
  --has_label True

# Test split reuses dictionaries
python gen_features.py \
  --data ../data/test_data.csv \
  --output ../data/test_features \
  --radius 2 --ngram 3 \
  --has_dict True \
  --dict_path ../data/dict \
  --has_label True
```

## 4. Mandatory Code Fixes Before Training

There are a few inconsistencies in the current repo that block a clean scratch retrain.

### 4.1 Fix function name mismatch in `code/train_functions.py`

The file defines:
- `DLTKcat_batch2tensor`
- `DLTKcat_train_eval`
- `DLTKcat_test`

But internally calls `batch2tensor`, `train_eval`, `test`.

Add aliases at the end of `code/train_functions.py`:

```python
batch2tensor = DLTKcat_batch2tensor
train_eval = DLTKcat_train_eval
test = DLTKcat_test
```

### 4.2 Update training entry script for real TVT

In `code/DLTKcat_run_train_test.py`:

1. Add a `--val_path` argument.
2. Replace hardcoded `/usr/data/...` paths with paths relative to your local repo.
3. Remove pretrained model loading for true scratch.
4. Load `val` features directly instead of random split from `train`.

Conceptually:

```python
val_data = load_data(val_path, True, 'val')
train_data = load_data(train_path, True, 'train')
test_data = load_data(test_path, True, 'test')

# remove M.load_state_dict(...)

train_eval(M, train_data, test_data, val_data, ...)
```

### 4.3 Ensure output folders exist

Training saves checkpoints and CSV logs into `data/my_performance`. Create it before running.

## 5. Train From Scratch

From `code` directory, after fixes:

```bash
cd /home/adhil/github/ProKcat/code
CUDA_VISIBLE_DEVICES=0 python DLTKcat_run_train_test.py \
  --train_path ../data \
  --val_path ../data \
  --test_path ../data \
  --param_dict_pkl ../data/hyparams/param_2.pkl \
  --lr 0.001 \
  --batch 16 \
  --lr_decay 0.5 \
  --decay_interval 5 \
  --num_epoch 40
```

Why the same root path for all three:
- `load_data(path, ..., 'train')` resolves to `path/train_features/...`
- `load_data(path, ..., 'val')` resolves to `path/val_features/...`
- `load_data(path, ..., 'test')` resolves to `path/test_features/...`

## 6. Evaluate and Export Predictions

You can reuse `code/DLTKcat_predict.py`, but it also has hardcoded path assumptions.
Update it similarly to use local `../data` paths and your selected checkpoint file.

Expected outputs:
- Trained model checkpoints in `data/my_performance`
- Metrics CSV in `data/my_performance`
- Optional prediction CSV from `DLTKcat_predict.py`

## 7. Recommended Run Validation Checklist

Before launching full training:

1. Confirm feature files exist:

```bash
ls ../data/train_features
ls ../data/val_features
ls ../data/test_features
```

2. Confirm dict files exist:

```bash
ls ../data/dict
```

3. Quick sanity run:
- Start with `--num_epoch 1` and small batch to ensure pipeline is valid.

4. Full run:
- Increase epochs and monitor RMSE/R2/PCC/MAE per epoch.

## 8. Common Failure Modes

1. `File ... does not exist`:
- Usually feature folders or dict path mismatch.

2. CUDA OOM:
- Reduce `--batch`.

3. Script still reading `/usr/data/...`:
- Missed one hardcoded path replacement.

4. NameError for `batch2tensor` / `train_eval` / `test`:
- Alias fix in `train_functions.py` not applied.

5. Invalid molecule errors from RDKit during feature generation:
- Upstream SMILES sanitation needed in your split files.

## 9. Optional: ESM Finetune Path

If you specifically want ESM finetuning (`run_esm_Kcat_finetune.py`), use the `_features_esm_seq` folder contract and align feature generation to `get_esm_seq_features(...)`.

That path is heavier and less clean than the baseline path. For first successful retraining on custom splits, start with baseline DLTKcat flow above.
