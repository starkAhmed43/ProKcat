import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm


def discover_threshold_dirs(value_root: Path, split_groups, explicit_thresholds=None):
    jobs = []
    for split_group in split_groups:
        split_root = value_root / split_group
        if not split_root.exists():
            continue

        if explicit_thresholds:
            threshold_dirs = [split_root / t for t in explicit_thresholds]
        else:
            threshold_dirs = [p for p in sorted(split_root.iterdir()) if p.is_dir() and p.name.startswith("threshold_")]

        for threshold_dir in threshold_dirs:
            if threshold_dir.exists():
                jobs.append((split_group, threshold_dir.name, threshold_dir))
    return jobs


def ensure_triplet(threshold_dir: Path):
    train = threshold_dir / "train.csv"
    val = threshold_dir / "val.csv"
    test = threshold_dir / "test.csv"
    if not train.exists() and (threshold_dir / "train.parquet").exists():
        train = threshold_dir / "train.parquet"
    if not val.exists() and (threshold_dir / "val.parquet").exists():
        val = threshold_dir / "val.parquet"
    if not test.exists() and (threshold_dir / "test.parquet").exists():
        test = threshold_dir / "test.parquet"
    return train, val, test


def _threshold_to_float(name: str):
    try:
        return float(str(name).split("threshold_")[-1])
    except Exception:
        return float("inf")


def _load_table_len(path: Path):
    if path.suffix.lower() == ".csv":
        return len(pd.read_csv(path))
    if path.suffix.lower() in {".parquet", ".pq"}:
        return len(pd.read_parquet(path))
    raise ValueError(f"Unsupported split file extension: {path.suffix}")


def get_split_meta(train_path: Path, val_path: Path, test_path: Path, ratio_tolerance: float):
    train_size = _load_table_len(train_path)
    val_size = _load_table_len(val_path)
    test_size = _load_table_len(test_path)
    total = train_size + val_size + test_size

    if total == 0:
        train_ratio = 0.0
        val_ratio = 0.0
        test_ratio = 0.0
    else:
        train_ratio = train_size / total
        val_ratio = val_size / total
        test_ratio = test_size / total

    target = (0.8, 0.1, 0.1)
    small_split_flag = int(
        abs(train_ratio - target[0]) > ratio_tolerance
        or abs(val_ratio - target[1]) > ratio_tolerance
        or abs(test_ratio - target[2]) > ratio_tolerance
    )

    return {
        "train_size": train_size,
        "val_size": val_size,
        "test_size": test_size,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "small_split_flag": small_split_flag,
    }


def run_cmd(cmd, dry_run=False):
    print(" ".join(cmd))
    if not dry_run:
        subprocess.run(cmd, check=True)


def _is_oom_failure(exc: subprocess.CalledProcessError):
    text = f"{exc.output or ''}\n{exc.stdout or ''}\n{exc.stderr or ''}".lower()
    return (
        "out of memory" in text
        or "cuda oom" in text
        or "cublas_status_alloc_failed" in text
    )


def _run_and_stream(cmd):
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    chunks = []
    assert proc.stdout is not None
    while True:
        ch = proc.stdout.read(1)
        if ch == "":
            break
        print(ch, end="", flush=True)
        chunks.append(ch)

    return_code = proc.wait()
    output = "".join(chunks)
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, cmd, output=output)


def build_split_features(csv_or_parquet, split_name, feats_root, args, has_dict):
    split_features_dir = Path(feats_root) / f"{split_name}_features"
    marker = split_features_dir / "log10_kcat.pkl"
    if marker.exists() and not args.overwrite:
        print(f"[feature-skip] Reusing existing {split_name} features: {split_features_dir}")
        return

    cmd = [
        sys.executable,
        "emulator_bench/build_tvt_features.py",
        "--input_path",
        str(csv_or_parquet),
        "--output_root",
        str(feats_root),
        "--split_name",
        split_name,
        "--dict_dir",
        str(args.dict_dir),
        "--has_dict",
        "True" if has_dict else "False",
        "--target_col",
        args.target_col,
        "--sequence_col",
        args.sequence_col,
        "--smiles_col",
        args.smiles_col,
        "--temp_col",
        args.temp_col,
        "--radius",
        str(args.radius),
        "--ngram",
        str(args.ngram),
        "--cache_dir",
        str(args.cache_dir),
    ]
    if args.target_is_log10:
        cmd.append("--target_is_log10")
    if args.no_cache_read:
        cmd.append("--no_cache_read")
    if args.no_cache_write:
        cmd.append("--no_cache_write")
    run_cmd(cmd, dry_run=args.dry_run)


def run_train(feats_root, out_dir, args, seed):
    batch_candidates = [args.batch_size] + [b for b in [8, 4, 2, 1] if b < args.batch_size]
    trainer_script = "emulator_bench/train_prokcat_mlp_tvt_amp.py" if args.use_amp_trainer else "emulator_bench/train_prokcat_mlp_tvt.py"

    for i, batch_size in enumerate(batch_candidates):
        cmd = [
            sys.executable,
            trainer_script,
            "--data_root",
            str(feats_root),
            "--dict_dir",
            str(args.dict_dir),
            "--param_dict_pkl",
            str(args.param_dict_pkl),
            "--out_dir",
            str(out_dir),
            "--epochs",
            str(args.epochs),
            "--batch_size",
            str(batch_size),
            "--lr",
            str(args.lr),
            "--lr_decay",
            str(args.lr_decay),
            "--decay_interval",
            str(args.decay_interval),
            "--seed",
            str(seed),
            "--device",
            args.device,
        ]

        if args.use_amp_trainer:
            if args.use_amp:
                cmd.append("--use_amp")
            cmd.extend(["--grad_accum_steps", str(args.grad_accum_steps)])
            cmd.extend(["--max_grad_norm", str(args.max_grad_norm)])

        if args.dry_run:
            run_cmd(cmd, dry_run=True)
            return

        print(" ".join(cmd))
        try:
            _run_and_stream(cmd)
            if i > 0:
                print(f"[oom-retry] training succeeded with reduced batch_size={batch_size}")
            return
        except subprocess.CalledProcessError as e:
            if e.output:
                print(e.output)

            if i < len(batch_candidates) - 1 and _is_oom_failure(e):
                next_bs = batch_candidates[i + 1]
                print(f"[oom-retry] CUDA OOM with batch_size={batch_size}; retrying with batch_size={next_bs}")
                continue
            raise


def main():
    parser = argparse.ArgumentParser(description="Run ProKcat TVT benchmark across split families and thresholds.")
    parser.add_argument(
        "--base_dir",
        default="/home/adhil/github/EMULaToR/data/processed/baselines/ProKcat",
        type=str,
    )
    parser.add_argument("--value_type", required=True, choices=["kcat", "km", "ki"], type=str)
    parser.add_argument("--split_groups", nargs="+", default=["enzyme_sequence_splits", "substrate_splits"])
    parser.add_argument("--thresholds", nargs="+", default=None)

    parser.add_argument("--target_col", default="log10_value", type=str)
    parser.add_argument(
        "--target_is_log10",
        action="store_true",
        help="Set when --target_col in split CSV/parquet is already log10-transformed.",
    )
    parser.add_argument(
        "--target_is_raw",
        action="store_true",
        help="Set when --target_col in split CSV/parquet is raw and should be log10-transformed.",
    )
    parser.add_argument("--sequence_col", default="sequence", type=str)
    parser.add_argument("--smiles_col", default="smiles", type=str)
    parser.add_argument("--temp_col", default="Temperature", type=str)

    parser.add_argument("--dict_dir", default="data/dict", type=str)
    parser.add_argument("--param_dict_pkl", default="data/hyparams/param_2.pkl", type=str)
    parser.add_argument("--cache_dir", default="emulator_bench/.cache_embeddings", type=str)
    parser.add_argument("--no_cache_read", action="store_true")
    parser.add_argument("--no_cache_write", action="store_true")

    parser.add_argument("--radius", default=2, type=int)
    parser.add_argument("--ngram", default=3, type=int)

    parser.add_argument("--epochs", default=40, type=int)
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--lr_decay", default=0.5, type=float)
    parser.add_argument("--decay_interval", default=5, type=int)
    parser.add_argument("--use_amp_trainer", action="store_true", help="Use separate AMP/grad-accum trainer file.")
    parser.add_argument("--use_amp", action="store_true", help="Enable AMP in AMP trainer mode.")
    parser.add_argument("--grad_accum_steps", default=1, type=int, help="Gradient accumulation steps (AMP trainer mode).")
    parser.add_argument("--max_grad_norm", default=0.0, type=float, help="Gradient clipping; <=0 disables clipping (AMP trainer mode).")
    parser.add_argument(
        "--hparams_json",
        type=str,
        default=None,
        help="Optional JSON file with tuned hyperparameters to override train settings.",
    )
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--seeds", nargs="+", type=int, default=None, help="Seed sweep list. Overrides --seed.")
    parser.add_argument("--ratio_tolerance", type=float, default=0.02, help="Tolerance for 80:10:10 split ratio flagging.")
    parser.add_argument(
        "--primary_metric",
        type=str,
        default="MSE",
        choices=["PCC", "SCC", "R2", "RMSE", "MSE", "MAE"],
        help="Metric used for ranking threshold summaries.",
    )
    parser.add_argument("--higher_is_better", action="store_true", help="Set ranking direction for primary metric.")
    parser.add_argument("--device", default="cuda:0", type=str)

    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry_run", action="store_true")

    args = parser.parse_args()

    if args.hparams_json:
        with open(args.hparams_json, "r", encoding="utf-8") as f:
            hp = json.load(f)

        key_map = {
            "batch_size": int,
            "train_batch_size": int,
            "lr": float,
            "epochs": int,
            "lr_decay": float,
            "decay_interval": int,
            "seed": int,
        }
        for k, caster in key_map.items():
            if k in hp:
                if k == "train_batch_size":
                    args.batch_size = caster(hp[k])
                else:
                    setattr(args, k, caster(hp[k]))

        print(f"Loaded hyperparameters from {args.hparams_json}")

    # Convenience default: log10_value is already log10.
    args.target_is_log10 = True if not args.target_is_raw else False

    base_dir = Path(args.base_dir)
    value_root = base_dir / args.value_type
    jobs = discover_threshold_dirs(value_root, args.split_groups, args.thresholds)
    seed_list = args.seeds if args.seeds is not None else [args.seed]

    if not jobs:
        raise RuntimeError("No threshold jobs found.")

    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<not-set>')}")
    print(f"Discovered {len(jobs)} jobs for value_type={args.value_type}")

    run_rows = []
    for split_group, threshold_name, threshold_dir in tqdm(jobs, desc="ProKcat benchmark", unit="job"):
        train_path, val_path, test_path = ensure_triplet(threshold_dir)
        if not (train_path.exists() and val_path.exists() and test_path.exists()):
            print(f"[skip] missing train/val/test in {threshold_dir}")
            continue

        split_meta = get_split_meta(train_path, val_path, test_path, args.ratio_tolerance)

        feats_root = threshold_dir / "prokcat_features"
        out_dir = threshold_dir / "prokcat_results"
        feats_root.mkdir(parents=True, exist_ok=True)
        out_dir.mkdir(parents=True, exist_ok=True)

        build_split_features(train_path, "train", feats_root, args, has_dict=False)
        build_split_features(val_path, "val", feats_root, args, has_dict=True)
        build_split_features(test_path, "test", feats_root, args, has_dict=True)

        for seed in seed_list:
            seed_out_dir = out_dir / f"seed_{seed}"
            marker = seed_out_dir / "final_results_test.csv"

            if not marker.exists() or args.overwrite:
                run_train(feats_root, seed_out_dir, args, seed)

            if marker.exists():
                row = pd.read_csv(marker).iloc[0].to_dict()
                row["value_type"] = args.value_type
                row["split_group"] = split_group
                row["threshold"] = threshold_name
                row["seed"] = seed
                row["results_dir"] = str(seed_out_dir)
                row.update(split_meta)
                run_rows.append(row)

    if run_rows:
        runs_df = pd.DataFrame(run_rows)
        runs_df["threshold_num"] = runs_df["threshold"].map(_threshold_to_float)
        runs_df = runs_df.sort_values(["split_group", "threshold_num", "seed"]).drop(columns=["threshold_num"])

        runs_path = value_root / "prokcat_summary_runs.csv"
        runs_df.to_csv(runs_path, index=False)

        metric_cols = [c for c in ["PCC", "SCC", "R2", "RMSE", "MSE", "MAE"] if c in runs_df.columns]
        group_cols = ["value_type", "split_group", "threshold"]

        threshold_rows = []
        for keys, g in runs_df.groupby(group_cols, sort=False):
            row = dict(zip(group_cols, keys))
            row["n_seeds"] = int(g["seed"].nunique())
            for c in ["train_size", "val_size", "test_size", "train_ratio", "val_ratio", "test_ratio", "small_split_flag"]:
                row[c] = g[c].iloc[0]
            for m in metric_cols:
                row[f"{m}_mean"] = float(g[m].mean())
                row[f"{m}_var"] = float(g[m].var(ddof=1)) if len(g) > 1 else 0.0
            threshold_rows.append(row)

        threshold_df = pd.DataFrame(threshold_rows)
        threshold_df["threshold_num"] = threshold_df["threshold"].map(_threshold_to_float)
        threshold_df = threshold_df.sort_values(["split_group", "threshold_num"]).drop(columns=["threshold_num"])

        threshold_path = value_root / "prokcat_summary_thresholds.csv"
        threshold_df.to_csv(threshold_path, index=False)

        compat_path = value_root / "prokcat_summary.csv"
        threshold_df.to_csv(compat_path, index=False)

        by_split_rows = []
        for split_group, g in threshold_df.groupby("split_group", sort=False):
            row = {"value_type": args.value_type, "split_group": split_group, "n_thresholds": len(g)}
            for m in metric_cols:
                row[f"{m}_mean_over_thresholds"] = float(g[f"{m}_mean"].mean())
                row[f"{m}_var_over_thresholds"] = float(g[f"{m}_mean"].var(ddof=1)) if len(g) > 1 else 0.0
            by_split_rows.append(row)

        by_split_df = pd.DataFrame(by_split_rows)
        by_split_path = value_root / "prokcat_summary_by_split_group.csv"
        by_split_df.to_csv(by_split_path, index=False)

        metric_key = f"{args.primary_metric}_mean"
        if metric_key in threshold_df.columns:
            ranked_df = threshold_df.sort_values(metric_key, ascending=not args.higher_is_better)
            ranked_path = value_root / "prokcat_summary_ranked.csv"
            ranked_df.to_csv(ranked_path, index=False)

        print(f"Saved runs summary: {runs_path}")
        print(f"Saved threshold summary: {threshold_path}")
        print(f"Saved split-group summary: {by_split_path}")
    else:
        print("No completed jobs to summarize.")


if __name__ == "__main__":
    main()
