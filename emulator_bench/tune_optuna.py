import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import subprocess
import sys
from pathlib import Path

import optuna
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

    jobs.sort(key=lambda x: (x[0], _threshold_to_float(x[1])))
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


def maybe_build_split_features(split_path, split_name, feats_root, args, has_dict):
    split_features_dir = feats_root / f"{split_name}_features"
    marker = split_features_dir / "log10_kcat.pkl"
    if marker.exists() and not args.overwrite_features:
        return

    cmd = [
        sys.executable,
        "emulator_bench/build_tvt_features.py",
        "--input_path",
        str(split_path),
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

    if args.target_is_raw:
        cmd.append("--target_is_raw")
    if args.no_cache_read:
        cmd.append("--no_cache_read")
    if args.no_cache_write:
        cmd.append("--no_cache_write")

    subprocess.run(cmd, check=True)


def run_training(data_root, out_dir, args, seed, hp, device):
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

    batch_candidates = [int(hp["batch_size"])] + [b for b in [8, 4, 2, 1] if b < int(hp["batch_size"])]

    for i, batch_size in enumerate(batch_candidates):
        cmd = [
            sys.executable,
            "emulator_bench/train_prokcat_mlp_tvt.py",
            "--data_root",
            str(data_root),
            "--dict_dir",
            str(args.dict_dir),
            "--param_dict_pkl",
            str(args.param_dict_pkl),
            "--out_dir",
            str(out_dir),
            "--epochs",
            str(hp["epochs"]),
            "--batch_size",
            str(batch_size),
            "--lr",
            str(hp["lr"]),
            "--lr_decay",
            str(hp["lr_decay"]),
            "--decay_interval",
            str(hp["decay_interval"]),
            "--seed",
            str(seed),
            "--device",
            device,
        ]

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


def _objective_direction(metric_name: str):
    return "maximize" if metric_name in {"PCC", "SCC", "R2"} else "minimize"


def main():
    parser = argparse.ArgumentParser(description="Optuna tuning for ProKcat notebook-derived MLP TVT workflow.")

    parser.add_argument("--base_dir", default="~/github/EMULaToR/data/processed/baselines/ProKcat", type=str)
    parser.add_argument("--value_type", required=True, choices=["kcat", "km", "ki"], type=str)
    parser.add_argument("--split_groups", nargs="+", default=["enzyme_sequence_splits", "substrate_splits"])
    parser.add_argument(
        "--separate_by_split_group",
        action="store_true",
        help="Run independent Optuna studies for each split group instead of one combined objective.",
    )
    parser.add_argument("--thresholds", nargs="+", default=None)
    parser.add_argument("--max_jobs", type=int, default=10, help="Use only first N discovered threshold jobs for tuning.")

    parser.add_argument("--target_col", default="log10_value", type=str)
    parser.add_argument("--target_is_raw", action="store_true")
    parser.add_argument("--sequence_col", default="sequence", type=str)
    parser.add_argument("--smiles_col", default="smiles", type=str)
    parser.add_argument("--temp_col", default="Temperature", type=str)

    parser.add_argument("--dict_dir", default="data/dict", type=str)
    parser.add_argument("--param_dict_pkl", default="data/hyparams/param_2.pkl", type=str)
    parser.add_argument("--cache_dir", default="emulator_bench/.cache_embeddings", type=str)
    parser.add_argument("--radius", default=2, type=int)
    parser.add_argument("--ngram", default=3, type=int)

    parser.add_argument("--no_cache_read", action="store_true")
    parser.add_argument("--no_cache_write", action="store_true")
    parser.add_argument("--overwrite_features", action="store_true")

    parser.add_argument("--device", default="cuda:0", type=str)
    parser.add_argument("--seeds", nargs="+", type=int, default=[41, 42, 43, 44, 45])

    parser.add_argument("--n_trials", type=int, default=40)
    parser.add_argument("--sampler_seed", type=int, default=42)
    parser.add_argument("--study_name", type=str, default=None)
    parser.add_argument("--storage", type=str, default=None, help="Optional Optuna storage URL")

    parser.add_argument("--metric", type=str, default="MSE", choices=["PCC", "SCC", "R2", "RMSE", "MSE", "MAE"])
    parser.add_argument("--eval_split", type=str, default="val", choices=["val", "test"])
    parser.add_argument("--parallel_runs_per_trial", type=int, default=2)
    parser.add_argument("--trial_parallelism", type=int, default=2)
    parser.add_argument("--devices", nargs="+", default=None)

    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--dry_run", action="store_true")

    args = parser.parse_args()

    if args.parallel_runs_per_trial < 1 or args.trial_parallelism < 1:
        raise ValueError("parallelism values must be >= 1")

    base_dir = Path(args.base_dir).expanduser()
    value_root = base_dir / args.value_type
    if not value_root.exists():
        raise FileNotFoundError(f"Value type directory not found: {value_root}")

    jobs = discover_threshold_dirs(value_root, args.split_groups, args.thresholds)
    if not jobs:
        raise RuntimeError("No threshold jobs discovered")

    if args.max_jobs and args.max_jobs > 0:
        jobs = jobs[: args.max_jobs]

    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<not-set>')}")
    print(
        f"Parallelism: parallel_runs_per_trial={args.parallel_runs_per_trial}, "
        f"trial_parallelism={args.trial_parallelism}"
    )
    if args.devices:
        print(f"Device pool: {args.devices}")
    print(f"Tuning jobs ({len(jobs)}):")
    for split_group, threshold_name, threshold_dir in jobs:
        print(f"- {split_group}/{threshold_name}: {threshold_dir}")

    if args.dry_run:
        return

    # Build/reuse features once.
    prepared_jobs = []
    for split_group, threshold_name, threshold_dir in tqdm(jobs, desc="Preparing features", unit="job"):
        train_path, val_path, test_path = ensure_triplet(threshold_dir)
        if not (train_path.exists() and val_path.exists() and test_path.exists()):
            continue

        feats_root = threshold_dir / "prokcat_features"
        feats_root.mkdir(parents=True, exist_ok=True)

        maybe_build_split_features(train_path, "train", feats_root, args, has_dict=False)
        maybe_build_split_features(val_path, "val", feats_root, args, has_dict=True)
        maybe_build_split_features(test_path, "test", feats_root, args, has_dict=True)

        prepared_jobs.append((split_group, threshold_name, threshold_dir, feats_root))

    if not prepared_jobs:
        raise RuntimeError("No valid jobs with train/val/test triplets were found.")

    base_study_name = args.study_name or f"prokcat_{args.value_type}_{args.metric.lower()}"
    direction = _objective_direction(args.metric)
    artifacts_dir = value_root / "optuna_studies"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    def run_one_study(study_name, jobs_subset, split_groups_for_metadata):
        sampler = optuna.samplers.TPESampler(seed=args.sampler_seed)
        if args.storage:
            study = optuna.create_study(
                study_name=study_name,
                storage=args.storage,
                load_if_exists=True,
                direction=direction,
                sampler=sampler,
            )
        else:
            print("Optuna storage: in-memory (no SQLite DB)")
            study = optuna.create_study(study_name=study_name, direction=direction, sampler=sampler)

        run_root = value_root / "prokcat_optuna_runs" / study_name
        run_root.mkdir(parents=True, exist_ok=True)

        def _assigned_device(task_idx):
            if args.devices:
                return args.devices[task_idx % len(args.devices)]
            return args.device

        def _run_single(task_idx, split_group, threshold_name, feats_root, trial_number, seed, hp):
            out_dir = run_root / f"trial_{trial_number}" / split_group / threshold_name / f"seed_{seed}"
            metric_csv = out_dir / f"final_results_{args.eval_split}.csv"

            if not metric_csv.exists():
                run_training(
                    data_root=feats_root,
                    out_dir=out_dir,
                    args=args,
                    seed=seed,
                    hp=hp,
                    device=_assigned_device(task_idx),
                )

            if not metric_csv.exists():
                raise RuntimeError(f"Missing metrics file for trial {trial_number}: {metric_csv}")

            df = pd.read_csv(metric_csv)
            if args.metric not in df.columns:
                raise RuntimeError(f"Metric {args.metric} not found in {metric_csv}")
            return float(df.iloc[0][args.metric])

        def objective(trial: optuna.Trial):
            hp = {
                "batch_size": trial.suggest_categorical("batch_size", [8, 16, 32, 64]),
                "lr": trial.suggest_float("lr", 1e-5, 5e-3, log=True),
                "lr_decay": trial.suggest_float("lr_decay", 0.2, 0.9),
                "decay_interval": trial.suggest_int("decay_interval", 2, 15),
                "epochs": args.epochs,
            }

            tasks = []
            task_idx = 0
            for split_group, threshold_name, _, feats_root in jobs_subset:
                for seed in args.seeds:
                    tasks.append((task_idx, split_group, threshold_name, feats_root, trial.number, seed, hp))
                    task_idx += 1

            metric_values = []
            if args.parallel_runs_per_trial == 1:
                for task in tasks:
                    try:
                        metric_values.append(_run_single(*task))
                    except subprocess.CalledProcessError as e:
                        raise optuna.TrialPruned(f"Training failed for trial {trial.number}: {e}")
                    except Exception as e:
                        raise optuna.TrialPruned(str(e))
            else:
                with ThreadPoolExecutor(max_workers=args.parallel_runs_per_trial) as ex:
                    futures = [ex.submit(_run_single, *task) for task in tasks]
                    for f in as_completed(futures):
                        try:
                            metric_values.append(f.result())
                        except subprocess.CalledProcessError as e:
                            raise optuna.TrialPruned(f"Training failed for trial {trial.number}: {e}")
                        except Exception as e:
                            raise optuna.TrialPruned(str(e))

            if not metric_values:
                raise optuna.TrialPruned("No metric values collected")

            mean_metric = float(sum(metric_values) / len(metric_values))
            trial.set_user_attr("n_runs", len(metric_values))
            trial.set_user_attr("metric", args.metric)
            trial.set_user_attr("eval_split", args.eval_split)
            trial.set_user_attr("mean_metric", mean_metric)
            return mean_metric

        study.optimize(objective, n_trials=args.n_trials, n_jobs=args.trial_parallelism)

        best_hp = dict(study.best_params)
        best_hp["epochs"] = args.epochs

        best_path = artifacts_dir / f"{study_name}_best_hparams.json"
        with open(best_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "base_dir": str(base_dir),
                    "value_type": args.value_type,
                    "metric": args.metric,
                    "direction": direction,
                    "eval_split": args.eval_split,
                    "seeds": args.seeds,
                    "split_groups": split_groups_for_metadata,
                    "thresholds": args.thresholds,
                    "max_jobs": args.max_jobs,
                    "best_trial_number": study.best_trial.number,
                    "best_value": float(study.best_value),
                    "batch_size": int(best_hp["batch_size"]),
                    "train_batch_size": int(best_hp["batch_size"]),
                    "lr": float(best_hp["lr"]),
                    "lr_decay": float(best_hp["lr_decay"]),
                    "decay_interval": int(best_hp["decay_interval"]),
                    "epochs": int(best_hp["epochs"]),
                },
                f,
                indent=2,
            )

        trials_path = artifacts_dir / f"{study_name}_trials.csv"
        study.trials_dataframe().to_csv(trials_path, index=False)

        print(f"[{study_name}] Best trial: {study.best_trial.number}")
        print(f"[{study_name}] Best value ({args.metric}): {study.best_value}")
        print(f"[{study_name}] Best params saved to: {best_path}")
        print(f"[{study_name}] Trial table saved to: {trials_path}")

    if args.separate_by_split_group:
        grouped = {}
        for row in prepared_jobs:
            grouped.setdefault(row[0], []).append(row)

        for split_group, jobs_subset in grouped.items():
            sub_name = f"{base_study_name}__{split_group}"
            print(f"Running separate study for split_group={split_group}: {sub_name}")
            run_one_study(sub_name, jobs_subset, [split_group])
    else:
        run_one_study(base_study_name, prepared_jobs, args.split_groups)


if __name__ == "__main__":
    main()
