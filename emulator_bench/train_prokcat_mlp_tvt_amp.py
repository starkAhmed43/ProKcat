import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy import stats
from torch import optim
from tqdm.auto import tqdm

from feature_utils import load_pickle
from original_data_utils import batch2tensor, iterate_batches, load_data, scores_metrics
from original_prokcat_mlp import NewMLPIndependent, ProKcatBackboneForMLP


# AMP variant of the notebook-derived ProKcat MLP trainer.
# Keeps architecture and data flow identical to train_prokcat_mlp_tvt.py,
# but adds mixed precision + gradient accumulation for better GPU utilization.


def evaluate(backbone, mlp_head, data_pack, batch_size, device):
    backbone.eval()
    mlp_head.eval()

    preds, labels = [], []
    for batch in iterate_batches(data_pack, batch_size=batch_size, shuffle=False):
        atoms_pad, atoms_mask, adj_pad, fps, amino_pad, amino_mask, inv_temp, temp, y = batch2tensor(batch, True, device)
        with torch.no_grad():
            cf, fps2, pf, inv_t, t = backbone(atoms_pad, atoms_mask, adj_pad, amino_pad, amino_mask, fps, inv_temp, temp)
            pred, _ = mlp_head(cf, fps2, pf, inv_t, t)

        preds += pred.float().cpu().numpy().reshape(-1).tolist()
        labels += y.float().cpu().numpy().reshape(-1).tolist()

    labels = np.array(labels, dtype=float)
    preds = np.array(preds, dtype=float)
    return full_metrics(labels, preds)


def evaluate_with_preds(backbone, mlp_head, data_pack, batch_size, device):
    backbone.eval()
    mlp_head.eval()

    preds, labels = [], []
    for batch in iterate_batches(data_pack, batch_size=batch_size, shuffle=False):
        atoms_pad, atoms_mask, adj_pad, fps, amino_pad, amino_mask, inv_temp, temp, y = batch2tensor(batch, True, device)
        with torch.no_grad():
            cf, fps2, pf, inv_t, t = backbone(atoms_pad, atoms_mask, adj_pad, amino_pad, amino_mask, fps, inv_temp, temp)
            pred, _ = mlp_head(cf, fps2, pf, inv_t, t)

        preds += pred.float().cpu().numpy().reshape(-1).tolist()
        labels += y.float().cpu().numpy().reshape(-1).tolist()

    return np.array(labels, dtype=float), np.array(preds, dtype=float)


def full_metrics(labels: np.ndarray, preds: np.ndarray):
    labels = np.asarray(labels, dtype=float).reshape(-1)
    preds = np.asarray(preds, dtype=float).reshape(-1)

    valid = np.isfinite(labels) & np.isfinite(preds)
    labels = labels[valid]
    preds = preds[valid]

    if labels.size == 0:
        return {
            "PCC": 0.0,
            "SCC": 0.0,
            "R2": 0.0,
            "RMSE": float("inf"),
            "MSE": float("inf"),
            "MAE": float("inf"),
        }

    rmse, r2, pcc, mae = scores_metrics(labels, preds)
    mse = float(np.mean((labels - preds) ** 2))
    try:
        scc = float(stats.spearmanr(labels, preds).correlation)
        if np.isnan(scc):
            scc = 0.0
    except Exception:
        scc = 0.0
    return {
        "PCC": float(pcc),
        "SCC": float(scc),
        "R2": float(r2),
        "RMSE": float(rmse),
        "MSE": float(mse),
        "MAE": float(mae),
    }


def train(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    amp_enabled = bool(args.use_amp and device.type == "cuda")
    amp_dtype = torch.bfloat16

    if device.type == "cuda":
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    param_dict = load_pickle(args.param_dict_pkl)
    atom_dict = load_pickle(str(Path(args.dict_dir) / "fingerprint_dict.pkl"))
    word_dict = load_pickle(str(Path(args.dict_dir) / "word_dict.pkl"))

    train_data = load_data(args.data_root, True, "train")
    val_data = load_data(args.data_root, True, "val")
    test_data = load_data(args.data_root, True, "test")

    backbone = ProKcatBackboneForMLP(
        len(atom_dict),
        len(word_dict),
        param_dict["comp_dim"],
        param_dict["prot_dim"],
        param_dict["gat_dim"],
        param_dict["num_head"],
        param_dict["dropout"],
        param_dict["alpha"],
        param_dict["window"],
        param_dict["layer_cnn"],
        param_dict["latent_dim"],
        param_dict["layer_out"],
    ).to(device)

    mlp_head = NewMLPIndependent(alpha=param_dict["alpha"], latent_dim=param_dict["latent_dim"]).to(device)

    optimizer = optim.Adam(
        list(backbone.parameters()) + list(mlp_head.parameters()),
        lr=args.lr,
        weight_decay=0,
        amsgrad=True,
    )
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.decay_interval, gamma=args.lr_decay)
    # bf16 mixed precision typically does not require dynamic loss scaling.
    scaler = torch.amp.GradScaler(enabled=False)
    criterion = F.mse_loss

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    best_val_rmse = float("inf")
    history = []
    start = time.time()

    accum_steps = max(1, int(args.grad_accum_steps))
    n_train = len(train_data[0])
    n_batches = (n_train + args.batch_size - 1) // args.batch_size

    epoch_bar = tqdm(range(args.epochs), desc="Training", unit="epoch", ascii=True, dynamic_ncols=True)
    for epoch in epoch_bar:
        backbone.train()
        mlp_head.train()
        optimizer.zero_grad(set_to_none=True)

        preds_train, labels_train = [], []
        batch_iter = iterate_batches(train_data, batch_size=args.batch_size, shuffle=True)
        batch_bar = tqdm(
            batch_iter,
            total=n_batches,
            desc=f"Epoch {epoch + 1}/{args.epochs} batches",
            unit="batch",
            leave=False,
            position=1,
            ascii=True,
            dynamic_ncols=True,
            mininterval=1.0,
            miniters=50,
        )

        for batch_idx, batch in enumerate(batch_bar, start=1):
            atoms_pad, atoms_mask, adj_pad, fps, amino_pad, amino_mask, inv_temp, temp, y = batch2tensor(batch, True, device)

            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled):
                cf, fps2, pf, inv_t, t = backbone(atoms_pad, atoms_mask, adj_pad, amino_pad, amino_mask, fps, inv_temp, temp)
                pred, _ = mlp_head(cf, fps2, pf, inv_t, t)
                loss = criterion(pred.float(), y.float())

            # AMP can occasionally produce non-finite values on unstable batches.
            if not torch.isfinite(loss):
                optimizer.zero_grad(set_to_none=True)
                tqdm.write(f"[warn] Non-finite loss at epoch={epoch + 1}, batch={batch_idx}; batch skipped")
                continue

            if not torch.isfinite(pred).all() or not torch.isfinite(y).all():
                optimizer.zero_grad(set_to_none=True)
                tqdm.write(f"[warn] Non-finite tensors at epoch={epoch + 1}, batch={batch_idx}; batch skipped")
                continue

            loss_scaled = loss / accum_steps
            scaler.scale(loss_scaled).backward()

            should_step = (batch_idx % accum_steps == 0) or (batch_idx == n_batches)
            if should_step:
                if args.max_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        list(backbone.parameters()) + list(mlp_head.parameters()),
                        args.max_grad_norm,
                    )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            preds_train += pred.detach().float().cpu().numpy().reshape(-1).tolist()
            labels_train += y.detach().float().cpu().numpy().reshape(-1).tolist()

            if batch_idx % 100 == 0 or batch_idx == n_batches:
                batch_bar.set_postfix(loss=f"{float(loss.item()):.4f}")

        train_metrics = full_metrics(np.array(labels_train, dtype=float), np.array(preds_train, dtype=float))
        val_metrics = evaluate(backbone, mlp_head, val_data, args.batch_size, device)

        history.append(
            {
                "epoch": epoch + 1,
                "RMSE_train": train_metrics["RMSE"],
                "R2_train": train_metrics["R2"],
                "PCC_train": train_metrics["PCC"],
                "SCC_train": train_metrics["SCC"],
                "MSE_train": train_metrics["MSE"],
                "MAE_train": train_metrics["MAE"],
                "RMSE_val": val_metrics["RMSE"],
                "R2_val": val_metrics["R2"],
                "PCC_val": val_metrics["PCC"],
                "SCC_val": val_metrics["SCC"],
                "MSE_val": val_metrics["MSE"],
                "MAE_val": val_metrics["MAE"],
            }
        )

        if val_metrics["RMSE"] < best_val_rmse:
            best_val_rmse = val_metrics["RMSE"]
            torch.save(backbone.state_dict(), out_dir / "best_backbone.pth")
            torch.save(mlp_head.state_dict(), out_dir / "best_mlp_head.pth")

        epoch_bar.set_postfix(
            train_r2=f"{train_metrics['R2']:.4f}",
            train_mse=f"{train_metrics['MSE']:.4f}",
            val_r2=f"{val_metrics['R2']:.4f}",
            val_mse=f"{val_metrics['MSE']:.4f}",
            amp=str(amp_enabled),
            accum=accum_steps,
        )

        if (epoch + 1) % args.log_every == 0:
            tqdm.write(
                f"Epoch {epoch + 1}/{args.epochs} | "
                f"train_r2={train_metrics['R2']:.4f} train_mse={train_metrics['MSE']:.4f} | "
                f"val_r2={val_metrics['R2']:.4f} val_mse={val_metrics['MSE']:.4f} | "
                f"amp={amp_enabled} accum={accum_steps}"
            )

        scheduler.step()

    hist_df = pd.DataFrame(history)
    hist_df.to_csv(out_dir / "logfile.csv", index=False)

    backbone.load_state_dict(torch.load(out_dir / "best_backbone.pth", map_location=device))
    mlp_head.load_state_dict(torch.load(out_dir / "best_mlp_head.pth", map_location=device))

    val_labels, val_preds = evaluate_with_preds(backbone, mlp_head, val_data, args.batch_size, device)
    test_labels, test_preds = evaluate_with_preds(backbone, mlp_head, test_data, args.batch_size, device)

    val_metrics = full_metrics(val_labels, val_preds)
    test_metrics = full_metrics(test_labels, test_preds)

    pd.DataFrame([val_metrics]).to_csv(out_dir / "final_results_val.csv", index=False)
    pd.DataFrame([test_metrics]).to_csv(out_dir / "final_results_test.csv", index=False)

    pd.DataFrame({"pred": val_preds, "label": val_labels}).to_csv(out_dir / "pred_label_val.csv", index=False)
    pd.DataFrame({"pred": test_preds, "label": test_labels}).to_csv(out_dir / "pred_label_test.csv", index=False)

    with open(out_dir / "time_running.dat", "w", encoding="utf-8") as f:
        f.write(str(time.time() - start))

    print(f"Training complete. Outputs written to: {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ProKcat MLP on explicit TVT splits (AMP + accumulation variant).")
    parser.add_argument("--data_root", required=True, type=str)
    parser.add_argument("--dict_dir", default="data/dict", type=str)
    parser.add_argument("--param_dict_pkl", default="data/hyparams/param_2.pkl", type=str)
    parser.add_argument("--out_dir", required=True, type=str)

    parser.add_argument("--epochs", default=40, type=int)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--lr_decay", default=0.5, type=float)
    parser.add_argument("--decay_interval", default=5, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--device", default="cuda:0", type=str)
    parser.add_argument("--log_every", default=1, type=int)

    parser.add_argument("--use_amp", action="store_true", help="Enable torch autocast + GradScaler on CUDA.")
    parser.add_argument("--grad_accum_steps", default=1, type=int, help="Gradient accumulation steps.")
    parser.add_argument("--max_grad_norm", default=0.0, type=float, help="Gradient clipping; <=0 disables clipping.")

    args = parser.parse_args()
    train(args)
