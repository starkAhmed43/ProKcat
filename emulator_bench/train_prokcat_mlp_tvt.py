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


# NOTE: This trainer follows the authors' notebook MLP path in code/run_train_test.ipynb:
# - backbone returns (cf, fps, pf, inverse_Temp, Temperature)
# - head is New_MLP_Independent
# KAN path is intentionally excluded per request.


def evaluate(backbone, mlp_head, data_pack, batch_size, device):
    backbone.eval()
    mlp_head.eval()

    preds, labels = [], []
    for batch in iterate_batches(data_pack, batch_size=batch_size, shuffle=False):
        atoms_pad, atoms_mask, adj_pad, fps, amino_pad, amino_mask, inv_temp, temp, y = batch2tensor(batch, True, device)
        with torch.no_grad():
            cf, fps2, pf, inv_t, t = backbone(atoms_pad, atoms_mask, adj_pad, amino_pad, amino_mask, fps, inv_temp, temp)
            pred, _ = mlp_head(cf, fps2, pf, inv_t, t)

        preds += pred.cpu().numpy().reshape(-1).tolist()
        labels += y.cpu().numpy().reshape(-1).tolist()

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

        preds += pred.cpu().numpy().reshape(-1).tolist()
        labels += y.cpu().numpy().reshape(-1).tolist()

    return np.array(labels, dtype=float), np.array(preds, dtype=float)


def full_metrics(labels: np.ndarray, preds: np.ndarray):
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

    # Notebook used optimizer on backbone only; here we optimize both backbone + MLP head.
    optimizer = optim.Adam(
        list(backbone.parameters()) + list(mlp_head.parameters()),
        lr=args.lr,
        weight_decay=0,
        amsgrad=True,
    )
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.decay_interval, gamma=args.lr_decay)
    criterion = F.mse_loss

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    best_val_rmse = float("inf")
    history = []
    start = time.time()

    epoch_bar = tqdm(range(args.epochs), desc="Training", unit="epoch", ascii=True, dynamic_ncols=True)
    n_train = len(train_data[0])
    n_batches = (n_train + args.batch_size - 1) // args.batch_size
    for epoch in epoch_bar:
        backbone.train()
        mlp_head.train()

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

            cf, fps2, pf, inv_t, t = backbone(atoms_pad, atoms_mask, adj_pad, amino_pad, amino_mask, fps, inv_temp, temp)
            pred, _ = mlp_head(cf, fps2, pf, inv_t, t)
            loss = criterion(pred.float(), y.float())

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if batch_idx % 100 == 0 or batch_idx == n_batches:
                batch_bar.set_postfix(loss=f"{float(loss.item()):.4f}")

            preds_train += pred.detach().cpu().numpy().reshape(-1).tolist()
            labels_train += y.detach().cpu().numpy().reshape(-1).tolist()

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
        )

        if (epoch + 1) % args.log_every == 0:
            tqdm.write(
                f"Epoch {epoch + 1}/{args.epochs} | "
                f"train_r2={train_metrics['R2']:.4f} train_mse={train_metrics['MSE']:.4f} | "
                f"val_r2={val_metrics['R2']:.4f} val_mse={val_metrics['MSE']:.4f}"
            )

        scheduler.step()

    hist_df = pd.DataFrame(history)
    hist_df.to_csv(out_dir / "logfile.csv", index=False)

    # Evaluate best saved checkpoints and export CataPro-like metric artifacts.
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
    parser = argparse.ArgumentParser(description="Train authors' ProKcat notebook MLP path on explicit TVT splits.")
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

    args = parser.parse_args()
    train(args)
