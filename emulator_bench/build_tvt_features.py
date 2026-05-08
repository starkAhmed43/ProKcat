import argparse
import json
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from feature_utils import (
    build_or_load_dicts,
    dump_pickle,
    ensure_temp_features,
    load_table,
    save_dicts,
    sequence_to_cached_ids,
    smiles_to_cached_graph,
    stable_dict_signature,
)


def _require_columns(df, cols):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def build_features(
    input_path,
    output_root,
    split_name,
    dict_dir,
    has_dict,
    target_col,
    target_is_log10=True,
    smiles_col="smiles",
    sequence_col="sequence",
    temp_col="Temperature",
    radius=2,
    ngram=3,
    cache_dir="emulator_bench/.cache_embeddings",
    cache_read=True,
    cache_write=True,
):
    df = load_table(input_path)
    _require_columns(df, [smiles_col, sequence_col, target_col])
    df = ensure_temp_features(df, temp_col=temp_col)
    _require_columns(df, ["Temp_K_norm", "Inv_Temp_norm"])

    atom_dict, bond_dict, fingerprint_dict, edge_dict, word_dict = build_or_load_dicts(dict_dir, has_dict)
    graph_cache_tag = stable_dict_signature(atom_dict, bond_dict, fingerprint_dict, edge_dict)
    seq_cache_tag = stable_dict_signature(word_dict)

    compounds, adjacencies, fps, proteins = [], [], [], []
    inv_temp, temp, target_log10_values = [], [], []

    smiles = df[smiles_col].astype(str).tolist()
    seqs = df[sequence_col].astype(str).tolist()
    targets = df[target_col].astype(float).tolist()

    row_iter = zip(smiles, seqs, targets)
    for i, (smi, seq, y) in enumerate(tqdm(row_iter, total=len(df), desc=f"Featurizing {split_name}", unit="row"), start=1):
        if (not target_is_log10) and y <= 0:
            raise ValueError(f"Target must be > 0 for log10 transform. Row {i-1} has {y}")

        compound, adjacency, fp1024 = smiles_to_cached_graph(
            smi,
            radius,
            atom_dict,
            bond_dict,
            fingerprint_dict,
            edge_dict,
            cache_tag=graph_cache_tag,
            cache_dir=cache_dir,
            cache_read=cache_read,
            cache_write=cache_write,
        )
        seq_ids = sequence_to_cached_ids(
            seq,
            ngram,
            word_dict,
            cache_tag=seq_cache_tag,
            cache_dir=cache_dir,
            cache_read=cache_read,
            cache_write=cache_write,
        )

        compounds.append(compound)
        adjacencies.append(adjacency)
        fps.append(fp1024)
        proteins.append(seq_ids)

        inv_temp.append(np.array([float(df.iloc[i - 1]["Inv_Temp_norm"])]))
        temp.append(np.array([float(df.iloc[i - 1]["Temp_K_norm"])]))
        y_log10 = y if target_is_log10 else np.log10(y)
        target_log10_values.append(np.array([float(y_log10)]))

    split_dir = Path(output_root) / f"{split_name}_features"
    split_dir.mkdir(parents=True, exist_ok=True)

    dump_pickle(compounds, split_dir / "compounds.pkl")
    dump_pickle(adjacencies, split_dir / "adjacencies.pkl")
    dump_pickle(fps, split_dir / "fps.pkl")
    dump_pickle(proteins, split_dir / "proteins.pkl")
    dump_pickle(inv_temp, split_dir / "inv_Temp.pkl")
    dump_pickle(temp, split_dir / "Temp.pkl")
    # NOTE: Filename retained for compatibility with existing ProKcat loaders.
    dump_pickle(target_log10_values, split_dir / "log10_kcat.pkl")

    # Metadata makes split provenance explicit for kcat/km/ki and transformed/raw targets.
    meta = {
        "input_path": str(input_path),
        "split_name": split_name,
        "target_col": target_col,
        "target_is_log10": bool(target_is_log10),
        "stored_label_file": "log10_kcat.pkl",
        "stored_label_semantics": "log10(target)",
        "sequence_col": sequence_col,
        "smiles_col": smiles_col,
        "temp_col": temp_col,
        "radius": int(radius),
        "ngram": int(ngram),
        "rows": int(len(df)),
    }
    with open(split_dir / "feature_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    save_dicts(dict_dir, atom_dict, bond_dict, fingerprint_dict, edge_dict, word_dict)

    print(f"Saved features at: {split_dir}")
    print(f"Rows: {len(df)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build ProKcat TVT split features with persistent cache.")
    parser.add_argument("--input_path", required=True, type=str, help="Input CSV/parquet path")
    parser.add_argument("--output_root", required=True, type=str, help="Output root (contains <split>_features)")
    parser.add_argument("--split_name", required=True, type=str, help="train|val|test")

    parser.add_argument("--dict_dir", default="data/dict", type=str)
    parser.add_argument("--has_dict", choices=["True", "False"], default="True")

    parser.add_argument("--target_col", default="log10_value", type=str)
    parser.add_argument(
        "--target_is_log10",
        action="store_true",
        help="Set when --target_col is already in log10-space (for example log10_value).",
    )
    parser.add_argument(
        "--target_is_raw",
        action="store_true",
        help="Set when --target_col is raw (not log10) and should be transformed with log10.",
    )
    parser.add_argument("--sequence_col", default="sequence", type=str)
    parser.add_argument("--smiles_col", default="smiles", type=str)
    parser.add_argument("--temp_col", default="Temperature", type=str)

    parser.add_argument("--radius", default=2, type=int)
    parser.add_argument("--ngram", default=3, type=int)

    parser.add_argument("--cache_dir", default="emulator_bench/.cache_embeddings", type=str)
    parser.add_argument("--no_cache_read", action="store_true")
    parser.add_argument("--no_cache_write", action="store_true")

    args = parser.parse_args()

    # Default behavior is convenient for EMULaToR tables: target_col=log10_value.
    target_is_log10 = True
    if args.target_is_raw:
        target_is_log10 = False
    if args.target_is_log10:
        target_is_log10 = True

    build_features(
        input_path=args.input_path,
        output_root=args.output_root,
        split_name=args.split_name,
        dict_dir=args.dict_dir,
        has_dict=args.has_dict == "True",
        target_col=args.target_col,
        target_is_log10=target_is_log10,
        smiles_col=args.smiles_col,
        sequence_col=args.sequence_col,
        temp_col=args.temp_col,
        radius=args.radius,
        ngram=args.ngram,
        cache_dir=args.cache_dir,
        cache_read=not args.no_cache_read,
        cache_write=not args.no_cache_write,
    )
