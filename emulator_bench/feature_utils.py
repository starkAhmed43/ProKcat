import gc
import hashlib
import os
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs
from rdkit.Chem import rdFingerprintGenerator


CACHE_VERSION = "v1"


_MORGAN_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024, includeChirality=True)


def stable_dict_signature(*objs):
    h = hashlib.sha256()
    h.update(CACHE_VERSION.encode("utf-8"))
    for obj in objs:
        # Hash the exact mapping state so cache keys track dictionary evolution.
        h.update(pickle.dumps(obj, protocol=4))
    return h.hexdigest()[:16]


def load_table(path):
    p = Path(path)
    if p.suffix.lower() == ".csv":
        return pd.read_csv(p)
    if p.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(p)
    raise ValueError(f"Unsupported file type: {p.suffix}")


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def dump_pickle(obj, path):
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump(obj, f)


def _ensure_cache_root(cache_dir):
    if cache_dir is None:
        return None
    root = Path(cache_dir)
    root.mkdir(parents=True, exist_ok=True)
    return root


def _cache_key(namespace, value):
    text = f"{CACHE_VERSION}|{namespace}|{value}"
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _cache_file(cache_root, namespace, key):
    subdir = cache_root / namespace
    subdir.mkdir(parents=True, exist_ok=True)
    return subdir / f"{key}.npy"


def _load_cache_vec(cache_root, namespace, key):
    fpath = _cache_file(cache_root, namespace, key)
    if not fpath.exists():
        return None
    try:
        return np.load(fpath, allow_pickle=False)
    except Exception:
        return None


def _save_cache_vec(cache_root, namespace, key, vec):
    fpath = _cache_file(cache_root, namespace, key)
    tmp = fpath.with_suffix(f".tmp.{os.getpid()}.npy")
    np.save(tmp, np.asarray(vec))
    os.replace(tmp, fpath)


def check_dict(item, dict2check):
    if item in dict2check.keys():
        return dict2check[item]
    if len(dict2check.keys()) == 0:
        dict2check[item] = 0
    else:
        dict2check[item] = max(list(dict2check.values())) + 1
    return dict2check[item]


def create_atoms(mol, atom_dict):
    atoms = [a.GetSymbol() for a in mol.GetAtoms()]
    for a in mol.GetAromaticAtoms():
        i = a.GetIdx()
        atoms[i] = (atoms[i], "aromatic")
    atoms = [check_dict(a, atom_dict) for a in atoms]
    return np.array(atoms)


def create_ijbonddict(mol, bond_dict):
    i_jbond_dict = defaultdict(lambda: [])
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bond = check_dict(str(b.GetBondType()), bond_dict)
        i_jbond_dict[i].append((j, bond))
        i_jbond_dict[j].append((i, bond))

    atoms_set = set(range(mol.GetNumAtoms()))
    isolate_atoms = atoms_set - set(i_jbond_dict.keys())
    bond = check_dict("nan", bond_dict)
    for a in isolate_atoms:
        i_jbond_dict[a].append((a, bond))

    return i_jbond_dict


def atom_features(atoms, i_jbond_dict, radius, fingerprint_dict, edge_dict):
    if (len(atoms) == 1) or (radius == 0):
        fingerprints = [check_dict(a, fingerprint_dict) for a in atoms]
    else:
        nodes = atoms
        i_jedge_dict = i_jbond_dict
        for _ in range(radius):
            fingerprints = []
            for i, j_edge in i_jedge_dict.items():
                neighbors = [(nodes[j], edge) for j, edge in j_edge]
                fingerprint = (nodes[i], tuple(sorted(neighbors)))
                fingerprints.append(check_dict(fingerprint, fingerprint_dict))

            nodes = fingerprints
            _i_jedge_dict = defaultdict(lambda: [])
            for i, j_edge in i_jedge_dict.items():
                for j, edge in j_edge:
                    both_side = tuple(sorted((nodes[i], nodes[j])))
                    edge = check_dict((both_side, edge), edge_dict)
                    _i_jedge_dict[i].append((j, edge))
            i_jedge_dict = _i_jedge_dict

    return np.array(fingerprints)


def create_adjacency(mol):
    adjacency = Chem.GetAdjacencyMatrix(mol)
    adjacency = np.array(adjacency)
    adjacency += np.eye(adjacency.shape[0], dtype=int)
    return adjacency


def get_fingerprint_1024(mol, radius):
    arr = np.zeros((0,), dtype=np.int8)
    if radius == 2:
        fp = _MORGAN_GENERATOR.GetFingerprint(mol)
    else:
        fp = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=1024, includeChirality=True).GetFingerprint(mol)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def split_sequence(sequence, ngram, word_dict):
    sequence = ">" + sequence + "<"
    words = [check_dict(sequence[i : i + ngram], word_dict) for i in range(len(sequence) - ngram + 1)]
    return np.array(words)


def build_or_load_dicts(dict_dir, has_dict):
    d = Path(dict_dir)
    d.mkdir(parents=True, exist_ok=True)
    if has_dict:
        atom_dict = load_pickle(d / "atom_dict.pkl")
        bond_dict = load_pickle(d / "bond_dict.pkl")
        fingerprint_dict = load_pickle(d / "fingerprint_dict.pkl")
        edge_dict = load_pickle(d / "edge_dict.pkl")
        word_dict = load_pickle(d / "word_dict.pkl")
    else:
        atom_dict, bond_dict, fingerprint_dict, edge_dict, word_dict = {}, {"nan": 0}, {}, {}, {}
    return atom_dict, bond_dict, fingerprint_dict, edge_dict, word_dict


def save_dicts(dict_dir, atom_dict, bond_dict, fingerprint_dict, edge_dict, word_dict):
    d = Path(dict_dir)
    d.mkdir(parents=True, exist_ok=True)
    dump_pickle(atom_dict, d / "atom_dict.pkl")
    dump_pickle(bond_dict, d / "bond_dict.pkl")
    dump_pickle(fingerprint_dict, d / "fingerprint_dict.pkl")
    dump_pickle(edge_dict, d / "edge_dict.pkl")
    dump_pickle(word_dict, d / "word_dict.pkl")


def ensure_temp_features(df, temp_col="Temperature"):
    out = df.copy()

    if "Temp_K_norm" in out.columns and "Inv_Temp_norm" in out.columns:
        out["Temp_K_norm"] = pd.to_numeric(out["Temp_K_norm"], errors="coerce")
        out["Inv_Temp_norm"] = pd.to_numeric(out["Inv_Temp_norm"], errors="coerce")
        return out

    if "Temp_K" not in out.columns:
        if temp_col not in out.columns:
            raise ValueError("Need Temp_K_norm/Inv_Temp_norm or a valid temperature column (e.g. Temperature/Temp/Temp_K)")

        temp_raw = pd.to_numeric(out[temp_col], errors="coerce")
        temp_name = str(temp_col).strip().lower()

        # Prefer explicit Kelvin-like names; otherwise use a simple value-based fallback.
        is_kelvin_named = temp_name in {"temperature", "temp_k", "temperature_k", "kelvin"} or temp_name.endswith("_k")
        median_val = temp_raw.dropna().median()
        is_kelvin_by_value = pd.notna(median_val) and float(median_val) > 170.0

        if is_kelvin_named or is_kelvin_by_value:
            out["Temp_K"] = temp_raw
        else:
            out["Temp_K"] = temp_raw + 273.15

    out["Temp_K"] = pd.to_numeric(out["Temp_K"], errors="coerce")
    out["Inv_Temp"] = 1.0 / out["Temp_K"]

    out["Temp_K_norm"] = (out["Temp_K"] - 273.15) / 100.0
    inv_min = 1.0 / 373.15
    inv_max = 1.0 / 273.15
    out["Inv_Temp_norm"] = (out["Inv_Temp"] - inv_min) / (inv_max - inv_min)

    return out


def smiles_to_cached_graph(
    smiles,
    radius,
    atom_dict,
    bond_dict,
    fingerprint_dict,
    edge_dict,
    cache_tag=None,
    cache_dir=None,
    cache_read=True,
    cache_write=True,
):
    cache_root = _ensure_cache_root(cache_dir)

    compound = adjacency = fp1024 = None
    if cache_root is not None and cache_read:
        ns_suffix = f"_tag{cache_tag}" if cache_tag else ""
        key = _cache_key(f"prokcat_smiles_r{radius}{ns_suffix}", smiles)
        compound = _load_cache_vec(cache_root, "compound", key)
        adjacency = _load_cache_vec(cache_root, "adjacency", key)
        fp1024 = _load_cache_vec(cache_root, "fp1024", key)

    if compound is None or adjacency is None or fp1024 is None:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")
        mol = Chem.AddHs(mol)

        atoms = create_atoms(mol, atom_dict)
        i_jbond_dict = create_ijbonddict(mol, bond_dict)
        compound = atom_features(atoms, i_jbond_dict, radius, fingerprint_dict, edge_dict)
        adjacency = create_adjacency(mol)
        fp1024 = get_fingerprint_1024(mol, radius)

        if cache_root is not None and cache_write:
            ns_suffix = f"_tag{cache_tag}" if cache_tag else ""
            key = _cache_key(f"prokcat_smiles_r{radius}{ns_suffix}", smiles)
            _save_cache_vec(cache_root, "compound", key, compound)
            _save_cache_vec(cache_root, "adjacency", key, adjacency)
            _save_cache_vec(cache_root, "fp1024", key, fp1024)

    return compound, adjacency, fp1024


def sequence_to_cached_ids(seq, ngram, word_dict, cache_tag=None, cache_dir=None, cache_read=True, cache_write=True):
    cache_root = _ensure_cache_root(cache_dir)

    seq_ids = None
    if cache_root is not None and cache_read:
        ns_suffix = f"_tag{cache_tag}" if cache_tag else ""
        key = _cache_key(f"prokcat_seq_ngram{ngram}{ns_suffix}", seq)
        seq_ids = _load_cache_vec(cache_root, "seq_ids", key)

    if seq_ids is None:
        seq_ids = split_sequence(seq, ngram, word_dict)
        if cache_root is not None and cache_write:
            ns_suffix = f"_tag{cache_tag}" if cache_tag else ""
            key = _cache_key(f"prokcat_seq_ngram{ngram}{ns_suffix}", seq)
            _save_cache_vec(cache_root, "seq_ids", key, seq_ids)

    return seq_ids
