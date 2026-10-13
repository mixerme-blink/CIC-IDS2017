"""Construit le graphe KNN IoT avec le contrat commun de huit features."""

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
from torch_geometric.data import Data

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import (
    HARMONIZED_FEATURES,
    IOT_GRAPH_PATH,
    IOT_RAW_DATA_DIR,
    IOT_SCALER_PATH,
    N_FEATURES,
    PROCESSED_DATA_DIR,
    RANDOM_STATE,
)


IOT_SOURCE_COLUMNS = {
    "packet_count": "Number",
    "total_bytes": "Tot sum",
    "packet_len_mean": "AVG",
    "packet_len_std": "Std",
    "packet_len_var": "Variance",
    "packet_len_min": "Min",
    "packet_len_max": "Max",
    "flow_rate": "Rate",
}
BENIGN_FOLDERS = {"benign", "benign_final"}


def sample_block(frame, size, rng):
    if len(frame) <= size:
        return frame
    return frame.iloc[np.sort(rng.choice(len(frame), size=size, replace=False))]


def load_split_blocks(raw_dir, rows_per_file):
    """Échantillonne chaque fichier par blocs 70/15/15 sans mélange temporel."""
    files = sorted(raw_dir.rglob("*.csv"))
    if not files:
        raise FileNotFoundError(f"Aucun CSV IoT trouvé dans : {raw_dir}")

    rng = np.random.default_rng(RANDOM_STATE)
    columns = list(IOT_SOURCE_COLUMNS.values())
    sizes = {
        "train": int(rows_per_file * 0.70),
        "val": int(rows_per_file * 0.15),
    }
    sizes["test"] = rows_per_file - sizes["train"] - sizes["val"]
    parts = {"train": [], "val": [], "test": []}

    for file_path in files:
        category = file_path.parent.name
        label = int(category.strip().lower() not in BENIGN_FOLDERS)
        frame = pd.read_csv(file_path, usecols=lambda name: name in columns)
        if len(frame) < 10:
            continue

        first = int(len(frame) * 0.70)
        second = int(len(frame) * 0.85)
        blocks = {
            "train": frame.iloc[:first],
            "val": frame.iloc[first:second],
            "test": frame.iloc[second:],
        }
        for split, block in blocks.items():
            selected = sample_block(block, sizes[split], rng).copy()
            selected["label_binary"] = label
            selected["category"] = category
            parts[split].append(selected)

    if not all(parts.values()):
        raise ValueError("Les trois splits IoT doivent contenir au moins un flow.")
    return {split: pd.concat(values, ignore_index=True) for split, values in parts.items()}


def clean(frame):
    frame = frame.copy()
    source_columns = list(IOT_SOURCE_COLUMNS.values())
    for column in source_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame[source_columns] = frame[source_columns].replace([np.inf, -np.inf], np.nan)
    return frame.dropna(subset=source_columns).reset_index(drop=True)


def build_graph(splits, neighbors):
    splits = {name: clean(frame) for name, frame in splits.items()}
    source_columns = [IOT_SOURCE_COLUMNS[name] for name in HARMONIZED_FEATURES]
    scaler = MinMaxScaler()
    train_x = splits["train"][source_columns].to_numpy(dtype=np.float32)
    scaler.fit(train_x)

    scaled = {
        name: scaler.transform(frame[source_columns].to_numpy(dtype=np.float32))
        for name, frame in splits.items()
    }
    frame = pd.concat(
        [splits["train"].assign(split="train"), splits["val"].assign(split="val"), splits["test"].assign(split="test")],
        ignore_index=True,
    )
    x = np.concatenate([scaled["train"], scaled["val"], scaled["test"]], axis=0)
    if x.shape[1] != N_FEATURES or not np.isfinite(x).all():
        raise ValueError("Les features IoT doivent être finies et respecter le contrat à 8 colonnes.")

    k = min(neighbors + 1, len(x))
    indices = NearestNeighbors(n_neighbors=k, n_jobs=-1).fit(x).kneighbors(x, return_distance=False)
    source = np.repeat(np.arange(len(x)), k - 1)
    target = indices[:, 1:].reshape(-1)
    edge_index = torch.tensor(
        np.column_stack([np.concatenate([source, target]), np.concatenate([target, source])]).T,
        dtype=torch.long,
    )

    data = Data(
        x=torch.tensor(x, dtype=torch.float32),
        edge_index=edge_index,
        y=torch.tensor(frame["label_binary"].to_numpy(), dtype=torch.long),
    )
    for split, attribute in (("train", "train_mask"), ("val", "val_mask"), ("test", "test_mask")):
        setattr(data, attribute, torch.tensor((frame["split"] == split).to_numpy(), dtype=torch.bool))
    data.feature_names = list(HARMONIZED_FEATURES)
    data.graph_type = "knn_on_harmonized_features"
    data.domain = "iot"
    data.scaler_fit_split = "train"
    return data, scaler, frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=IOT_RAW_DATA_DIR)
    parser.add_argument("--output", type=Path, default=IOT_GRAPH_PATH)
    parser.add_argument("--scaler-output", type=Path, default=IOT_SCALER_PATH)
    parser.add_argument("--rows-per-file", type=int, default=540)
    parser.add_argument("--neighbors", type=int, default=5)
    args = parser.parse_args()

    splits = load_split_blocks(args.raw_dir, args.rows_per_file)
    data, scaler, metadata = build_graph(splits, args.neighbors)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, args.output)
    joblib.dump(scaler, args.scaler_output)
    metadata.to_parquet(PROCESSED_DATA_DIR / "iot_flow_metadata.parquet", index=False)
    with open(PROCESSED_DATA_DIR / "iot_feature_mapping.json", "w", encoding="utf-8") as file:
        json.dump({"feature_order": HARMONIZED_FEATURES, "source_columns": IOT_SOURCE_COLUMNS}, file, indent=2)

    print(f"Graphe IoT : {data.num_nodes:,} nœuds, {data.num_edges:,} arêtes")
    print(f"data.x : {tuple(data.x.shape)} | features : {data.feature_names}")
    print(f"Masques train/val/test : {int(data.train_mask.sum())}/{int(data.val_mask.sum())}/{int(data.test_mask.sum())}")
    print(f"Graphe : {args.output}\nScaler Min-Max train : {args.scaler_output}")


if __name__ == "__main__":
    main()
