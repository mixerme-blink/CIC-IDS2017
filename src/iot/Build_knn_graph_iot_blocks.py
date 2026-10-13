"""
Construction du graphe k-NN IoT — split par BLOCS CONTIGUS dans chaque fichier.
PFA - Hybrid Feature Selection and GNN for Intrusion Detection

Pourquoi : CICIoT2023 n'a pas de colonne Timestamp. Le split par fichier
entier (version précédente) laissait 15 catégories d'attaque (un seul
fichier chacune) uniquement en train, et les catégories à 2 fichiers sans
train : le test devenait artificiellement facile.

Ici, CHAQUE fichier est coupé dans l'ordre de ses lignes :
    premiers 70% -> train | 15% suivants -> validation | derniers 15% -> test
Toutes les catégories sont donc représentées dans les 3 ensembles, et on
évite le mélange aléatoire de flows voisins dans le temps.

HYPOTHÈSE À DOCUMENTER : l'ordre des lignes des CSV suit l'ordre de capture
(non garanti explicitement par la documentation du dataset).

Utilisation :
    python build_knn_graph_iot_blocks.py
"""

import os
import glob
import json
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import RobustScaler
import torch
from torch_geometric.data import Data

# ============================================================
# CONFIGURATION
# ============================================================
RAW_CSV_DIR = r"D:\CICIoT2023\CSV"
OUTPUT_FILE = r"D:\CICIoT2023\graph_iot_knn_blocks.pt"
CATEGORIES_FILE = r"D:\CICIoT2023\graph_iot_knn_blocks_categories.json"

ROWS_PER_FILE = 540        # budget de lignes par fichier (réparti 70/15/15)
TRAIN_FRAC, VAL_FRAC = 0.70, 0.15   # le reste = test
K_NEIGHBORS = 5
RANDOM_STATE = 42
BENIGN_FOLDER_NAMES = {"benign_final", "benign"}

EDGE_FEATURES_SRC = [
    "Protocol Type", "Number", "Tot sum", "AVG", "Std", "Variance",
    "Min", "Max", "Rate", "IAT",
    "syn_flag_number", "ack_flag_number", "fin_flag_number",
    "rst_flag_number", "psh_flag_number",
]
EDGE_LOG1P_COLUMNS = [c for c in EDGE_FEATURES_SRC if c != "Protocol Type"]
NODE_FEATURES_SRC = [
    "Number", "Tot sum", "AVG", "Std", "Variance", "Min", "Max", "Rate",
]


# ============================================================
# ÉTAPE 1 — Split par blocs contigus, fichier par fichier
# ============================================================
def sample_block(block_df, k, rng):
    if len(block_df) <= k:
        return block_df
    idx = np.sort(rng.choice(len(block_df), size=k, replace=False))
    return block_df.iloc[idx]


def load_blocks(raw_csv_dir, needed_cols, rows_per_file, random_state):
    rng = np.random.default_rng(random_state)
    files = sorted(glob.glob(os.path.join(raw_csv_dir, "**", "*.csv"), recursive=True))
    print(f"[INFO] {len(files)} fichiers trouvés")

    parts = {"train": [], "val": [], "test": []}
    k_train = int(rows_per_file * TRAIN_FRAC)
    k_val = int(rows_per_file * VAL_FRAC)
    k_test = rows_per_file - k_train - k_val

    for i, f in enumerate(files, 1):
        category = os.path.basename(os.path.dirname(f))
        label = 0 if category.strip().lower() in BENIGN_FOLDER_NAMES else 1
        df = pd.read_csv(f, usecols=lambda c: c in needed_cols)
        n = len(df)
        if n < 10:
            continue

        i_train = int(n * TRAIN_FRAC)
        i_val = int(n * (TRAIN_FRAC + VAL_FRAC))
        blocks = {
            "train": (df.iloc[:i_train], k_train),
            "val": (df.iloc[i_train:i_val], k_val),
            "test": (df.iloc[i_val:], k_test),
        }
        for split, (block, k) in blocks.items():
            sub = sample_block(block, k, rng).copy()
            sub["label_binary"] = label
            sub["category"] = category
            parts[split].append(sub)

        if i % 50 == 0:
            print(f"[INFO] {i}/{len(files)} fichiers traités")

    out = {s: pd.concat(p, ignore_index=True) for s, p in parts.items()}
    for s, d in out.items():
        n_b = (d["label_binary"] == 0).sum()
        print(f"[INFO] {s.upper():5s} : {len(d)} lignes "
              f"({n_b} Benign = {n_b/len(d):.1%}) | "
              f"{d['category'].nunique()} catégories")
    return out["train"], out["val"], out["test"]


# ============================================================
# ÉTAPE 2/3 — Nettoyage + normalisation (fit sur train uniquement)
# ============================================================
def clean_features(df, cols):
    df = df.copy()
    df[cols] = df[cols].replace([np.inf, -np.inf], np.nan)
    n0 = len(df)
    df = df.dropna(subset=cols)
    if len(df) < n0:
        print(f"[INFO] {n0 - len(df)} lignes supprimées (NaN/inf)")
    return df


def normalize(train_df, val_df, test_df):
    for col in EDGE_LOG1P_COLUMNS:
        shift = -train_df[col].min() if train_df[col].min() < 0 else 0.0
        for d in (train_df, val_df, test_df):
            d[col] = np.log1p(d[col] + shift)

    edge_scaler = RobustScaler()
    train_df[EDGE_FEATURES_SRC] = edge_scaler.fit_transform(train_df[EDGE_FEATURES_SRC])
    val_df[EDGE_FEATURES_SRC] = edge_scaler.transform(val_df[EDGE_FEATURES_SRC])
    test_df[EDGE_FEATURES_SRC] = edge_scaler.transform(test_df[EDGE_FEATURES_SRC])

    node_scaler = RobustScaler()
    train_df[NODE_FEATURES_SRC] = node_scaler.fit_transform(train_df[NODE_FEATURES_SRC])
    val_df[NODE_FEATURES_SRC] = node_scaler.transform(val_df[NODE_FEATURES_SRC])
    test_df[NODE_FEATURES_SRC] = node_scaler.transform(test_df[NODE_FEATURES_SRC])
    print("[INFO] log1p + RobustScaler (fit sur train uniquement)")
    return train_df, val_df, test_df


# ============================================================
# ÉTAPE 4 — Graphe k-NN
# ============================================================

def build_graph(train_df, val_df, test_df, k):
    # Identifier l'ensemble auquel appartient chaque ligne
    train_df = train_df.copy().assign(split="train")
    val_df = val_df.copy().assign(split="val")
    test_df = test_df.copy().assign(split="test")

    df = pd.concat(
        [train_df, val_df, test_df],
        ignore_index=True
    )

    # Features et labels binaires utilisés par les modèles GNN
    X_edges = df[EDGE_FEATURES_SRC].values.astype(np.float32)
    X_nodes = df[NODE_FEATURES_SRC].values.astype(np.float32)
    y = df["label_binary"].values.astype(np.int64)

    # Créer une correspondance stable entre catégories et identifiants
    category_names = sorted(df["category"].astype(str).unique().tolist())
    category_to_id = {
        name: idx for idx, name in enumerate(category_names)
    }
    category_ids = (
        df["category"].astype(str)
        .map(category_to_id)
        .values.astype(np.int64)
    )

    print(f"[INFO] Nombre de catégories : {len(category_names)}")
    print(f"[INFO] k-NN (k={k}) sur {len(df)} nœuds...")

    # Construire le graphe k-NN
    nbrs = NearestNeighbors(
        n_neighbors=k + 1,
        n_jobs=-1
    ).fit(X_edges)

    _, idx = nbrs.kneighbors(X_edges)

    edges = []
    for i, neigh in enumerate(idx):
        for j in neigh[1:]:
            edges.append([i, j])
            edges.append([j, i])

    edge_index = torch.tensor(
        edges, dtype=torch.long
    ).t().contiguous()

    # Construire le graphe avec les catégories de chaque nœud
    data = Data(
        x=torch.tensor(X_nodes, dtype=torch.float32),
        edge_index=edge_index,
        y=torch.tensor(y, dtype=torch.long),
        category=torch.tensor(category_ids, dtype=torch.long)
    )

    # Masques des trois ensembles
    data.train_mask = torch.tensor(
        (df["split"] == "train").values,
        dtype=torch.bool
    )
    data.val_mask = torch.tensor(
        (df["split"] == "val").values,
        dtype=torch.bool
    )
    data.test_mask = torch.tensor(
        (df["split"] == "test").values,
        dtype=torch.bool
    )

    # Sauvegarder la correspondance ID -> nom de catégorie
    with open(CATEGORIES_FILE, "w", encoding="utf-8") as f:
        json.dump(category_names, f, ensure_ascii=False, indent=2)

    print(f"[INFO] Graphe : {data.num_nodes} nœuds, "
          f"{data.num_edges} arêtes | "
          f"Train/Val/Test : "
          f"{data.train_mask.sum().item()}/"
          f"{data.val_mask.sum().item()}/"
          f"{data.test_mask.sum().item()}")

    print("[INFO] Catégories enregistrées dans data.category")
    print(f"[OK] Correspondance des catégories : {CATEGORIES_FILE}")

    return data



if __name__ == "__main__":
    needed = list(set(EDGE_FEATURES_SRC) | set(NODE_FEATURES_SRC))
    train_df, val_df, test_df = load_blocks(RAW_CSV_DIR, needed, ROWS_PER_FILE, RANDOM_STATE)

    train_df = clean_features(train_df, needed)
    val_df = clean_features(val_df, needed)
    test_df = clean_features(test_df, needed)

    train_df, val_df, test_df = normalize(train_df, val_df, test_df)
    graph = build_graph(train_df, val_df, test_df, K_NEIGHBORS)

    torch.save(graph, OUTPUT_FILE)
    print(f"\n[OK] Graphe sauvegardé : {OUTPUT_FILE}")
    print("[OK] Dans les scripts d'entraînement, pointe GRAPH_FILE vers ce fichier "
          "et utilise des noms de modèle distincts (ex. model_iot_blocks.pt).")