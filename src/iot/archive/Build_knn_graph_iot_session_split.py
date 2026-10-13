"""
Construction du graphe k-NN IoT — split par SESSION DE CAPTURE (fichier).
PFA - Hybrid Feature Selection and GNN for Intrusion Detection

CONTEXTE : CICIoT2023 n'a pas de colonne Timestamp exploitable. Pour éviter
la fuite de données entre flows d'une même session de capture (le risque
principal qu'un split temporel cherche à éviter), on répartit des FICHIERS
ENTIERS entre train/val/test plutôt que des lignes individuelles.

Concrètement : pour chaque catégorie (dossier), les fichiers .pcap.csv
disponibles sont répartis ~70% train / 15% val / 15% test PAR FICHIER
(jamais un même fichier scindé entre plusieurs splits).

Étapes :
1. Lister les fichiers par catégorie, les répartir train/val/test
2. Charger les lignes de chaque groupe de fichiers, enrichir le train en
   Benign réel (comme avant), garder val/test à proportion naturelle
3. Nettoyage, normalisation (log1p + RobustScaler, fit sur train)
4. k-NN sur EDGE_FEATURES, graphe avec x = NODE_FEATURES
5. Sauvegarde .pt

Utilisation :
    python build_knn_graph_iot_session_split.py
"""

import os
import glob
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import RobustScaler
import torch
from torch_geometric.data import Data

# ============================================================
# CONFIGURATION
# ============================================================
RAW_CSV_DIR = r"D:\CICIoT2023\CSV"          # dossier avec les sous-dossiers par catégorie
OUTPUT_FILE = r"D:\CICIoT2023\graph_iot_knn_session.pt"

N_ROWS_PER_SPLIT_CAP = 100_000  # relevé : le train doit rester largement le plus gros ensemble
RANDOM_STATE = 42
TARGET_BENIGN_RATIO_TRAIN = None  # ⚠️ enrichissement désactivé (voir note ci-dessous)

# NOTE : avec seulement 4 fichiers Benign au total dans tout le dataset,
# répartis 2/1/1 entre train/val/test, forcer un ratio Benign artificiel
# (comme dans la V5 aléatoire) fait fondre le train à quelques milliers de
# lignes -- pire que le problème qu'on cherchait à résoudre. On garde donc
# le ratio naturel de chaque split, et on laisse class_weights (déjà dans
# Train_gnn_iot.py) gérer le déséquilibre plutôt que le sous-échantillonnage.

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
# ÉTAPE 1 — Répartir les FICHIERS (pas les lignes) en train/val/test
# ============================================================
def split_files_by_session(raw_csv_dir, random_state):
    rng = np.random.default_rng(random_state)
    categories = [d for d in os.listdir(raw_csv_dir)
                  if os.path.isdir(os.path.join(raw_csv_dir, d))]

    split_files = {"train": [], "val": [], "test": []}

    for cat in categories:
        files = sorted(glob.glob(os.path.join(raw_csv_dir, cat, "*.csv")))
        if not files:
            continue
        rng.shuffle(files)

        n = len(files)
        if n == 1:
            # Une seule session pour cette catégorie -> pas de split possible
            # sans fuite ; on la met en TRAIN uniquement (limite à documenter :
            # cette catégorie n'est représentée que dans le train).
            split_files["train"].extend(files)
            continue

        n_test = max(1, round(n * 0.15))
        n_val = max(1, round(n * 0.15))
        n_train = n - n_test - n_val
        if n_train < 1:
            n_train, n_val, n_test = n - 2, 1, 1

        split_files["train"].extend(files[:n_train])
        split_files["val"].extend(files[n_train:n_train + n_val])
        split_files["test"].extend(files[n_train + n_val:])

    for split_name, files in split_files.items():
        print(f"[INFO] {split_name.upper()} : {len(files)} fichiers "
              f"(sessions de capture)")

    return split_files


# ============================================================
# ÉTAPE 2 — Charger les lignes de chaque split (avec enrichissement Benign
#           uniquement pour le train)
# ============================================================
def load_split(files, needed_cols, cap_rows, random_state, benign_folder_names={"benign_final", "benign"}):
    rng = np.random.default_rng(random_state)
    dfs = []
    for f in files:
        category = os.path.basename(os.path.dirname(f))
        df_chunk = pd.read_csv(f, usecols=lambda c: c in needed_cols)
        df_chunk["label_binary"] = 0 if category.strip().lower() in benign_folder_names else 1
        dfs.append(df_chunk)
    df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame(columns=needed_cols + ["label_binary"])

    if len(df) > cap_rows:
        df = df.groupby("label_binary", group_keys=False).sample(
            frac=cap_rows / len(df), random_state=random_state
        )
    return df


def enrich_train_benign(train_df, target_ratio, random_state):
    n_benign = (train_df["label_binary"] == 0).sum()
    n_attack = (train_df["label_binary"] == 1).sum()
    current_ratio = n_benign / len(train_df) if len(train_df) else 0
    print(f"[INFO] TRAIN avant enrichissement : {n_benign} Benign / {n_attack} Attack "
          f"({current_ratio:.1%} Benign)")
    if current_ratio >= target_ratio or n_benign == 0:
        print("[INFO] Pas d'enrichissement nécessaire (déjà au-dessus de la cible, "
              "ou aucun Benign disponible dans ces sessions).")
        return train_df

    # Sous-échantillonne les Attack pour remonter la proportion de Benign
    n_attack_target = int(n_benign * (1 - target_ratio) / target_ratio)
    attack_df = train_df[train_df["label_binary"] == 1].sample(
        n=min(n_attack_target, n_attack), random_state=random_state
    )
    benign_df = train_df[train_df["label_binary"] == 0]
    result = pd.concat([benign_df, attack_df], ignore_index=True).sample(
        frac=1.0, random_state=random_state
    ).reset_index(drop=True)
    print(f"[INFO] TRAIN après enrichissement : {(result['label_binary']==0).sum()} Benign / "
          f"{(result['label_binary']==1).sum()} Attack "
          f"({(result['label_binary']==0).mean():.1%} Benign)")
    return result


# ============================================================
# ÉTAPE 3 — Nettoyage + normalisation (identique à la V5)
# ============================================================
def clean_features(df, all_needed):
    df = df.copy()
    df[all_needed] = df[all_needed].replace([np.inf, -np.inf], np.nan)
    n_before = len(df)
    df = df.dropna(subset=all_needed)
    if len(df) < n_before:
        print(f"[INFO] {n_before - len(df)} lignes supprimées (NaN/inf)")
    return df


def apply_log1p(train_df, val_df, test_df, columns):
    for col in columns:
        min_train = train_df[col].min()
        shift = -min_train if min_train < 0 else 0.0
        train_df[col] = np.log1p(train_df[col] + shift)
        val_df[col] = np.log1p(val_df[col] + shift)
        test_df[col] = np.log1p(test_df[col] + shift)
    return train_df, val_df, test_df


def normalize_features(train_df, val_df, test_df, edge_features_src, edge_log1p_cols, node_features_src):
    train_df, val_df, test_df = apply_log1p(train_df, val_df, test_df, edge_log1p_cols)

    edge_scaler = RobustScaler()
    train_df[edge_features_src] = edge_scaler.fit_transform(train_df[edge_features_src])
    val_df[edge_features_src] = edge_scaler.transform(val_df[edge_features_src])
    test_df[edge_features_src] = edge_scaler.transform(test_df[edge_features_src])

    node_scaler = RobustScaler()
    train_df[node_features_src] = node_scaler.fit_transform(train_df[node_features_src])
    val_df[node_features_src] = node_scaler.transform(val_df[node_features_src])
    test_df[node_features_src] = node_scaler.transform(test_df[node_features_src])

    return train_df, val_df, test_df


# ============================================================
# ÉTAPE 4 — Construire le graphe
# ============================================================
def build_knn_graph(train_df, val_df, test_df, edge_features_src, node_features_src, k=5):
    train_df = train_df.copy(); val_df = val_df.copy(); test_df = test_df.copy()
    train_df["split"] = "train"; val_df["split"] = "val"; test_df["split"] = "test"
    df_final = pd.concat([train_df, val_df, test_df], ignore_index=True)

    X_edges = df_final[edge_features_src].values.astype(np.float32)
    X_nodes = df_final[node_features_src].values.astype(np.float32)
    y = df_final["label_binary"].values.astype(np.int64)

    print(f"[INFO] Calcul des {k} plus proches voisins pour {len(X_edges)} nœuds...")
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm="auto", n_jobs=-1).fit(X_edges)
    _, indices = nbrs.kneighbors(X_edges)

    edge_list = []
    for node_idx, neighbors in enumerate(indices):
        for neighbor_idx in neighbors[1:]:
            edge_list.append([node_idx, neighbor_idx])
            edge_list.append([neighbor_idx, node_idx])

    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    x = torch.tensor(X_nodes, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    train_mask = torch.tensor((df_final["split"] == "train").values, dtype=torch.bool)
    val_mask = torch.tensor((df_final["split"] == "val").values, dtype=torch.bool)
    test_mask = torch.tensor((df_final["split"] == "test").values, dtype=torch.bool)

    data = Data(x=x, edge_index=edge_index, y=y_tensor)
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    print(f"[INFO] Graphe : {data.num_nodes} nœuds, {data.num_edges} arêtes")
    print(f"[INFO] Train/Val/Test : {train_mask.sum().item()}/{val_mask.sum().item()}/{test_mask.sum().item()}")
    return data


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    needed_cols = list(set(EDGE_FEATURES_SRC) | set(NODE_FEATURES_SRC))

    split_files = split_files_by_session(RAW_CSV_DIR, RANDOM_STATE)

    train_df = load_split(split_files["train"], needed_cols, N_ROWS_PER_SPLIT_CAP, RANDOM_STATE)
    val_df = load_split(split_files["val"], needed_cols, N_ROWS_PER_SPLIT_CAP // 3, RANDOM_STATE)
    test_df = load_split(split_files["test"], needed_cols, N_ROWS_PER_SPLIT_CAP // 3, RANDOM_STATE)

    if TARGET_BENIGN_RATIO_TRAIN is not None:
        train_df = enrich_train_benign(train_df, TARGET_BENIGN_RATIO_TRAIN, RANDOM_STATE)
    else:
        n_b = (train_df["label_binary"] == 0).sum()
        n_a = (train_df["label_binary"] == 1).sum()
        print(f"[INFO] TRAIN (ratio naturel, pas d'enrichissement) : "
              f"{n_b} Benign / {n_a} Attack ({n_b/(n_b+n_a):.1%} Benign)")

    train_df = clean_features(train_df, needed_cols)
    val_df = clean_features(val_df, needed_cols)
    test_df = clean_features(test_df, needed_cols)

    train_df, val_df, test_df = normalize_features(
        train_df, val_df, test_df, EDGE_FEATURES_SRC, EDGE_LOG1P_COLUMNS, NODE_FEATURES_SRC
    )

    graph = build_knn_graph(train_df, val_df, test_df, EDGE_FEATURES_SRC, NODE_FEATURES_SRC)

    torch.save(graph, OUTPUT_FILE)
    print(f"\n[OK] Graphe sauvegardé (split par session) : {OUTPUT_FILE}")
    print("[OK] Relance les scripts d'entraînement en pointant GRAPH_FILE vers ce fichier "
          "pour comparer au split aléatoire précédent.")