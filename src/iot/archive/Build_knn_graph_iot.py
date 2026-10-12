"""
Construction du graphe k-NN pour le pipeline IoT (CICIoT2023).
PFA - Hybrid Feature Selection and GNN for Intrusion Detection

VERSION 5 — 3 corrections par rapport à la V4 :

1. NODE_FEATURES_SRC aligné sur la liste FINALE fixée avec Malak (contrat,
   section 3) : packet_count, total_bytes, packet_len_mean/std/var/min/max,
   flow_rate. Retrait de IAT et ack_flag_number (hors liste finale).

2. NODE_FEATURES normalisées avec log1p + RobustScaler (comme les EDGE
   FEATURES), pour rester cohérent avec la proposition envoyée à Malak
   (RobustScaler des deux côtés, pour gérer les distributions à queue
   lourde qu'on a diagnostiquées).

3. Seul le TRAIN est enrichi en vrais Benign (TARGET_BENIGN_RATIO). Le TEST
   reste à la proportion naturelle (~2.35%) pour une évaluation réaliste et
   comparable aux runs précédents (diagnostic FN, homophilie). Le pool
   Benign est donc séparé en deux AVANT le split : une partie pour enrichir
   le train, une partie mise de côté à proportion naturelle pour le test.

Étapes :
1. Séparer le pool Benign/Attack, réserver une portion Benign proportion
   naturelle pour le test, enrichir le reste (train) selon TARGET_BENIGN_RATIO
2. Extraire EDGE_FEATURES (15) et NODE_FEATURES (8, liste finale)
3. Normalisation : log1p + RobustScaler pour les deux espaces (fit sur train)
4. k-NN sur EDGE_FEATURES
5. Graphe avec x = NODE_FEATURES, sauvegarde .pt

Utilisation :
    python Build_knn_graph_iot.py
"""

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import RobustScaler
import torch
from torch_geometric.data import Data

# ============================================================
# CONFIGURATION
# ============================================================
INPUT_FILE = r"D:\CICIoT2023\ciciot2023_sample.csv"
OUTPUT_FILE = r"D:\CICIoT2023\graph_iot_knn.pt"

N_NODES_TARGET = 50_000
K_NEIGHBORS = 5
TEST_SIZE = 0.3
RANDOM_STATE = 42

# Proportion de VRAIS Benign visée dans le TRAIN uniquement (le test reste
# à la proportion naturelle, cf. correction n°3 ci-dessus)
TARGET_BENIGN_RATIO_TRAIN = 0.18

EDGE_FEATURES_SRC = [
    "Protocol Type", "Number", "Tot sum", "AVG", "Std", "Variance",
    "Min", "Max", "Rate", "IAT",
    "syn_flag_number", "ack_flag_number", "fin_flag_number",
    "rst_flag_number", "psh_flag_number",
]
EDGE_LOG1P_COLUMNS = [c for c in EDGE_FEATURES_SRC if c != "Protocol Type"]

# ⚠️ CORRECTION 1 — liste FINALE alignée avec le contrat (section 3),
# fixée après comparaison des sélections hybrides avec Malak.
NODE_FEATURES_SRC = [
    "Number", "Tot sum", "AVG", "Std", "Variance", "Min", "Max", "Rate",
]
NODE_FEATURES_UNIFIED_NAMES = [
    "feat_02_packet_count", "feat_03_total_bytes", "feat_04_packet_len_mean",
    "feat_05_packet_len_std", "feat_06_packet_len_var", "feat_07_packet_len_min",
    "feat_08_packet_len_max", "feat_09_flow_rate",
]
NODE_LOG1P_COLUMNS = list(NODE_FEATURES_SRC)  # toutes, même traitement que les edges


# ============================================================
# ÉTAPE 1 — Charger, réserver le test à proportion naturelle,
#           enrichir uniquement le train en vrais Benign
# ============================================================
def load_and_split_with_natural_test(input_file, n_target, target_benign_ratio_train,
                                       test_size, random_state):
    print(f"[INFO] Chargement de {input_file} ...")
    needed_cols = list(set(EDGE_FEATURES_SRC) | set(NODE_FEATURES_SRC) | {"label_binary"})
    df = pd.read_csv(input_file, usecols=needed_cols)
    print(f"[INFO] Fichier chargé : {len(df)} lignes")

    benign_pool = df[df["label_binary"] == 0]
    attack_pool = df[df["label_binary"] == 1]
    print(f"[INFO] Benign disponibles dans le CSV complet : {len(benign_pool)}")

    rng = np.random.default_rng(random_state)

    # Taille totale du graphe et répartition train/test approximative
    n_test_total = int(n_target * test_size)
    n_train_total = n_target - n_test_total

    # TEST : proportion naturelle (~2.35%, mesurée sur le pool complet)
    natural_ratio = len(benign_pool) / len(df)
    n_benign_test = max(1, int(n_test_total * natural_ratio))
    n_attack_test = n_test_total - n_benign_test

    # TRAIN : proportion enrichie (TARGET_BENIGN_RATIO_TRAIN)
    n_benign_train = int(n_train_total * target_benign_ratio_train)
    n_attack_train = n_train_total - n_benign_train

    n_benign_needed = n_benign_test + n_benign_train
    if n_benign_needed > len(benign_pool):
        raise ValueError(
            f"Besoin de {n_benign_needed} Benign réels (train+test) mais "
            f"seulement {len(benign_pool)} disponibles. Réduis "
            f"TARGET_BENIGN_RATIO_TRAIN ou N_NODES_TARGET."
        )

    benign_idx_all = rng.choice(benign_pool.index, size=n_benign_needed, replace=False)
    benign_idx_test = benign_idx_all[:n_benign_test]
    benign_idx_train = benign_idx_all[n_benign_test:]

    attack_idx_all = rng.choice(
        attack_pool.index, size=n_attack_test + n_attack_train, replace=False
    )
    attack_idx_test = attack_idx_all[:n_attack_test]
    attack_idx_train = attack_idx_all[n_attack_test:]

    test_df = pd.concat([df.loc[benign_idx_test], df.loc[attack_idx_test]], ignore_index=True)
    train_df = pd.concat([df.loc[benign_idx_train], df.loc[attack_idx_train]], ignore_index=True)

    test_df = test_df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    train_df = train_df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    print(f"[INFO] TRAIN : {len(train_df)} lignes "
          f"({train_df['label_binary'].value_counts().to_dict()}) "
          f"-> {n_benign_train / len(train_df):.1%} Benign (enrichi)")
    print(f"[INFO] TEST  : {len(test_df)} lignes "
          f"({test_df['label_binary'].value_counts().to_dict()}) "
          f"-> {n_benign_test / len(test_df):.1%} Benign (proportion naturelle)")

    # ⚠️ AJOUT — split validation : on prélève 20% du TRAIN pour la
    # validation (utilisée pour l'early stopping), le reste sert vraiment
    # à apprendre. Le TEST reste totalement à part, jamais utilisé avant
    # le chiffre final.
    from sklearn.model_selection import train_test_split as _tts
    train_df, val_df = _tts(
        train_df, test_size=0.2, stratify=train_df["label_binary"],
        random_state=random_state
    )
    print(f"[INFO] Split train/val (80/20 du train ci-dessus) : "
          f"{len(train_df)} train / {len(val_df)} validation")

    return train_df, val_df, test_df


# ============================================================
# ÉTAPE 2 — Nettoyage
# ============================================================
def clean_features(df, all_needed):
    df = df.copy()
    df[all_needed] = df[all_needed].replace([np.inf, -np.inf], np.nan)
    n_before = len(df)
    df = df.dropna(subset=all_needed)
    if len(df) < n_before:
        print(f"[INFO] {n_before - len(df)} lignes supprimées (NaN/inf)")
    return df


# ============================================================
# ÉTAPE 3 — Normalisation : log1p + RobustScaler, cohérente edge/node
# ============================================================
def apply_log1p(train_df, val_df, test_df, columns):
    shift_by_col = {}
    for col in columns:
        min_train = train_df[col].min()
        shift = -min_train if min_train < 0 else 0.0
        shift_by_col[col] = shift
        if shift > 0:
            print(f"[ATTENTION] {col} contient des valeurs négatives dans "
                  f"le train (min={min_train}) — décalage de {shift} appliqué.")
        train_df[col] = np.log1p(train_df[col] + shift)
        val_df[col] = np.log1p(val_df[col] + shift)
        test_df[col] = np.log1p(test_df[col] + shift)
    return train_df, val_df, test_df, shift_by_col


def normalize_features(train_df, val_df, test_df, edge_features_src, edge_log1p_cols,
                        node_features_src, node_log1p_cols):
    train_df, val_df, test_df, edge_shift = apply_log1p(
        train_df, val_df, test_df, edge_log1p_cols
    )
    # NODE_FEATURES_SRC est un sous-ensemble d'EDGE_FEATURES_SRC ici, donc le
    # log1p est déjà appliqué sur ces colonnes au moment du bloc ci-dessus —
    # pas besoin de le refaire (éviterait un double log1p).

    edge_scaler = RobustScaler()
    train_df[edge_features_src] = edge_scaler.fit_transform(train_df[edge_features_src])
    val_df[edge_features_src] = edge_scaler.transform(val_df[edge_features_src])
    test_df[edge_features_src] = edge_scaler.transform(test_df[edge_features_src])

    node_scaler = RobustScaler()
    train_df[node_features_src] = node_scaler.fit_transform(train_df[node_features_src])
    val_df[node_features_src] = node_scaler.transform(val_df[node_features_src])
    test_df[node_features_src] = node_scaler.transform(test_df[node_features_src])

    print("[INFO] EDGE_FEATURES  : log1p + RobustScaler (fit sur train)")
    print("[INFO] NODE_FEATURES  : RobustScaler (fit sur train, log1p déjà appliqué "
          "car sous-ensemble des edge features)")

    return train_df, val_df, test_df, edge_scaler, node_scaler


# ============================================================
# ÉTAPE 4 — Construire le graphe
# ============================================================
def build_knn_graph(train_df, val_df, test_df, edge_features_src, node_features_src, k):
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()
    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"
    df_final = pd.concat([train_df, val_df, test_df], ignore_index=True)

    X_edges = df_final[edge_features_src].values.astype(np.float32)
    X_nodes = df_final[node_features_src].values.astype(np.float32)
    y = df_final["label_binary"].values.astype(np.int64)

    print(f"[INFO] Calcul des {k} plus proches voisins sur l'espace "
          f"EDGE_FEATURES ({X_edges.shape[1]}D) pour {len(X_edges)} nœuds...")
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

    print(f"[INFO] Graphe construit : {data.num_nodes} nœuds, {data.num_edges} arêtes, "
          f"{data.x.shape[1]} features par nœud")
    print(f"[INFO] Train/Val/Test : {train_mask.sum().item()}/"
          f"{val_mask.sum().item()}/{test_mask.sum().item()}")
    return data


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    train_df, val_df, test_df = load_and_split_with_natural_test(
        INPUT_FILE, N_NODES_TARGET, TARGET_BENIGN_RATIO_TRAIN, TEST_SIZE, RANDOM_STATE
    )

    all_needed = list(set(EDGE_FEATURES_SRC) | set(NODE_FEATURES_SRC))
    train_df = clean_features(train_df, all_needed)
    val_df = clean_features(val_df, all_needed)
    test_df = clean_features(test_df, all_needed)

    train_df, val_df, test_df, edge_scaler, node_scaler = normalize_features(
        train_df, val_df, test_df, EDGE_FEATURES_SRC, EDGE_LOG1P_COLUMNS,
        NODE_FEATURES_SRC, NODE_LOG1P_COLUMNS
    )

    graph = build_knn_graph(train_df, val_df, test_df, EDGE_FEATURES_SRC, NODE_FEATURES_SRC, K_NEIGHBORS)

    torch.save(graph, OUTPUT_FILE)
    print(f"\n[OK] Graphe sauvegardé : {OUTPUT_FILE}")
    print(f"[OK] {graph.num_nodes} nœuds, {graph.num_edges} arêtes, "
          f"{graph.x.shape[1]} features par nœud")
    print("[OK] Train enrichi en vrais Benign, val et test à proportion naturelle.")
    print("[OK] Relance les scripts d'entraînement (train/val/test corrigé).")