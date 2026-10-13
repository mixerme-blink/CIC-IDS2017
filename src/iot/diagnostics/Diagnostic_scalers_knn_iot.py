"""
Diagnostic — impact du scaler sur la topologie du graphe k-NN (CICIoT2023).
PFA - Hybrid Feature Selection and GNN for Intrusion Detection

Objectif :
On a confirmé que les 7 features ajoutées à EDGE_FEATURES ne sont PAS plates
en valeur brute (CV jusqu'à 10.8, milliers de valeurs uniques). L'hypothèse
actuelle est que le Min-Max, très sensible aux valeurs extrêmes, écrase les
distributions heavy-tailed (IAT, Variance, Std, Rate, ...) dans une bande
proche de 0, ce qui neutralise leur pouvoir de séparation dans le calcul de
distance euclidienne du k-NN.

Ce script NE relance PAS l'entraînement GNN. Il compare uniquement la
TOPOLOGIE du graphe (quels sont les voisins de chaque nœud) obtenue avec
3 stratégies de normalisation de EDGE_FEATURES :

    A) MinMaxScaler                  (baseline actuelle)
    B) RobustScaler                  (médiane / IQR, moins sensible aux outliers)
    C) log1p puis RobustScaler       (compresse la queue, PUIS robust scale)

Métrique : pourcentage moyen de voisins communs (Jaccard-like overlap) entre
la baseline (A) et chacune des alternatives (B, C), sur les k plus proches
voisins de chaque nœud.

Interprétation attendue :
    - overlap(A, B) proche de 100%  -> RobustScaler seul ne change presque
      rien, l'échelle n'était pas le facteur dominant.
    - overlap(A, C) nettement < 100% -> log1p change réellement la
      topologie -> c'est la queue lourde qui écrasait le signal, pas
      l'échelle seule. Dans ce cas, on utilise directement (C) pour la
      version finale du graphe, pas la peine de tester (B) en vrai
      entraînement.

Utilisation :
    python diagnostic_scalers_knn_iot.py
"""

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler, RobustScaler

# ============================================================
# CONFIGURATION — mêmes valeurs que build_knn_graph_iot.py
# ============================================================
INPUT_FILE = r"D:\CICIoT2023\ciciot2023_sample.csv"

N_NODES_TARGET = 50_000
K_NEIGHBORS = 5
RANDOM_STATE = 42

EDGE_FEATURES_SRC = [
    "Protocol Type", "Number", "Tot sum", "AVG", "Std", "Variance",
    "Min", "Max", "Rate", "IAT",
    "syn_flag_number", "ack_flag_number", "fin_flag_number",
    "rst_flag_number", "psh_flag_number",
]

# Colonnes sur lesquelles on applique log1p AVANT le RobustScaler, dans la
# version (C). On exclut "Protocol Type" (catégorielle, 5 valeurs, pas de
# queue lourde) — le log1p n'a rien à y faire.
LOG1P_COLUMNS = [c for c in EDGE_FEATURES_SRC if c != "Protocol Type"]


# ============================================================
# ÉTAPE 1 — Charger, sous-échantillonner, nettoyer (identique au script principal)
# ============================================================
def load_and_clean(input_file, n_target, edge_features_src, random_state):
    print(f"[INFO] Chargement de {input_file} ...")
    df = pd.read_csv(input_file)
    print(f"[INFO] Fichier chargé : {len(df)} lignes")

    if len(df) > n_target:
        frac = n_target / len(df)
        df = df.groupby("label_binary", group_keys=False).sample(
            frac=frac, random_state=random_state
        )
        print(f"[INFO] Après sous-échantillonnage : {len(df)} lignes")

    df = df[edge_features_src + ["label_binary"]].copy()
    df[edge_features_src] = df[edge_features_src].replace([np.inf, -np.inf], np.nan)
    n_before = len(df)
    df = df.dropna(subset=edge_features_src).reset_index(drop=True)
    if len(df) < n_before:
        print(f"[INFO] {n_before - len(df)} lignes supprimées (NaN/inf)")

    return df


# ============================================================
# ÉTAPE 2 — Construire les 3 versions de l'espace EDGE_FEATURES
# ============================================================
def build_feature_spaces(df, edge_features_src, log1p_columns):
    X_raw = df[edge_features_src].values.astype(np.float64)

    # A) MinMax — baseline actuelle
    X_minmax = MinMaxScaler().fit_transform(X_raw)

    # B) RobustScaler seul
    X_robust = RobustScaler().fit_transform(X_raw)

    # C) log1p (sur les colonnes concernées) puis RobustScaler
    df_log = df[edge_features_src].copy()
    for col in log1p_columns:
        # log1p suppose des valeurs >= 0 ; on sécurise au cas où (ne devrait
        # pas arriver pour des compteurs/débits/délais réseau, mais on
        # protège contre un bug amont plutôt que de planter ici).
        min_val = df_log[col].min()
        if min_val < 0:
            print(f"[ATTENTION] {col} contient des valeurs négatives "
                  f"(min={min_val}) — décalage appliqué avant log1p.")
            df_log[col] = df_log[col] - min_val
        df_log[col] = np.log1p(df_log[col])
    X_log_robust = RobustScaler().fit_transform(df_log.values.astype(np.float64))

    return {
        "A_minmax": X_minmax,
        "B_robust": X_robust,
        "C_log1p_robust": X_log_robust,
    }


# ============================================================
# ÉTAPE 3 — k plus proches voisins pour chaque espace
# ============================================================
def compute_knn_indices(X, k):
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm="auto", n_jobs=-1).fit(X)
    _, indices = nbrs.kneighbors(X)
    # on retire le nœud lui-même (toujours en position 0)
    return indices[:, 1:]


# ============================================================
# ÉTAPE 4 — Taux de recouvrement des voisins entre deux versions
# ============================================================
def neighbor_overlap(indices_a, indices_b):
    """
    Pour chaque nœud, % de voisins en commun entre les deux listes de
    k plus proches voisins (ordre non pris en compte, ensemble seulement).
    Retourne la moyenne sur tous les nœuds + quelques stats de dispersion.
    """
    n_nodes, k = indices_a.shape
    overlaps = np.empty(n_nodes, dtype=np.float64)
    for i in range(n_nodes):
        set_a = set(indices_a[i])
        set_b = set(indices_b[i])
        overlaps[i] = len(set_a & set_b) / k
    return overlaps


def summarize(name_a, name_b, overlaps):
    print(f"\n--- Recouvrement des {K_NEIGHBORS} voisins : {name_a} vs {name_b} ---")
    print(f"    Moyenne     : {overlaps.mean() * 100:6.2f} %")
    print(f"    Médiane     : {np.median(overlaps) * 100:6.2f} %")
    print(f"    Écart-type  : {overlaps.std() * 100:6.2f} %")
    print(f"    % de nœuds avec overlap == 100% : "
          f"{(overlaps == 1.0).mean() * 100:6.2f} %")
    print(f"    % de nœuds avec overlap == 0%   : "
          f"{(overlaps == 0.0).mean() * 100:6.2f} %")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    df = load_and_clean(INPUT_FILE, N_NODES_TARGET, EDGE_FEATURES_SRC, RANDOM_STATE)

    print("\n[INFO] Construction des 3 espaces de features...")
    spaces = build_feature_spaces(df, EDGE_FEATURES_SRC, LOG1P_COLUMNS)

    print(f"[INFO] Calcul des {K_NEIGHBORS} plus proches voisins pour chaque "
          f"espace ({len(df)} nœuds)...")
    knn = {name: compute_knn_indices(X, K_NEIGHBORS) for name, X in spaces.items()}

    print("\n" + "=" * 60)
    print("RÉSULTATS — recouvrement des voisins vs baseline (A_minmax)")
    print("=" * 60)

    overlap_ab = neighbor_overlap(knn["A_minmax"], knn["B_robust"])
    summarize("A_minmax", "B_robust", overlap_ab)

    overlap_ac = neighbor_overlap(knn["A_minmax"], knn["C_log1p_robust"])
    summarize("A_minmax", "C_log1p_robust", overlap_ac)

    # Bonus : B vs C, pour savoir si le log1p change encore beaucoup les
    # choses une fois qu'on est déjà passé à un scaler robuste.
    overlap_bc = neighbor_overlap(knn["B_robust"], knn["C_log1p_robust"])
    summarize("B_robust", "C_log1p_robust", overlap_bc)

    print("\n" + "=" * 60)
    print("LECTURE DES RÉSULTATS")
    print("=" * 60)
    print("""
- Si overlap(A, B) est proche de 100% :
    -> RobustScaler seul ne suffit pas, l'échelle n'explique pas tout.
- Si overlap(A, C) est nettement < 100% (ex: en dessous de 70-80%) :
    -> confirme l'hypothèse de la queue lourde : le log1p change vraiment
       qui sont les voisins de chaque nœud. Utiliser directement la
       version C (log1p + RobustScaler) pour reconstruire le graphe final
       et relancer train_gnn_iot.py + baseline_no_graph_iot.py.
- Si même overlap(A, C) reste proche de 100% :
    -> le problème n'est probablement pas la normalisation mais autre
       chose (ex: choix de k, features intrinsèquement peu discriminantes
       pour cette tâche, ou GraphSAGE qui n'exploite pas bien le signal
       même quand il est présent). Documenter ce résultat négatif comme
       limite légitime plutôt que de continuer à itérer sur le scaler.
""")