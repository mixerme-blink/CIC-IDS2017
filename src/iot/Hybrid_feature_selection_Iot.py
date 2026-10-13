"""
Sélection hybride de features (filter + embedded) — pipeline IoT (CICIoT2023).
PFA - Hybrid Feature Selection and GNN for Intrusion Detection

IMPORTANT : la sélection ne porte PAS sur les 39 colonnes brutes du dataset,
mais UNIQUEMENT sur les 15 features candidates déjà harmonisées avec le
pipeline cloud (voir contrat_features_harmonisees.md). On ne peut pas
sélectionner librement sur les 39 colonnes, sinon on retombe sur des features
absentes côté cloud (HTTP, TTL...) et le GNN partagé devient impossible.

Étapes :
1. Charger l'échantillon et extraire les 15 features candidates
2. FILTER  : information mutuelle entre chaque feature et le label
3. EMBEDDED : importance des variables via Random Forest
4. Combiner les deux scores (rang moyen) et proposer un top-N
5. Exporter un tableau de scores à comparer avec celui de Malak (même méthode,
   côté cloud, sur les mêmes 15 candidates)

Utilisation :
    python hybrid_feature_selection_iot.py
"""

import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler

# ============================================================
# CONFIGURATION
# ============================================================
INPUT_FILE = r"D:\CICIoT2023\ciciot2023_sample.csv"
OUTPUT_SCORES_FILE = r"D:\CICIoT2023\feature_selection_scores_iot.csv"
TOP_N = 8                     # nombre de features à retenir au final
RANDOM_STATE = 42
N_SAMPLES_FOR_SELECTION = 200_000  # sous-échantillon pour accélérer le calcul

# Les 15 features candidates, DANS LE MÊME ORDRE que le contrat
FEATURE_MAPPING = {
    "Protocol Type": "feat_01_protocol",
    "Number": "feat_02_packet_count",
    "Tot sum": "feat_03_total_bytes",
    "AVG": "feat_04_packet_len_mean",
    "Std": "feat_05_packet_len_std",
    "Variance": "feat_06_packet_len_var",
    "Min": "feat_07_packet_len_min",
    "Max": "feat_08_packet_len_max",
    "Rate": "feat_09_flow_rate",
    "IAT": "feat_10_iat_mean",
    "syn_flag_number": "feat_11_syn_flag",
    "ack_flag_number": "feat_12_ack_flag",
    "fin_flag_number": "feat_13_fin_flag",
    "rst_flag_number": "feat_14_rst_flag",
    "psh_flag_number": "feat_15_psh_flag",
}
FEATURE_COLS_UNIFIED = list(FEATURE_MAPPING.values())


# ============================================================
# ÉTAPE 1 — Charger et préparer les 15 candidates
# ============================================================
def load_candidates(input_file, mapping, n_samples, random_state):
    print(f"[INFO] Chargement de {input_file} ...")
    df = pd.read_csv(input_file, usecols=list(mapping.keys()) + ["label_binary"])
    print(f"[INFO] {len(df)} lignes chargées")

    if len(df) > n_samples:
        df = df.groupby("label_binary", group_keys=False).sample(
            frac=n_samples / len(df), random_state=random_state
        )
        print(f"[INFO] Sous-échantillonné à {len(df)} lignes pour accélérer le calcul")

    df = df.rename(columns=mapping)

    # Nettoyage des valeurs manquantes / infinies
    feature_cols = list(mapping.values())
    n_before = len(df)

    nan_counts = df[feature_cols].isna().sum()
    if nan_counts.sum() > 0:
        print(f"[INFO] Valeurs manquantes détectées par colonne :")
        print(nan_counts[nan_counts > 0])

    # Rate/IAT peuvent produire des divisions par zéro -> +/- inf
    import numpy as np
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)

    df = df.dropna(subset=feature_cols)
    n_after = len(df)
    if n_after < n_before:
        print(f"[INFO] {n_before - n_after} lignes supprimées (NaN/inf) sur {n_before} "
              f"({(n_before - n_after) / n_before * 100:.2f}%)")

    return df


# ============================================================
# ÉTAPE 2 — FILTER : information mutuelle
# ============================================================
def compute_mutual_information(df, feature_cols, random_state):
    print("\n=== FILTER : Information mutuelle ===")
    X = df[feature_cols].values
    y = df["label_binary"].values

    mi_scores = mutual_info_classif(X, y, random_state=random_state)
    mi_series = pd.Series(mi_scores, index=feature_cols).sort_values(ascending=False)

    print(mi_series)
    return mi_series


# ============================================================
# ÉTAPE 3 — EMBEDDED : importance Random Forest
# ============================================================
def compute_rf_importance(df, feature_cols, random_state):
    print("\n=== EMBEDDED : Importance Random Forest ===")
    X = df[feature_cols].values
    y = df["label_binary"].values

    rf = RandomForestClassifier(
        n_estimators=100, max_depth=10, random_state=random_state, n_jobs=-1
    )
    rf.fit(X, y)

    rf_series = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(
        ascending=False
    )
    print(rf_series)
    return rf_series


# ============================================================
# ÉTAPE 4 — Combiner les deux scores (rang moyen)
# ============================================================
def combine_scores(mi_series, rf_series, feature_cols, top_n):
    print("\n=== COMBINAISON DES SCORES (rang moyen filter + embedded) ===")

    mi_rank = mi_series.rank(ascending=False)
    rf_rank = rf_series.rank(ascending=False)

    combined = pd.DataFrame({
        "feature": feature_cols,
        "mutual_info_score": [mi_series[f] for f in feature_cols],
        "mutual_info_rank": [mi_rank[f] for f in feature_cols],
        "rf_importance_score": [rf_series[f] for f in feature_cols],
        "rf_importance_rank": [rf_rank[f] for f in feature_cols],
    })
    combined["avg_rank"] = (combined["mutual_info_rank"] + combined["rf_importance_rank"]) / 2
    combined = combined.sort_values("avg_rank")

    print(combined.to_string(index=False))

    top_features = combined.head(top_n)["feature"].tolist()
    print(f"\n[INFO] Top {top_n} features retenues (IoT) : {top_features}")

    return combined, top_features


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    df = load_candidates(INPUT_FILE, FEATURE_MAPPING, N_SAMPLES_FOR_SELECTION, RANDOM_STATE)

    mi_series = compute_mutual_information(df, FEATURE_COLS_UNIFIED, RANDOM_STATE)
    rf_series = compute_rf_importance(df, FEATURE_COLS_UNIFIED, RANDOM_STATE)

    combined, top_features = combine_scores(mi_series, rf_series, FEATURE_COLS_UNIFIED, TOP_N)

    combined.to_csv(OUTPUT_SCORES_FILE, index=False)
    print(f"\n[OK] Tableau de scores sauvegardé : {OUTPUT_SCORES_FILE}")
    print("[OK] Envoie ce fichier à Malak pour qu'elle fasse le même calcul "
          "sur son côté (mêmes 15 candidates), afin de comparer et retenir "
          "l'intersection des features importantes des deux domaines.")