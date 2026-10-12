"""
Sélection hybride de features (filter + embedded) — ANALYSE COMPLÉMENTAIRE
sur les 39 colonnes BRUTES du dataset IoT (CICIoT2023).

⚠️ IMPORTANT — CE SCRIPT NE REMPLACE PAS hybrid_feature_selection_iot.py :
- Le résultat ici sert UNIQUEMENT à une analyse in-domain (performance
  maximale possible côté IoT seul, discussion sur les features spécifiques
  à l'IoT non-transférables au cloud).
- Il NE PEUT PAS être utilisé pour construire le vecteur du GNN partagé,
  car des features comme HTTP, DNS, ARP, ICMP, TTL n'existent pas côté
  cloud (CICIDS2017) — les inclure casserait la compatibilité cross-domain.
- Pour le pipeline partagé, utiliser hybrid_feature_selection_iot.py
  (restreint aux 15 features harmonisées avec Malak).

Utilité de cette analyse complémentaire :
1. Voir si des features spécifiques à l'IoT (ARP, ICMP...) sont très
   discriminantes pour certaines attaques (ex. ARP pour MITM-ArpSpoofing)
2. Comparer la performance "GNN partagé (8 features)" vs "GNN IoT pur
   (features optimales locales)" dans le rapport — quantifier le coût de
   l'harmonisation cross-domain
3. Justifier scientifiquement pourquoi certaines features sont absentes
   du set harmonisé

Utilisation :
    python hybrid_feature_selection_iot_39raw.py
"""

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier

# ============================================================
# CONFIGURATION
# ============================================================
INPUT_FILE = r"D:\CICIoT2023\ciciot2023_sample.csv"
OUTPUT_SCORES_FILE = r"D:\CICIoT2023\feature_selection_scores_iot_39raw.csv"
TOP_N = 15                    # on garde plus large ici, c'est une analyse exploratoire
RANDOM_STATE = 42
N_SAMPLES_FOR_SELECTION = 200_000

# Les 39 colonnes brutes du dataset (toutes, sauf le label)
RAW_FEATURE_COLS = [
    "Header_Length", "Protocol Type", "Time_To_Live", "Rate",
    "fin_flag_number", "syn_flag_number", "rst_flag_number", "psh_flag_number",
    "ack_flag_number", "ece_flag_number", "cwr_flag_number",
    "ack_count", "syn_count", "fin_count", "rst_count",
    "HTTP", "HTTPS", "DNS", "Telnet", "SMTP", "SSH", "IRC",
    "TCP", "UDP", "DHCP", "ARP", "ICMP", "IGMP", "IPv", "LLC",
    "Tot sum", "Min", "Max", "AVG", "Std", "Tot size", "IAT", "Number", "Variance",
]

# Marque les features qui font DÉJÀ partie du pool harmonisé (15) —
# utile pour repérer facilement dans les résultats ce qui est nouveau
HARMONIZED_POOL = {
    "Protocol Type", "Number", "Tot sum", "AVG", "Std", "Variance", "Min", "Max",
    "Rate", "IAT", "syn_flag_number", "ack_flag_number", "fin_flag_number",
    "rst_flag_number", "psh_flag_number",
}


# ============================================================
# ÉTAPE 1 — Charger, sous-échantillonner, nettoyer
# ============================================================
def load_and_clean(input_file, feature_cols, n_samples, random_state):
    print(f"[INFO] Chargement de {input_file} ...")
    df = pd.read_csv(input_file, usecols=feature_cols + ["label_binary"])
    print(f"[INFO] {len(df)} lignes chargées")

    if len(df) > n_samples:
        df = df.groupby("label_binary", group_keys=False).sample(
            frac=n_samples / len(df), random_state=random_state
        )
        print(f"[INFO] Sous-échantillonné à {len(df)} lignes")

    n_before = len(df)
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    nan_counts = df[feature_cols].isna().sum()
    if nan_counts.sum() > 0:
        print(f"[INFO] Valeurs manquantes/infinies par colonne :")
        print(nan_counts[nan_counts > 0])
    df = df.dropna(subset=feature_cols)
    n_after = len(df)
    if n_after < n_before:
        print(f"[INFO] {n_before - n_after} lignes supprimées "
              f"({(n_before - n_after) / n_before * 100:.2f}%)")

    return df


# ============================================================
# ÉTAPE 2 — FILTER : information mutuelle
# ============================================================
def compute_mutual_information(df, feature_cols, random_state):
    print("\n=== FILTER : Information mutuelle (39 features brutes) ===")
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
    print("\n=== EMBEDDED : Importance Random Forest (39 features brutes) ===")
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
# ÉTAPE 4 — Combiner et annoter (nouveau vs déjà dans le pool harmonisé)
# ============================================================
def combine_and_annotate(mi_series, rf_series, feature_cols, top_n, harmonized_pool):
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
    combined["deja_dans_pool_harmonise"] = combined["feature"].isin(harmonized_pool)
    combined = combined.sort_values("avg_rank")

    print("\n=== CLASSEMENT COMPLET (39 features) ===")
    print(combined.to_string(index=False))

    nouvelles_features_top = combined[
        ~combined["deja_dans_pool_harmonise"]
    ].head(5)
    print(f"\n[INFO] Top 5 features IoT-spécifiques (absentes du pool harmonisé, "
          f"donc non-transférables au GNN partagé) :")
    print(nouvelles_features_top[["feature", "avg_rank"]].to_string(index=False))

    top_features = combined.head(top_n)["feature"].tolist()
    print(f"\n[INFO] Top {top_n} toutes features confondues : {top_features}")

    return combined


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    df = load_and_clean(INPUT_FILE, RAW_FEATURE_COLS, N_SAMPLES_FOR_SELECTION, RANDOM_STATE)

    mi_series = compute_mutual_information(df, RAW_FEATURE_COLS, RANDOM_STATE)
    rf_series = compute_rf_importance(df, RAW_FEATURE_COLS, RANDOM_STATE)

    combined = combine_and_annotate(
        mi_series, rf_series, RAW_FEATURE_COLS, TOP_N, HARMONIZED_POOL
    )

    combined.to_csv(OUTPUT_SCORES_FILE, index=False)
    print(f"\n[OK] Résultats sauvegardés : {OUTPUT_SCORES_FILE}")
    print("[OK] Rappel : ces résultats servent à l'analyse in-domain / discussion "
          "dans le rapport, PAS au vecteur du GNN partagé (voir "
          "hybrid_feature_selection_iot.py pour ça).")