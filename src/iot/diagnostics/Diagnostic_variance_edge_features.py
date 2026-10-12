"""
Diagnostic : pourquoi le graphe k-NN a-t-il si peu changé après l'ajout
des 7 dimensions supplémentaires (Protocol Type, Std, Variance, flags) ?

Hypothèse : ces colonnes ont très peu de variance sur cet échantillon
(quasi-constantes), donc elles ne changent presque rien au calcul de
distance k-NN, même après normalisation Min-Max.

Utilisation :
    python diagnostic_variance_edge_features.py
"""

import pandas as pd

INPUT_FILE = r"D:\CICIoT2023\ciciot2023_sample.csv"
N_SAMPLE = 50_000
RANDOM_STATE = 42

EXTRA_FEATURES = [
    "Protocol Type", "Std", "Variance",
    "syn_flag_number", "fin_flag_number", "rst_flag_number", "psh_flag_number",
]
BASE_FEATURES = [
    "Number", "Tot sum", "IAT", "AVG", "Rate", "Max", "Min", "ack_flag_number",
]

if __name__ == "__main__":
    df = pd.read_csv(INPUT_FILE, usecols=EXTRA_FEATURES + BASE_FEATURES + ["label_binary"])
    df = df.groupby("label_binary", group_keys=False).sample(
        frac=N_SAMPLE / len(df), random_state=RANDOM_STATE
    )

    print("=== Coefficient de variation (std / mean) — plus c'est bas, plus la colonne est 'plate' ===")
    print("\n7 features AJOUTÉES pour le graphe :")
    for col in EXTRA_FEATURES:
        mean = df[col].mean()
        std = df[col].std()
        cv = std / mean if mean != 0 else float("nan")
        print(f"  {col:20s} : mean={mean:10.4f}  std={std:10.4f}  CV={cv:.4f}  "
              f"valeurs_uniques={df[col].nunique()}")

    print("\n8 features DE BASE (déjà dans le GNN) :")
    for col in BASE_FEATURES:
        mean = df[col].mean()
        std = df[col].std()
        cv = std / mean if mean != 0 else float("nan")
        print(f"  {col:20s} : mean={mean:10.4f}  std={std:10.4f}  CV={cv:.4f}  "
              f"valeurs_uniques={df[col].nunique()}")