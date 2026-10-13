"""
Diagnostic : les Attack ratees (Faux Negatifs) ressemblent-elles
naturellement aux Benign en termes de features, ou est-ce un
comportement bizarre / artefact du modele ?

Principe : on reentraine le RF (le meilleur modele jusqu'ici, donc la
reference la plus fiable) sur le graphe actuel, on identifie les Attack
mal classees (FN), puis on compare la moyenne de leurs 8 features a la
moyenne des Attack bien classees ET a la moyenne des vrais Benign.

Si les FN sont "entre les deux" ou plus proches des Benign -> ce sont
des attaques discretes/low-rate qui imitent du trafic normal, ambiguite
naturelle du probleme (pas un bug, a documenter comme limite honnete).

Si les FN ne ressemblent pas particulierement aux Benign -> le probleme
vient d'ailleurs (mauvais reglage RF, features insuffisantes pour CE
sous-ensemble d'attaques, etc.)

Utilisation :
    python diagnostic_fn_attack.py
"""

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier

GRAPH_FILE = r"D:\CICIoT2023\graph_iot_knn.pt"
RANDOM_STATE = 42

NODE_FEATURE_NAMES = [
    "packet_count", "total_bytes", "iat_mean", "packet_len_mean",
    "flow_rate", "packet_len_max", "packet_len_min", "ack_flag",
]

if __name__ == "__main__":
    print(f"[INFO] Chargement du graphe : {GRAPH_FILE}")
    data = torch.load(GRAPH_FILE, weights_only=False)

    X_train = data.x[data.train_mask].numpy()
    y_train = data.y[data.train_mask].numpy()
    X_test = data.x[data.test_mask].numpy()
    y_test = data.y[data.test_mask].numpy()

    rf = RandomForestClassifier(
        n_estimators=100, max_depth=10, class_weight="balanced",
        random_state=RANDOM_STATE, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    preds = rf.predict(X_test)

    is_fn = (y_test == 1) & (preds == 0)     # vraie Attack, prédite Benign
    is_tp = (y_test == 1) & (preds == 1)     # vraie Attack, bien classée
    is_true_benign = (y_test == 0)

    print(f"\n[INFO] Attack ratées (FN)       : {is_fn.sum()}")
    print(f"[INFO] Attack bien classées (TP) : {is_tp.sum()}")
    print(f"[INFO] Vrais Benign (test)       : {is_true_benign.sum()}")

    mean_fn = X_test[is_fn].mean(axis=0)
    mean_tp = X_test[is_tp].mean(axis=0)
    mean_benign = X_test[is_true_benign].mean(axis=0)

    print("\n=== COMPARAISON DES FEATURES MOYENNES (normalisées 0-1) ===")
    print(f"{'Feature':20s} {'FN (Attack ratée)':>18s} {'TP (Attack ok)':>16s} {'Benign':>10s}")
    for i, name in enumerate(NODE_FEATURE_NAMES):
        print(f"{name:20s} {mean_fn[i]:18.4f} {mean_tp[i]:16.4f} {mean_benign[i]:10.4f}")

    # Distance euclidienne moyenne FN->Benign vs FN->TP(Attack normales)
    dist_fn_to_benign = np.linalg.norm(mean_fn - mean_benign)
    dist_fn_to_tp = np.linalg.norm(mean_fn - mean_tp)

    print(f"\n[INFO] Distance (features moyennes) FN <-> Benign : {dist_fn_to_benign:.4f}")
    print(f"[INFO] Distance (features moyennes) FN <-> Attack normales (TP) : {dist_fn_to_tp:.4f}")

    print("\n=== VERDICT ===")
    if dist_fn_to_benign < dist_fn_to_tp:
        print("Les Attack ratées ressemblent en moyenne PLUS aux Benign qu'aux")
        print("Attack normalement détectées.")
        print("-> Ambiguïté naturelle : ce sont probablement des attaques")
        print("   discrètes / low-rate qui imitent le trafic normal -- une")
        print("   limite honnête du dataset/features, pas un artefact du")
        print("   modèle. A documenter tel quel dans le rapport.")
    else:
        print("Les Attack ratées ne ressemblent pas particulièrement aux")
        print("Benign -- elles diffèrent aussi des Attack normalement")
        print("détectées. Le problème n'est probablement pas une ambiguïté")
        print("naturelle mais un sous-ensemble d'attaques mal représenté")
        print("dans le train, ou un réglage RF à ajuster (max_depth, etc.)")