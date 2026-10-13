"""
Baseline SANS graphe (Random Forest) — étude d'ablation.
PFA - Hybrid Feature Selection and GNN for Intrusion Detection

But : vérifier si le GNN apporte une vraie valeur ajoutée par rapport à un
modèle classique entraîné sur EXACTEMENT les mêmes 8 features, sans aucune
notion de graphe/voisinage. Si les deux scores sont quasi identiques, cela
indique que la performance vient surtout de la qualité des features (déjà
bien séparables), pas de la structure du graphe elle-même.

Utilise le même graphe .pt en entrée pour garantir un split train/test
identique à celui du GNN (comparaison strictement équitable).

Utilisation :
    python baseline_no_graph_iot.py
"""

import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)

GRAPH_FILE = r"D:\CICIoT2023\graph_iot_knn_blocks.pt"
RANDOM_STATE = 42

if __name__ == "__main__":
    print(f"[INFO] Chargement de {GRAPH_FILE} (mêmes données que le GNN)...")
    data = torch.load(GRAPH_FILE, weights_only=False)

    X_train = data.x[data.train_mask].numpy()
    y_train = data.y[data.train_mask].numpy()
    X_test = data.x[data.test_mask].numpy()
    y_test = data.y[data.test_mask].numpy()

    print(f"[INFO] Train : {len(X_train)} | Test : {len(X_test)}")
    print(f"[INFO] Répartition test - Benign: {(y_test == 0).sum()}, "
          f"Attack: {(y_test == 1).sum()}")

    print("\n=== ENTRAÎNEMENT Random Forest (SANS graphe) ===")
    rf = RandomForestClassifier(
        n_estimators=100, max_depth=10, class_weight="balanced",
        random_state=RANDOM_STATE, n_jobs=-1
    )
    rf.fit(X_train, y_train)

    preds = rf.predict(X_test)

    acc = accuracy_score(y_test, preds)
    precision = precision_score(y_test, preds, zero_division=0)
    recall = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_test, preds, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    print("\n=== RÉSULTATS BASELINE (SANS graphe) ===")
    print(f"  accuracy     : {acc:.4f}")
    print(f"  precision    : {precision:.4f}")
    print(f"  recall       : {recall:.4f}")
    print(f"  f1           : {f1:.4f}")
    print(f"  fpr          : {fpr:.4f}")
    print(f"  balanced_acc : {(recall + (1 - fpr)) / 2:.4f}")
    print(f"\n  Matrice de confusion : TN={tn}, FP={fp}, FN={fn}, TP={tp}")

    print("\n[INFO] Compare ces chiffres à ceux du GNN (train_gnn_iot.py) :")
    print("  - Si très proches -> le graphe n'apporte pas de valeur ajoutée")
    print("    claire ; à documenter honnêtement dans le rapport")
    print("  - Si le GNN fait nettement mieux -> bon argument en faveur")
    print("    de l'approche par graphe")