"""
Comparaison détaillée des erreurs sur la classe Benign entre le GNN
(GraphSAGE) et le Random Forest, sur le MÊME test set (15 000 flows réels,
jamais touchés par SMOTE).

But : savoir si les deux modèles se trompent sur les MÊMES flows Benign
(-> le graphe n'apporte rien de nouveau) ou sur des flows DIFFÉRENTS
(-> le graphe capture un signal complémentaire, bon argument pour le rapport).

Utilisation :
    python compare_errors_gnn_rf.py

Prérequis :
    - graph_iot_knn.pt doit être le graphe SMOTE (56 860 nœuds) déjà utilisé
      pour entraîner model_iot.pt
    - model_iot.pt doit exister (généré par train_gnn_iot.py)

⚠️ Le Random Forest est ré-entraîné ici (car baseline_no_graph_iot.py n'a
pas été partagé) avec class_weight="balanced" et random_state=42, sur les
mêmes NODE_FEATURES (x) et le même train_mask/test_mask que le GNN. Si ton
vrai script RF utilise d'autres hyperparamètres (n_estimators, max_depth,
etc.), remplace-les ci-dessous pour retrouver EXACTEMENT les mêmes chiffres
que ton run précédent (TN=346, FP=6, FN=210, TP=14438).
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

# ============================================================
# CONFIGURATION — adapte si besoin
# ============================================================
GRAPH_FILE = r"D:\CICIoT2023\graph_iot_knn.pt"
MODEL_FILE = r"D:\CICIoT2023\model_iot.pt"

HIDDEN_DIM = 32
RANDOM_STATE = 42

# Hyperparamètres RF — copiés depuis baseline_no_graph_iot.py pour
# retomber exactement sur les mêmes chiffres (TN=346, FP=6, FN=210, TP=14438)
RF_N_ESTIMATORS = 100
RF_MAX_DEPTH = 10
RF_CLASS_WEIGHT = "balanced"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# Architecture GNN — identique à train_gnn_iot.py
# ============================================================
class GraphSAGE_IDS(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes=2):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.classifier = torch.nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        out = self.classifier(x)
        return out


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    print(f"[INFO] Chargement du graphe : {GRAPH_FILE}")
    data = torch.load(GRAPH_FILE, weights_only=False).to(DEVICE)
    print(f"[INFO] {data.num_nodes} nœuds, {data.x.shape[1]} features")

    # ---- Prédictions GNN ----
    print(f"[INFO] Chargement du modèle GNN : {MODEL_FILE}")
    model = GraphSAGE_IDS(in_channels=data.x.shape[1], hidden_channels=HIDDEN_DIM).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_FILE, map_location=DEVICE, weights_only=True))
    model.eval()

    with torch.no_grad():
        out = model(data.x, data.edge_index)
        gnn_preds_all = out.argmax(dim=1).cpu().numpy()

    test_mask = data.test_mask.cpu().numpy()
    y_test = data.y.cpu().numpy()[test_mask]
    gnn_preds = gnn_preds_all[test_mask]

    # ---- Prédictions RF (ré-entraîné sur les mêmes features/masks) ----
    print("[INFO] Entraînement du Random Forest (mêmes NODE_FEATURES, même train/test)...")
    X = data.x.cpu().numpy()
    y = data.y.cpu().numpy()
    train_mask = data.train_mask.cpu().numpy()

    rf = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        class_weight=RF_CLASS_WEIGHT,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    rf.fit(X[train_mask], y[train_mask])
    rf_preds = rf.predict(X[test_mask])

    # ============================================================
    # Comparaison des erreurs sur la classe Benign (label 0)
    # ============================================================
    benign_idx = np.where(y_test == 0)[0]
    n_benign = len(benign_idx)
    print(f"\n[INFO] {n_benign} exemples Benign réels dans le test set")

    gnn_wrong = set(benign_idx[gnn_preds[benign_idx] != 0])
    rf_wrong = set(benign_idx[rf_preds[benign_idx] != 0])

    both_wrong = gnn_wrong & rf_wrong
    only_gnn_wrong = gnn_wrong - rf_wrong
    only_rf_wrong = rf_wrong - gnn_wrong

    print("\n=== MATRICES DE CONFUSION (test set, 15 000 flows) ===")
    print("GNN :")
    tn, fp, fn, tp = confusion_matrix(y_test, gnn_preds, labels=[0, 1]).ravel()
    print(f"  TN={tn}, FP={fp}, FN={fn}, TP={tp}")

    print("RF  :")
    tn, fp, fn, tp = confusion_matrix(y_test, rf_preds, labels=[0, 1]).ravel()
    print(f"  TN={tn}, FP={fp}, FN={fn}, TP={tp}")

    print(f"\n=== ERREURS SUR LES {n_benign} EXEMPLES BENIGN ===")
    print(f"  Erreurs GNN uniquement       : {len(only_gnn_wrong):4d}")
    print(f"  Erreurs RF uniquement        : {len(only_rf_wrong):4d}")
    print(f"  Erreurs communes (les deux)  : {len(both_wrong):4d}")
    print(f"  Total erreurs GNN            : {len(gnn_wrong):4d}")
    print(f"  Total erreurs RF             : {len(rf_wrong):4d}")

    # ============================================================
    # Interprétation automatique
    # ============================================================
    print("\n=== INTERPRÉTATION ===")
    if len(gnn_wrong) == 0 and len(rf_wrong) == 0:
        print("Aucun des deux modèles ne se trompe sur Benign -> rien à comparer ici.")
    else:
        overlap_ratio = len(both_wrong) / len(gnn_wrong | rf_wrong) if (gnn_wrong | rf_wrong) else 0
        print(f"Recouvrement des erreurs : {overlap_ratio:.1%}")
        if overlap_ratio > 0.7:
            print("-> Les deux modèles se trompent surtout sur LES MÊMES flows.")
            print("   Le graphe ne semble PAS capturer un signal différent du RF")
            print("   sur ces cas difficiles -- à documenter comme limite.")
        else:
            print("-> Les erreurs sont largement DIFFÉRENTES entre les deux modèles.")
            print("   Le graphe capture potentiellement un signal complémentaire")
            print("   au RF -- bon argument pour le rapport, à illustrer avec")
            print("   quelques exemples concrets de flows dans only_gnn_wrong / only_rf_wrong.")