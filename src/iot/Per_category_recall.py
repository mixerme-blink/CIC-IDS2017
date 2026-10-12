"""
Analyse par catégorie d'attaque : taux de détection de chaque modèle.
PFA - Hybrid Feature Selection and GNN for Intrusion Detection

Pour chaque catégorie présente dans le TEST (34 au total), on calcule :
  - pour une catégorie d'attaque : % de flows correctement détectés comme attaque
  - pour Benign_Final           : % de flows correctement reconnus comme normaux

Modèles comparés : Random Forest (sans graphe, réentraîné ici), GraphSAGE,
GCN, GAT (poids déjà sauvegardés, évalués sur le graphe).

PRÉREQUIS : avoir relancé build_knn_graph_iot_blocks.py (version qui
enregistre les catégories) -> graph_iot_knn_blocks.pt + le fichier
graph_iot_knn_blocks_categories.json.

Utilisation :
    python per_category_recall.py
"""

import json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GCNConv, GATConv
from sklearn.ensemble import RandomForestClassifier

# ============================================================
# CONFIGURATION
# ============================================================
GRAPH_FILE = r"D:\CICIoT2023\graph_iot_knn_blocks.pt"
CATEGORIES_FILE = r"D:\CICIoT2023\graph_iot_knn_blocks_categories.json"
MODEL_FILES = {
    "GraphSAGE": r"D:\CICIoT2023\model_iot_gnn_blocks.pt",
    "GCN": r"D:\CICIoT2023\model_iot_gcn_blocks.pt",
    "GAT": r"D:\CICIoT2023\model_iot_gat_blocks.pt",
}
OUTPUT_CSV = r"D:\CICIoT2023\per_category_recall.csv"
HIDDEN_DIM = 32
RANDOM_STATE = 42
BENIGN_NAMES = {"benign_final", "benign"}


# ============================================================
# ARCHITECTURES (identiques à celles des scripts d'entraînement)
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
        return self.classifier(x)


class GCN_IDS(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes=2):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.classifier = torch.nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        return self.classifier(x)


class GAT_IDS(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes=2, heads=4, dropout=0.3):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        self.conv2 = GATConv(hidden_channels * heads, hidden_channels, heads=1,
                              concat=False, dropout=dropout)
        self.classifier = torch.nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.elu(x)
        return self.classifier(x)


ARCHS = {"GraphSAGE": GraphSAGE_IDS, "GCN": GCN_IDS, "GAT": GAT_IDS}


@torch.no_grad()
def predict_gnn(name, path, data):
    model = ARCHS[name](data.x.shape[1], HIDDEN_DIM)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    out = model(data.x, data.edge_index)
    return out.argmax(dim=1).numpy()


if __name__ == "__main__":
    print(f"[INFO] Chargement du graphe : {GRAPH_FILE}")
    data = torch.load(GRAPH_FILE, weights_only=False)
    if not hasattr(data, "category"):
        raise SystemExit("Le graphe ne contient pas les catégories : relance d'abord "
                         "build_knn_graph_iot_blocks.py (version mise à jour).")
    with open(CATEGORIES_FILE, encoding="utf-8") as f:
        category_names = json.load(f)

    test_mask = data.test_mask.numpy()
    y_all = data.y.numpy()
    cat_all = data.category.numpy()

    # Prédictions de chaque modèle sur TOUS les nœuds, puis on garde le test
    preds = {}

    print("[INFO] Random Forest (sans graphe)...")
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, class_weight="balanced",
                                random_state=RANDOM_STATE, n_jobs=-1)
    train_mask = data.train_mask.numpy()
    rf.fit(data.x.numpy()[train_mask], y_all[train_mask])
    preds["Random Forest"] = rf.predict(data.x.numpy())

    for name, path in MODEL_FILES.items():
        print(f"[INFO] {name}...")
        preds[name] = predict_gnn(name, path, data)

    # Table par catégorie
    rows = []
    for code, cat in enumerate(category_names):
        sel = test_mask & (cat_all == code)
        n = int(sel.sum())
        if n == 0:
            continue
        is_benign = cat.strip().lower() in BENIGN_NAMES
        row = {"categorie": cat, "type": "Benign" if is_benign else "Attack", "n_test": n}
        for model_name, p in preds.items():
            correct = (p[sel] == 0) if is_benign else (p[sel] == 1)
            row[model_name] = round(100 * correct.mean(), 1)
            row[f"ratés_{model_name}"] = int((~correct).sum())
        rows.append(row)

    df = pd.DataFrame(rows)
    model_cols = list(preds.keys())
    df["moyenne"] = df[model_cols].mean(axis=1).round(1)
    df = df.sort_values("moyenne").reset_index(drop=True)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_rows", 100)
    print("\n=== TAUX DE DÉTECTION PAR CATÉGORIE (% correct), du plus raté au mieux détecté ===")
    print(df[["categorie", "type", "n_test"] + model_cols + ["moyenne"]].to_string(index=False))

    # Où se concentrent les attaques ratées ?
    attacks = df[df["type"] == "Attack"].copy()
    print("\n=== OÙ SE CONCENTRENT LES ATTAQUES RATÉES (GraphSAGE) ===")
    total_missed = attacks["ratés_GraphSAGE"].sum()
    top = attacks.sort_values("ratés_GraphSAGE", ascending=False).head(8)
    for _, r in top.iterrows():
        part = 100 * r["ratés_GraphSAGE"] / total_missed if total_missed else 0
        print(f"  {r['categorie']:28s} {int(r['ratés_GraphSAGE']):5d} ratées "
              f"({part:4.1f}% de toutes les attaques ratées)")
    print(f"  Total attaques ratées : {int(total_missed)}")

    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n[OK] Tableau complet sauvegardé : {OUTPUT_CSV}")