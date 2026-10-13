"""
Entraînement d'un GNN (GCN — Graph Convolutional Network) sur le graphe
k-NN IoT.
PFA - Hybrid Feature Selection and GNN for Intrusion Detection

Miroir de Train_gnn_iot.py (GraphSAGE) et Train_gat_iot.py (GAT) : même
graphe, même split, mêmes hyperparamètres d'entraînement, MÊME early
stopping (poids restaurés à la meilleure epoch selon le F1 test) -- seule
l'architecture change (GCNConv).

GCN est le plus simple des trois : normalisation symétrique du graphe
(degré des noeuds) + moyenne des voisins, sans mécanisme d'attention ni
d'échantillonnage. Sert de point de comparaison "minimal" pour situer
GAT et GraphSAGE.

Utilisation :
    python Train_gcn_iot.py
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)

# ============================================================
# CONFIGURATION -- identique à Train_gnn_iot.py / Train_gat_iot.py
# ============================================================
GRAPH_FILE = r"D:\CICIoT2023\graph_iot_knn_blocks.pt"
MODEL_OUTPUT_FILE = r"D:\CICIoT2023\model_iot_gcn_blocks.pt"

HIDDEN_DIM = 32
NUM_EPOCHS = 100
LEARNING_RATE = 0.01
WEIGHT_DECAY = 5e-4
RANDOM_STATE = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# ARCHITECTURE — GCN, même structure générale que GraphSAGE_IDS / GAT_IDS
# ============================================================
class GCN_IDS(torch.nn.Module):
    """Architecture GCN pour la détection d'intrusion.
    Miroir de GraphSAGE_IDS et GAT_IDS : 2 couches de convolution de
    graphe + classifieur linéaire, mêmes dimensions cachées."""

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
        out = self.classifier(x)
        return out


# ============================================================
# ÉTAPE 1 — Entraînement
# ============================================================
def train(model, data, optimizer, class_weights):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(
        out[data.train_mask], data.y[data.train_mask], weight=class_weights
    )
    loss.backward()
    optimizer.step()
    return loss.item()


# ============================================================
# ÉTAPE 2 — Évaluation
# ============================================================
@torch.no_grad()
def evaluate(model, data, mask):
    model.eval()
    out = model(data.x, data.edge_index)
    preds = out[mask].argmax(dim=1).cpu().numpy()
    labels = data.y[mask].cpu().numpy()

    acc = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, zero_division=0)
    recall = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)

    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return {
        "accuracy": acc, "precision": precision, "recall": recall,
        "f1": f1, "fpr": fpr,
        "balanced_acc": (recall + (1 - fpr)) / 2,
    }


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    torch.manual_seed(RANDOM_STATE)

    print(f"[INFO] Device utilisé : {DEVICE}")
    print(f"[INFO] Chargement du graphe : {GRAPH_FILE}")
    data = torch.load(GRAPH_FILE, weights_only=False).to(DEVICE)

    print(f"[INFO] {data.num_nodes} nœuds, {data.num_edges} arêtes, "
          f"{data.x.shape[1]} features")

    n_benign = (data.y[data.train_mask] == 0).sum().item()
    n_attack = (data.y[data.train_mask] == 1).sum().item()
    class_weights = torch.tensor(
        [len(data.y[data.train_mask]) / (2 * n_benign),
         len(data.y[data.train_mask]) / (2 * n_attack)],
        dtype=torch.float32,
    ).to(DEVICE)
    print(f"[INFO] Pondération de classe (déséquilibre) : "
          f"Benign={class_weights[0]:.2f}, Attack={class_weights[1]:.2f}")

    model = GCN_IDS(
        in_channels=data.x.shape[1], hidden_channels=HIDDEN_DIM
    ).to(DEVICE)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    print(f"\n=== ENTRAÎNEMENT ({NUM_EPOCHS} epochs, early stopping sur VALIDATION, critère = balanced accuracy) ===")
    best_f1 = -1.0
    best_epoch = -1
    best_state_dict = None

    for epoch in range(1, NUM_EPOCHS + 1):
        loss = train(model, data, optimizer, class_weights)
        val_metrics = evaluate(model, data, data.val_mask)

        if val_metrics["balanced_acc"] > best_f1:
            best_f1 = val_metrics["balanced_acc"]
            best_epoch = epoch
            best_state_dict = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | Loss: {loss:.4f} | "
                  f"Val Acc: {val_metrics['accuracy']:.4f} | "
                  f"Val F1: {val_metrics['f1']:.4f} | Val FPR: {val_metrics['fpr']:.4f} | "
                  f"Val BalAcc: {val_metrics['balanced_acc']:.4f}")

    print(f"\n[INFO] Meilleure epoch (sur VALIDATION) : {best_epoch} (balanced accuracy={best_f1:.4f}) "
          f"-> poids restaurés à ce point")
    model.load_state_dict(best_state_dict)

    print("\n=== ÉVALUATION FINALE (meilleure epoch, TEST jamais vu avant) ===")
    final_metrics = evaluate(model, data, data.test_mask)
    for k, v in final_metrics.items():
        print(f"  {k:12s} : {v:.4f}")

    torch.save(model.state_dict(), MODEL_OUTPUT_FILE)
    print(f"\n[OK] Modèle sauvegardé (meilleure epoch {best_epoch}) : {MODEL_OUTPUT_FILE}")
    print("[OK] Compare ces résultats à GraphSAGE, GAT et RF pour la "
          "comparaison à 3 modèles du PFA.")