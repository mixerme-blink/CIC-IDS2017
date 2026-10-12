"""
Comparaison de 3 architectures GNN (GraphSAGE, GCN, GAT) sur le graphe k-NN IoT.
PFA - Hybrid Feature Selection and GNN for Intrusion Detection

But : (1) voir si une architecture différente de GraphSAGE fait mieux face
au baseline Random Forest, (2) fournir une comparaison inter-architectures
GNN pour le rapport, indépendamment du résultat face au RF.

Les 3 architectures utilisent le même early stopping (meilleur F1 sur le
test, poids restaurés à la fin) que le script GraphSAGE précédent, pour une
comparaison strictement équitable.

Utilisation :
    python compare_gnn_architectures_iot.py
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GCNConv, GATConv
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)

# ============================================================
# CONFIGURATION
# ============================================================
GRAPH_FILE = r"D:\CICIoT2023\graph_iot_knn.pt"

HIDDEN_DIM = 32
NUM_EPOCHS = 100
LEARNING_RATE = 0.01
WEIGHT_DECAY = 5e-4
RANDOM_STATE = 42
GAT_HEADS = 4  # nombre de têtes d'attention pour GAT

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# ARCHITECTURES
# ============================================================
class GraphSAGE_IDS(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes=2):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.classifier = torch.nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.3, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        return self.classifier(x)


class GCN_IDS(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes=2):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.classifier = torch.nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.3, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        return self.classifier(x)


class GAT_IDS(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes=2, heads=4):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=0.3)
        self.conv2 = GATConv(hidden_channels * heads, hidden_channels, heads=1, dropout=0.3)
        self.classifier = torch.nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index):
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.3, training=self.training)
        x = F.elu(self.conv2(x, edge_index))
        return self.classifier(x)


# ============================================================
# ENTRAÎNEMENT / ÉVALUATION (partagés entre les 3 architectures)
# ============================================================
def train_step(model, data, optimizer, class_weights):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask], weight=class_weights)
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def evaluate(model, data, mask):
    model.eval()
    out = model(data.x, data.edge_index)
    preds = out[mask].argmax(dim=1).cpu().numpy()
    labels = data.y[mask].cpu().numpy()
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
        "fpr": fp / (fp + tn) if (fp + tn) > 0 else 0.0,
    }


def train_with_early_stopping(model, data, class_weights, num_epochs, lr, weight_decay):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_f1, best_epoch, best_state = -1.0, -1, None

    for epoch in range(1, num_epochs + 1):
        train_step(model, data, optimizer, class_weights)
        val_metrics = evaluate(model, data, data.val_mask)
        if val_metrics["f1"] > best_f1:
            best_f1, best_epoch = val_metrics["f1"], epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    final_metrics = evaluate(model, data, data.test_mask)
    return final_metrics, best_epoch


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    torch.manual_seed(RANDOM_STATE)

    print(f"[INFO] Device utilisé : {DEVICE}")
    data = torch.load(GRAPH_FILE, weights_only=False).to(DEVICE)
    print(f"[INFO] {data.num_nodes} nœuds, {data.num_edges} arêtes, {data.x.shape[1]} features")

    n_benign = (data.y[data.train_mask] == 0).sum().item()
    n_attack = (data.y[data.train_mask] == 1).sum().item()
    class_weights = torch.tensor(
        [len(data.y[data.train_mask]) / (2 * n_benign),
         len(data.y[data.train_mask]) / (2 * n_attack)],
        dtype=torch.float32,
    ).to(DEVICE)

    architectures = {
        "GraphSAGE": GraphSAGE_IDS(data.x.shape[1], HIDDEN_DIM).to(DEVICE),
        "GCN": GCN_IDS(data.x.shape[1], HIDDEN_DIM).to(DEVICE),
        "GAT": GAT_IDS(data.x.shape[1], HIDDEN_DIM, heads=GAT_HEADS).to(DEVICE),
    }

    results = {}
    for name, model in architectures.items():
        print(f"\n=== Entraînement {name} ({NUM_EPOCHS} epochs, early stopping) ===")
        metrics, best_epoch = train_with_early_stopping(
            model, data, class_weights, NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY
        )
        results[name] = metrics
        print(f"[INFO] {name} — meilleure epoch {best_epoch} :")
        for k, v in metrics.items():
            print(f"    {k:12s} : {v:.4f}")

    print("\n" + "=" * 70)
    print("=== TABLEAU COMPARATIF FINAL ===")
    print("=" * 70)
    header = f"{'Architecture':<12}" + "".join(f"{m:>12}" for m in
                                                 ["accuracy", "precision", "recall", "f1", "fpr"])
    print(header)
    for name, metrics in results.items():
        row = f"{name:<12}" + "".join(f"{metrics[m]:>12.4f}" for m in
                                       ["accuracy", "precision", "recall", "f1", "fpr"])
        print(row)

    print("\n[INFO] Compare ce tableau à Baseline_no_graph_iot.py (Random Forest) "
          "pour la conclusion finale de l'étude d'ablation.")