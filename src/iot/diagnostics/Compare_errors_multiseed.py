"""
Vérifie si le faible recouvrement d'erreurs Benign entre GNN et RF
(observé sur un seul run : 9.1%) est un effet réel et stable, ou juste
du bruit lié à l'initialisation aléatoire d'un seul entraînement.

⚠️ Limite importante : le split train/test est déjà figé DANS le fichier
graph_iot_knn.pt (construit une seule fois avec RANDOM_STATE=42). Ce script
fait donc varier uniquement :
  - l'initialisation des poids + l'ordre stochastique de l'entraînement GNN
  - le random_state du Random Forest
...mais PAS le split train/test lui-même, qui reste identique à chaque run.
Résultat : ce script mesure la stabilité du signal face au hasard de
l'entraînement, pas face au hasard du split. C'est un test plus rapide
mais partiel -- à mentionner comme limite dans le rapport si tu utilises
ces résultats.

Réentraîne le GNN au complet (100 epochs) pour chaque seed -> peut prendre
plusieurs minutes sur CPU selon le nombre de seeds (N_SEEDS).

Utilisation :
    python compare_errors_multiseed.py
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

# ============================================================
# CONFIGURATION
# ============================================================
GRAPH_FILE = r"D:\CICIoT2023\graph_iot_knn.pt"

N_SEEDS = 10
SEEDS = list(range(N_SEEDS))

HIDDEN_DIM = 32
NUM_EPOCHS = 100
LEARNING_RATE = 0.01
WEIGHT_DECAY = 5e-4

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


def train_gnn_one_seed(data, seed):
    torch.manual_seed(seed)
    model = GraphSAGE_IDS(in_channels=data.x.shape[1], hidden_channels=HIDDEN_DIM).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    n_benign = (data.y[data.train_mask] == 0).sum().item()
    n_attack = (data.y[data.train_mask] == 1).sum().item()
    class_weights = torch.tensor(
        [len(data.y[data.train_mask]) / (2 * n_benign),
         len(data.y[data.train_mask]) / (2 * n_attack)],
        dtype=torch.float32,
    ).to(DEVICE)

    model.train()
    for epoch in range(1, NUM_EPOCHS + 1):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask], weight=class_weights)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        preds_all = out.argmax(dim=1).cpu().numpy()
    return preds_all


def train_rf_one_seed(X_train, y_train, X_test, seed):
    rf = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        class_weight=RF_CLASS_WEIGHT,
        random_state=seed,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    return rf.predict(X_test)


if __name__ == "__main__":
    print(f"[INFO] Chargement du graphe : {GRAPH_FILE}")
    data = torch.load(GRAPH_FILE, weights_only=False).to(DEVICE)
    print(f"[INFO] {data.num_nodes} nœuds, {data.x.shape[1]} features")

    test_mask = data.test_mask.cpu().numpy()
    y_test = data.y.cpu().numpy()[test_mask]
    X = data.x.cpu().numpy()
    y = data.y.cpu().numpy()
    train_mask = data.train_mask.cpu().numpy()

    benign_idx = np.where(y_test == 0)[0]
    n_benign = len(benign_idx)

    overlap_ratios = []
    gnn_fn_counts = []
    rf_fn_counts = []
    gnn_benign_errors = []
    rf_benign_errors = []

    print(f"\n[INFO] {N_SEEDS} seeds -> réentraînement complet GNN + RF à chaque fois...\n")

    for seed in SEEDS:
        print(f"--- Seed {seed} ---")
        gnn_preds_all = train_gnn_one_seed(data, seed)
        gnn_preds = gnn_preds_all[test_mask]

        rf_preds = train_rf_one_seed(X[train_mask], y[train_mask], X[test_mask], seed)

        gnn_wrong = set(benign_idx[gnn_preds[benign_idx] != 0])
        rf_wrong = set(benign_idx[rf_preds[benign_idx] != 0])
        union = gnn_wrong | rf_wrong
        both = gnn_wrong & rf_wrong
        overlap = len(both) / len(union) if union else np.nan

        tn_g, fp_g, fn_g, tp_g = confusion_matrix(y_test, gnn_preds, labels=[0, 1]).ravel()
        tn_r, fp_r, fn_r, tp_r = confusion_matrix(y_test, rf_preds, labels=[0, 1]).ravel()

        print(f"  GNN: TN={tn_g} FP={fp_g} FN={fn_g} TP={tp_g}  |  "
              f"RF: TN={tn_r} FP={fp_r} FN={fn_r} TP={tp_r}  |  "
              f"Recouvrement Benign erreurs: {overlap:.1%}" if union else
              f"  GNN: TN={tn_g} FP={fp_g} FN={fn_g} TP={tp_g}  |  "
              f"RF: TN={tn_r} FP={fp_r} FN={fn_r} TP={tp_r}  |  "
              f"Aucune erreur Benign des deux côtés")

        overlap_ratios.append(overlap)
        gnn_fn_counts.append(fn_g)
        rf_fn_counts.append(fn_r)
        gnn_benign_errors.append(len(gnn_wrong))
        rf_benign_errors.append(len(rf_wrong))

    overlap_arr = np.array([o for o in overlap_ratios if not np.isnan(o)])

    print("\n" + "=" * 60)
    print("=== RÉSUMÉ SUR L'ENSEMBLE DES SEEDS ===")
    print("=" * 60)
    print(f"Recouvrement erreurs Benign (GNN vs RF), sur {len(overlap_arr)} seeds valides :")
    print(f"  Moyenne : {overlap_arr.mean():.1%}")
    print(f"  Écart-type : {overlap_arr.std():.1%}")
    print(f"  Min / Max : {overlap_arr.min():.1%} / {overlap_arr.max():.1%}")

    print(f"\nFaux négatifs (Attack ratés) :")
    print(f"  GNN -- moyenne : {np.mean(gnn_fn_counts):.1f}  (std {np.std(gnn_fn_counts):.1f})")
    print(f"  RF  -- moyenne : {np.mean(rf_fn_counts):.1f}  (std {np.std(rf_fn_counts):.1f})")

    print(f"\nErreurs Benign totales :")
    print(f"  GNN -- moyenne : {np.mean(gnn_benign_errors):.1f}  (std {np.std(gnn_benign_errors):.1f})")
    print(f"  RF  -- moyenne : {np.mean(rf_benign_errors):.1f}  (std {np.std(rf_benign_errors):.1f})")

    print("\n=== INTERPRÉTATION ===")
    if overlap_arr.std() > 0.20:
        print("Le taux de recouvrement varie BEAUCOUP d'un seed à l'autre (std > 20 pts).")
        print("-> Le faible recouvrement observé sur un seul run n'est probablement PAS")
        print("   un effet stable -- prudence avant de le présenter comme une conclusion")
        print("   forte dans le rapport.")
    else:
        print("Le taux de recouvrement reste relativement stable d'un seed à l'autre.")
        print("-> C'est un indice plus solide (mais toujours pas une preuve définitive,")
        print("   car le split train/test lui-même reste fixe sur tous ces runs).")

    if np.mean(gnn_fn_counts) > np.mean(rf_fn_counts) * 1.1:
        print("\nSur les Attack ratées (FN), le RF reste systématiquement meilleur que le")
        print("GNN en moyenne sur les seeds testés -- à mentionner clairement, ça nuance")
        print("l'argument 'le graphe capture un signal complémentaire utile'.")