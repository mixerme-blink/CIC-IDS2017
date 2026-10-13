"""Entraîne la même architecture GNN sur un graphe Cloud ou IoT."""

import argparse
import copy
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "models"))

from config import (
    CHECKPOINT_DIR,
    CLOUD_GRAPH_PATH,
    HARMONIZED_FEATURES,
    IOT_GRAPH_PATH,
    N_FEATURES,
    RANDOM_STATE,
)
from gnn_models import get_model


def metrics(model, data, mask):
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        prediction = logits[mask].argmax(dim=1)
        labels = data.y[mask]
        return {
            "loss": float(F.cross_entropy(logits[mask], labels)),
            "accuracy": accuracy_score(labels.cpu(), prediction.cpu()),
            "precision": precision_score(labels.cpu(), prediction.cpu(), zero_division=0),
            "recall": recall_score(labels.cpu(), prediction.cpu(), zero_division=0),
            "f1": f1_score(labels.cpu(), prediction.cpu(), zero_division=0),
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", choices=("cloud", "iot"), required=True)
    parser.add_argument("--model", choices=("gcn", "graphsage", "gat"), default="graphsage")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=8)
    args = parser.parse_args()

    torch.manual_seed(RANDOM_STATE)
    graph_path = CLOUD_GRAPH_PATH if args.domain == "cloud" else IOT_GRAPH_PATH
    if not graph_path.exists():
        raise FileNotFoundError(f"Graphe absent : {graph_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = torch.load(graph_path, map_location=device, weights_only=False).to(device)
    if data.x.shape[1] != N_FEATURES or list(data.feature_names) != HARMONIZED_FEATURES:
        raise ValueError(f"Le graphe {args.domain} ne respecte pas le contrat à {N_FEATURES} features.")

    model = get_model(args.model, in_channels=N_FEATURES).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    counts = torch.bincount(data.y[data.train_mask], minlength=2).float()
    class_weights = (counts.sum() / (2 * counts)).to(device)
    best_state, best_f1, stalled = None, -1.0, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(data.x, data.edge_index)
        loss = F.cross_entropy(
            logits[data.train_mask],
            data.y[data.train_mask],
            weight=class_weights,
        )
        loss.backward()
        optimizer.step()
        validation = metrics(model, data, data.val_mask)
        if validation["f1"] > best_f1:
            best_state, best_f1, stalled = copy.deepcopy(model.state_dict()), validation["f1"], 0
        else:
            stalled += 1
        if epoch == 1 or epoch % 10 == 0:
            print(f"Epoch {epoch:03d} | train loss={loss.item():.4f} | val F1={validation['f1']:.4f}")
        if stalled >= args.patience:
            break

    model.load_state_dict(best_state)
    result = metrics(model, data, data.test_mask)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint = CHECKPOINT_DIR / f"{args.domain}_{args.model}_best.pt"
    torch.save({"domain": args.domain, "model_name": args.model, "in_channels": N_FEATURES,
                "model_state_dict": model.state_dict()}, checkpoint)
    metrics_path = CHECKPOINT_DIR / f"{args.domain}_{args.model}_metrics.json"
    metrics_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Test {args.domain}/{args.model}: {json.dumps(result)}")
    print(f"Checkpoint : {checkpoint}")


if __name__ == "__main__":
    main()
