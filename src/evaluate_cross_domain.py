"""Évalue un modèle entraîné sur Cloud ou IoT sur l'autre domaine."""

import argparse
import json
import sys
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "models"))

from config import CHECKPOINT_DIR, CLOUD_GRAPH_PATH, HARMONIZED_FEATURES, IOT_GRAPH_PATH, N_FEATURES
from gnn_models import get_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=("cloud", "iot"), required=True)
    parser.add_argument("--target", choices=("cloud", "iot"), required=True)
    parser.add_argument("--model", choices=("gcn", "graphsage", "gat"), default="graphsage")
    args = parser.parse_args()
    if args.source == args.target:
        raise ValueError("Utilise train_domain_gnn.py pour une évaluation in-domain.")

    graph_path = CLOUD_GRAPH_PATH if args.target == "cloud" else IOT_GRAPH_PATH
    checkpoint_path = CHECKPOINT_DIR / f"{args.source}_{args.model}_best.pt"
    if not graph_path.exists() or not checkpoint_path.exists():
        raise FileNotFoundError("Le graphe cible et le checkpoint source doivent être générés avant l'évaluation.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = torch.load(graph_path, map_location=device, weights_only=False).to(device)
    if data.x.shape[1] != N_FEATURES or list(data.feature_names) != HARMONIZED_FEATURES:
        raise ValueError("Le graphe cible ne respecte pas le contrat de features harmonisées.")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = get_model(args.model, in_channels=N_FEATURES).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)[data.test_mask]
        labels = data.y[data.test_mask]
        predicted = logits.argmax(dim=1)
        probabilities = torch.softmax(logits, dim=1)[:, 1]
    result = {
        "source": args.source, "target": args.target, "model": args.model,
        "accuracy": accuracy_score(labels.cpu(), predicted.cpu()),
        "precision": precision_score(labels.cpu(), predicted.cpu(), zero_division=0),
        "recall": recall_score(labels.cpu(), predicted.cpu(), zero_division=0),
        "f1": f1_score(labels.cpu(), predicted.cpu(), zero_division=0),
        "roc_auc": roc_auc_score(labels.cpu(), probabilities.cpu()),
    }
    output = CHECKPOINT_DIR / f"cross_{args.source}_to_{args.target}_{args.model}_metrics.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Résultats : {output}")


if __name__ == "__main__":
    main()
