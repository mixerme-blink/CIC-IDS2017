from pathlib import Path
import re

import matplotlib.pyplot as plt


ROOT_DIR = Path(__file__).resolve().parent.parent

CHECKPOINT_DIR = (
    ROOT_DIR
    / "models"
    / "checkpoints"
)

FIGURES_DIR = (
    ROOT_DIR
    / "reports"
    / "figures"
)

FIGURES_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


MODELS = [
    "gcn",
    "graphsage",
    "gat",
]


def read_metrics(model_name):

    path = (
        CHECKPOINT_DIR
        / f"{model_name}_metrics.txt"
    )

    if not path.exists():

        return None

    text = path.read_text(
        encoding="utf-8"
    )

    metrics = {}

    patterns = {
        "Accuracy": r"Accuracy:\s*([0-9.]+)",
        "Precision": r"Precision:\s*([0-9.]+)",
        "Recall": r"Recall:\s*([0-9.]+)",
        "F1-score": r"F1-score:\s*([0-9.]+)",
        "ROC-AUC": r"ROC-AUC:\s*([0-9.]+)",
    }

    for name, pattern in patterns.items():

        match = re.search(
            pattern,
            text,
        )

        if match:
            metrics[name] = float(
                match.group(1)
            )

    return metrics


def main():

    results = {}

    for model in MODELS:

        metrics = read_metrics(model)

        if metrics is not None:

            results[model.upper()] = metrics

    if not results:

        print(
            "Aucun résultat trouvé.\n"
            "Évalue d'abord les modèles."
        )

        return

    print("=" * 70)
    print("COMPARAISON DES GNN")
    print("=" * 70)

    for model, metrics in results.items():

        print(f"\n{model}")

        for metric, value in metrics.items():

            print(
                f"{metric:<10}: {value:.4f}"
            )

    # -----------------------------------------------------
    # PLOT
    # -----------------------------------------------------

    metrics_to_plot = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1-score",
        "ROC-AUC",
    ]

    models = list(results.keys())

    x = range(len(models))

    width = 0.15

    fig, ax = plt.subplots(
        figsize=(12, 7)
    )

    for i, metric in enumerate(
        metrics_to_plot
    ):

        values = [
            results[m].get(
                metric,
                0
            )
            for m in models
        ]

        positions = [
            value + (i - 2) * width
            for value in x
        ]

        ax.bar(
            positions,
            values,
            width=width,
            label=metric,
        )

    ax.set_xticks(
        list(x)
    )

    ax.set_xticklabels(
        models
    )

    ax.set_ylim(
        0,
        1.05
    )

    ax.set_ylabel(
        "Score"
    )

    ax.set_title(
        "Comparaison des modèles GNN - CIC-IDS2017"
    )

    ax.legend()

    ax.grid(
        axis="y",
        alpha=0.3,
    )

    plt.tight_layout()

    figure_path = (
        FIGURES_DIR
        / "gnn_model_comparison.png"
    )

    plt.savefig(
        figure_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"\nFigure sauvegardée : {figure_path}"
    )


if __name__ == "__main__":
    main()