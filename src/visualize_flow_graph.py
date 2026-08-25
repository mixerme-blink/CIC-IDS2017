from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import networkx as nx
import numpy as np
import torch


ROOT_DIR = Path(__file__).resolve().parent.parent

GRAPH_PATH = ROOT_DIR / "data" / "processed" / "cloud_flow_graph.pt"
OUTPUT_PATH = (
    ROOT_DIR / "reports" / "figures" / "cloud_flow_graph_sample.png"
)

RANDOM_STATE = 42
MAX_DISPLAY_NODES = 120
ATTACK_SEED_COUNT = 20
TARGET_PER_CLASS = 60


def add_nodes_until(selected_nodes, candidates, maximum, rng):
    """Ajoute des nœuds sans dépasser la taille souhaitée."""
    candidates = np.asarray(candidates)
    rng.shuffle(candidates)

    for node in candidates:
        if len(selected_nodes) >= maximum:
            break
        selected_nodes.add(int(node))


def main():
    if not GRAPH_PATH.exists():
        raise FileNotFoundError(
            "Le graphe est introuvable.\n"
            "Exécute d'abord : python src/build_graph.py"
        )

    cloud_graph = torch.load(
        GRAPH_PATH,
        weights_only=False,
    )

    edge_index = cloud_graph.edge_index.cpu().numpy()
    labels = cloud_graph.y.cpu().numpy()

    source_nodes = edge_index[0]
    destination_nodes = edge_index[1]

    rng = np.random.default_rng(RANDOM_STATE)

    attack_nodes = np.where(labels == 1)[0]
    normal_nodes = np.where(labels == 0)[0]

    if len(attack_nodes) < TARGET_PER_CLASS:
        raise ValueError("Pas assez de flows d'attaque.")

    if len(normal_nodes) < TARGET_PER_CLASS:
        raise ValueError("Pas assez de flows normaux.")

    # On commence par 20 attaques pour conserver une structure connectée.
    attack_seeds = rng.choice(
        attack_nodes,
        size=min(ATTACK_SEED_COUNT, len(attack_nodes)),
        replace=False,
    )

    # Nœuds voisins des attaques sélectionnées.
    incident_mask = (
        np.isin(source_nodes, attack_seeds)
        | np.isin(destination_nodes, attack_seeds)
    )

    neighboring_nodes = np.unique(
        np.concatenate([
            source_nodes[incident_mask],
            destination_nodes[incident_mask],
        ])
    )

    nearby_attacks = neighboring_nodes[
        labels[neighboring_nodes] == 1
    ]

    nearby_normals = neighboring_nodes[
        labels[neighboring_nodes] == 0
    ]

    # --- Sélection exacte de 60 attaques ---
    selected_attacks = set(
        int(node) for node in attack_seeds
    )

    remaining_attack_candidates = [
        int(node)
        for node in nearby_attacks
        if int(node) not in selected_attacks
    ]

    rng.shuffle(remaining_attack_candidates)

    for node in remaining_attack_candidates:
        if len(selected_attacks) >= TARGET_PER_CLASS:
            break
        selected_attacks.add(node)

    # Complète avec des attaques aléatoires si nécessaire.
    if len(selected_attacks) < TARGET_PER_CLASS:
        remaining_attacks = [
            int(node)
            for node in attack_nodes
            if int(node) not in selected_attacks
        ]

        rng.shuffle(remaining_attacks)

        for node in remaining_attacks:
            if len(selected_attacks) >= TARGET_PER_CLASS:
                break
            selected_attacks.add(node)

    # --- Sélection exacte de 60 flows normaux ---
    normal_candidates = [int(node) for node in nearby_normals]

    if len(normal_candidates) >= TARGET_PER_CLASS:
        selected_normals = list(
            rng.choice(
                normal_candidates,
                size=TARGET_PER_CLASS,
                replace=False,
            )
        )
    else:
        selected_normals = normal_candidates.copy()

        remaining_normals = [
            int(node)
            for node in normal_nodes
            if int(node) not in selected_normals
        ]

        rng.shuffle(remaining_normals)

        selected_normals.extend(
            remaining_normals[
                :TARGET_PER_CLASS - len(selected_normals)
            ]
        )

    # Résultat final : exactement 120 nœuds.
    selected_nodes = np.array(
        sorted(
            list(selected_attacks)
            + list(selected_normals)
        ),
        dtype=int,
    )

    # Conserve seulement les arêtes entre les nœuds affichés.
    subgraph_edge_mask = (
        np.isin(source_nodes, selected_nodes)
        & np.isin(destination_nodes, selected_nodes)
    )

    graph = nx.Graph()
    graph.add_nodes_from(selected_nodes.tolist())

    for source, destination in zip(
        source_nodes[subgraph_edge_mask],
        destination_nodes[subgraph_edge_mask],
    ):
        graph.add_edge(int(source), int(destination))

    # Vert = normal ; rouge = attaque.
    node_colors = [
        "#d62728" if labels[node] == 1 else "#2ca02c"
        for node in graph.nodes()
    ]

    # Taille selon les connexions, limitée pour rester lisible.
    node_sizes = [
        min(450, 70 + 55 * np.sqrt(graph.degree(node)))
        for node in graph.nodes()
    ]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(14, 9))

    positions = nx.spring_layout(
        graph,
        seed=RANDOM_STATE,
        k=0.7,
    )

    nx.draw_networkx_edges(
        graph,
        positions,
        edge_color="#999999",
        alpha=0.35,
        width=0.8,
    )

    nx.draw_networkx_nodes(
        graph,
        positions,
        node_color=node_colors,
        node_size=node_sizes,
        alpha=0.85,
    )

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Flow normal",
            markerfacecolor="#2ca02c",
            markersize=10,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Flow d'attaque",
            markerfacecolor="#d62728",
            markersize=10,
        ),
    ]

    plt.legend(handles=legend_elements, loc="upper right")

    attack_count = sum(
        labels[node] == 1
        for node in graph.nodes()
    )

    plt.title(
        "Sous-graphe Cloud équilibré — CIC-IDS2017\n"
        "60 flows normaux et 60 flows d'attaque\n"
        "Nœud = flow | Arête = relation entre flows"
    )

    plt.axis("off")
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=220, bbox_inches="tight")

    print(f"Image créée : {OUTPUT_PATH}")
    print(f"Flows affichés : {graph.number_of_nodes()}")
    print(f"Relations affichées : {graph.number_of_edges()}")
    print(f"Flows d'attaque affichés : {attack_count}")

    plt.show()


if __name__ == "__main__":
    main()