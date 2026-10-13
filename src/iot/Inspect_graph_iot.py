"""
Inspection et visualisation du graphe k-NN IoT (graph_iot_knn.pt).
PFA - Hybrid Feature Selection and GNN for Intrusion Detection

Étapes :
1. Charger le graphe et afficher un résumé statistique complet
2. Visualiser un sous-échantillon MIXTE (Benign + Attack) avec
   networkx/matplotlib, colorié par label (normal vs attaque).

   Note : sur un graphe très homophile (~93%), un snowball parti d'un
   seul nœud aléatoire reste piégé dans le cluster de la classe
   majoritaire (Attack, 82%) et n'atteint jamais les Benign. On force
   donc DEUX départs -- un Benign, un Attack -- puis on combine les
   deux échantillons pour une visualisation qui montre vraiment les
   deux classes.

Utilisation :
    python Inspect_graph_iot.py
"""

import torch
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx, subgraph
from matplotlib.lines import Line2D

# ============================================================
# CONFIGURATION
# ============================================================
GRAPH_FILE = r"D:\CICIoT2023\graph_iot_knn.pt"
N_NODES_TO_PLOT = 1000   # sous-échantillon pour la visualisation (au-delà, illisible)
OUTPUT_IMAGE = r"D:\CICIoT2023\graph_iot_sample_visualization.png"


# ============================================================
# ÉTAPE 1 — Résumé statistique du graphe complet
# ============================================================
def print_summary(data):
    print("=== RÉSUMÉ DU GRAPHE ===")
    print(f"Nombre de nœuds       : {data.num_nodes}")
    print(f"Nombre d'arêtes       : {data.num_edges}")
    print(f"Features par nœud     : {data.x.shape[1]}")
    print(f"Degré moyen           : {data.num_edges / data.num_nodes:.1f}")

    print(f"\nRépartition des labels (0=Benign, 1=Attack) :")
    unique, counts = torch.unique(data.y, return_counts=True)
    for u, c in zip(unique.tolist(), counts.tolist()):
        pct = c / data.num_nodes * 100
        label_name = "Benign" if u == 0 else "Attack"
        print(f"  {label_name} ({u}) : {c} nœuds ({pct:.2f}%)")

    print(f"\nTrain / Test : {data.train_mask.sum().item()} / {data.test_mask.sum().item()}")

    print(f"\nStatistiques des features (min/max/mean par colonne, sur données normalisées) :")
    print(f"  Min  : {data.x.min(dim=0).values}")
    print(f"  Max  : {data.x.max(dim=0).values}")
    print(f"  Mean : {data.x.mean(dim=0)}")


# ============================================================
# ÉTAPE 2 — Visualiser un sous-échantillon mixte (snowball x2)
# ============================================================
def snowball_sample(data, n_nodes, seed=42, start_node=None):
    """Échantillonnage par propagation (BFS) pour garder une vraie structure
    connectée, contrairement à un tirage aléatoire pur qui casse les arêtes.
    Si start_node est fourni, force le point de départ (utile pour
    garantir un échantillon dans une classe spécifique sur un graphe très
    homophile, où un départ aléatoire reste piégé dans le cluster
    majoritaire)."""
    torch.manual_seed(seed)
    if start_node is None:
        start_node = torch.randint(0, data.num_nodes, (1,)).item()

    edge_index = data.edge_index
    adjacency = {}
    for src, dst in edge_index.t().tolist():
        adjacency.setdefault(src, set()).add(dst)
        adjacency.setdefault(dst, set()).add(src)

    visited = {start_node}
    frontier = [start_node]
    while len(visited) < n_nodes and frontier:
        next_frontier = []
        for node in frontier:
            for neighbor in adjacency.get(node, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    next_frontier.append(neighbor)
                    if len(visited) >= n_nodes:
                        break
            if len(visited) >= n_nodes:
                break
        frontier = next_frontier

    print(f"[INFO] Échantillonnage snowball : {len(visited)} nœuds atteints "
          f"depuis le nœud de départ {start_node}")
    return torch.tensor(list(visited))


def visualize_subgraph(data, n_nodes, output_image):
    print(f"\n=== VISUALISATION D'UN SOUS-GRAPHE ({n_nodes} nœuds, snowball mixte) ===")

    # Deux nœuds de départ forcés : un Benign, un Attack -- pour être
    # sûr de voir les deux classes malgré la forte homophilie du graphe.
    benign_nodes = (data.y == 0).nonzero(as_tuple=True)[0]
    attack_nodes = (data.y == 1).nonzero(as_tuple=True)[0]

    torch.manual_seed(42)
    benign_start = benign_nodes[torch.randint(0, len(benign_nodes), (1,))].item()
    attack_start = attack_nodes[torch.randint(0, len(attack_nodes), (1,))].item()

    n_half = n_nodes // 2
    benign_sample = snowball_sample(data, n_half, seed=42, start_node=benign_start)
    attack_sample = snowball_sample(data, n_half, seed=43, start_node=attack_start)

    node_indices = torch.unique(torch.cat([benign_sample, attack_sample]))
    n_final = len(node_indices)

    edge_index_sub, _ = subgraph(
        node_indices, data.edge_index, relabel_nodes=True, num_nodes=data.num_nodes
    )
    labels_sub = data.y[node_indices]

    G = nx.Graph()
    G.add_nodes_from(range(n_final))
    edges = edge_index_sub.t().tolist()
    G.add_edges_from(edges)

    colors = ["#2ecc71" if lab == 0 else "#e74c3c" for lab in labels_sub.tolist()]

    n_benign_shown = sum(1 for lab in labels_sub.tolist() if lab == 0)
    n_attack_shown = n_final - n_benign_shown
    print(f"[INFO] Échantillon final affiché : {n_benign_shown} Benign, "
          f"{n_attack_shown} Attack")

    plt.figure(figsize=(12, 10))
    pos = nx.spring_layout(G, seed=42, k=0.3)
    nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=60, alpha=0.85)
    nx.draw_networkx_edges(G, pos, alpha=0.15, width=0.5)

    plt.title(f"Graphe k-NN IoT — échantillon mixte de {n_final} nœuds\n"
              f"Vert = Benign, Rouge = Attack", fontsize=13)
    plt.axis("off")

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Benign',
               markerfacecolor='#2ecc71', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Attack',
               markerfacecolor='#e74c3c', markersize=10),
    ]
    plt.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()
    plt.savefig(output_image, dpi=150)
    print(f"[OK] Visualisation sauvegardée : {output_image}")
    plt.show()


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print(f"[INFO] Chargement de {GRAPH_FILE} ...")
    data = torch.load(GRAPH_FILE, weights_only=False)

    print_summary(data)
    visualize_subgraph(data, N_NODES_TO_PLOT, OUTPUT_IMAGE)