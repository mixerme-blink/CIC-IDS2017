"""
Diagnostic d'HOMOPHILIE du graphe k-NN IoT.

Question posée : quand un nœud a pour voisins d'autres nœuds dans le
graphe, ces voisins ont-ils la MÊME classe (Benign/Attack) que lui ?

Pourquoi c'est important : le message passing (SAGEConv, GCN, GAT, etc.)
fonctionne en mélangeant/moyennant les features d'un nœud avec celles de
ses voisins. Si les voisins d'un nœud Benign sont majoritairement des
Attack (hétérophilie locale), chaque couche du GNN va DILUER le signal
Benign vers celui d'Attack -- ce qui peut faire plus de mal que de bien,
surtout pour la classe minoritaire.

Ce script calcule, pour chaque nœud, la proportion de ses voisins qui
partagent sa classe, puis compare ça à ce qu'on obtiendrait avec un
graphe purement aléatoire (même degré, mais voisins tirés au hasard) --
c'est la référence "aucune structure utile".

Utilisation :
    python diagnostic_homophilie.py
"""

import numpy as np
import torch

GRAPH_FILE = r"D:\CICIoT2023\graph_iot_knn.pt"

if __name__ == "__main__":
    print(f"[INFO] Chargement du graphe : {GRAPH_FILE}")
    data = torch.load(GRAPH_FILE, weights_only=False)

    y = data.y.numpy()
    edge_index = data.edge_index.numpy()
    n_nodes = data.num_nodes

    src, dst = edge_index[0], edge_index[1]
    same_class = (y[src] == y[dst])

    # ---- Homophilie globale (toutes arêtes confondues) ----
    global_homophily = same_class.mean()
    print(f"\n=== HOMOPHILIE GLOBALE ===")
    print(f"Proportion d'arêtes reliant deux nœuds de MÊME classe : {global_homophily:.1%}")

    # ---- Référence aléatoire (à quoi s'attendre si le graphe était sans structure) ----
    p_benign = (y == 0).mean()
    p_attack = (y == 1).mean()
    random_baseline = p_benign**2 + p_attack**2
    print(f"Référence (graphe aléatoire, même proportions de classes) : {random_baseline:.1%}")
    print(f"  -> Un score proche de la référence = le graphe n'apporte AUCUNE structure")
    print(f"     utile par rapport au hasard.")
    print(f"  -> Un score nettement AU-DESSUS = bon signe (les voisins se ressemblent).")
    print(f"  -> Un score EN-DESSOUS = mauvais signe (les voisins sont anormalement")
    print(f"     différents -- pire que le hasard).")

    # ---- Homophilie spécifique à la classe Benign (celle qui nous intéresse) ----
    print(f"\n=== HOMOPHILIE SPÉCIFIQUE À LA CLASSE BENIGN (0) ===")
    benign_node_ids = np.where(y == 0)[0]
    per_node_benign_ratio = []
    degrees = []

    for node in benign_node_ids:
        neighbor_mask = (src == node)
        neighbors = dst[neighbor_mask]
        if len(neighbors) == 0:
            continue
        ratio = (y[neighbors] == 0).mean()
        per_node_benign_ratio.append(ratio)
        degrees.append(len(neighbors))

    per_node_benign_ratio = np.array(per_node_benign_ratio)
    print(f"Nombre de nœuds Benign analysés : {len(per_node_benign_ratio)}")
    print(f"Degré moyen de ces nœuds : {np.mean(degrees):.1f}")
    print(f"Proportion moyenne de voisins Benign, PAR nœud Benign : "
          f"{per_node_benign_ratio.mean():.1%}")
    print(f"  (Référence attendue si aucune structure : {p_benign:.1%} -- "
          f"la proportion naturelle de Benign dans tout le graphe)")

    pct_isolated = (per_node_benign_ratio == 0).mean()
    print(f"\nProportion de nœuds Benign qui n'ont AUCUN voisin Benign "
          f"(100% de leurs voisins sont Attack) : {pct_isolated:.1%}")

    # ---- Focus sur le test set (là où ça compte pour l'évaluation) ----
    if hasattr(data, "test_mask"):
        test_mask = data.test_mask.numpy()
        test_benign_ids = np.where((y == 0) & test_mask)[0]
        print(f"\n=== FOCUS TEST SET ({len(test_benign_ids)} nœuds Benign réels) ===")
        ratios_test = []
        for node in test_benign_ids:
            neighbors = dst[src == node]
            if len(neighbors) == 0:
                continue
            ratios_test.append((y[neighbors] == 0).mean())
        ratios_test = np.array(ratios_test)
        print(f"Proportion moyenne de voisins Benign : {ratios_test.mean():.1%}")
        print(f"Proportion de nœuds Benign test isolés (0 voisin Benign) : "
              f"{(ratios_test == 0).mean():.1%}")

    print("\n=== VERDICT ===")
    if per_node_benign_ratio.mean() <= p_benign * 1.3:
        print("L'homophilie sur la classe Benign est proche (ou pire) que le hasard.")
        print("-> Le graphe k-NN ne regroupe PAS efficacement les Benign entre eux.")
        print("-> Le message passing dilue probablement le signal minoritaire au lieu")
        print("   de le renforcer. C'est cohérent avec le fait que le GNN sous-performe")
        print("   le RF de façon stable sur tous les seeds testés.")
    else:
        print("L'homophilie Benign est nettement supérieure au hasard.")
        print("-> Le graphe regroupe correctement les Benign -- le problème vient")
        print("   probablement d'ailleurs (architecture, hyperparamètres, profondeur).")