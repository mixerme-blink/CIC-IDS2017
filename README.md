# Hybrid Feature Selection and GNN for Network Intrusion Detection

Projet PFA de détection binaire d'intrusions réseau dans les domaines Cloud
(CIC-IDS2017) et IoT (CICIoT2023). Chaque flow est un nœud et porte le même
contrat de huit features. Les graphes restent adaptés aux domaines : relations
IP/temps pour le Cloud, KNN pour l'IoT.

## Contrat commun

L'ordre est strict et identique dans les deux graphes :

```text
packet_count, total_bytes, packet_len_mean, packet_len_std,
packet_len_var, packet_len_min, packet_len_max, flow_rate
```

La normalisation est `MinMaxScaler`, ajustée sur le split d'entraînement du
domaine concerné uniquement. Le label binaire est `0 = BENIGN`, `1 = ATTACK`.

## Installation

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Exécution

Les artefacts Cloud sont déjà générables avec :

```powershell
python src\feature_engineering.py
python src\feature_alignment.py
python src\feature_selection.py
python src\build_graph.py
```

Placez les CSV CICIoT2023 sous `data/raw/ciciot2023/`, en préservant les
dossiers de catégories. Construisez ensuite le graphe IoT :

```powershell
python src\iot\build_graph.py --raw-dir data\raw\ciciot2023
```

Entraînez la même architecture sur chaque domaine :

```powershell
python src\train_domain_gnn.py --domain cloud --model graphsage
python src\train_domain_gnn.py --domain iot --model graphsage
```

Évaluez ensuite le transfert de domaine :

```powershell
python src\evaluate_cross_domain.py --source cloud --target iot --model graphsage
python src\evaluate_cross_domain.py --source iot --target cloud --model graphsage
```

Les checkpoints et résultats sont enregistrés dans `models/checkpoints/`.

## Limite importante

Le cross-domain mesure la généralisation entre deux distributions différentes.
Il est normal que ses scores soient moins élevés que les scores in-domain.
Les deux graphes ont les mêmes features de nœuds, mais des topologies
différentes par choix méthodologique.
