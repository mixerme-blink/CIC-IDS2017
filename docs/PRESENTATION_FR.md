# Guide de présentation

## Problème

Un IDS (Intrusion Detection System) cherche à classer un flow réseau comme
normal (`0`) ou attaque (`1`). Les modèles classiques traitent chaque flow de
manière isolée. Notre approche GNN utilise aussi les relations entre flows.

## Contribution

Nous combinons une sélection/harmonisation de features avec des GNN pour deux
domaines différents : le Cloud (CIC-IDS2017) et l'IoT (CICIoT2023). La
contribution centrale est un contrat de huit features communes :

1. `packet_count`
2. `total_bytes`
3. `packet_len_mean`
4. `packet_len_std`
5. `packet_len_var`
6. `packet_len_min`
7. `packet_len_max`
8. `flow_rate`

Ces variables décrivent le volume, le nombre et la distribution des paquets
d'un flow. Leur ordre est identique dans `data.x` des deux graphes.

## Pipeline

```text
Dataset brut -> nettoyage -> features harmonisées -> split 70/10/20
-> Min-Max fit sur train -> graphe -> GNN -> classification binaire
```

Le scaler est ajusté uniquement sur train afin d'éviter le data leakage. Les
splits validation et test sont seulement transformés par ce scaler.

## Graphes

Pour le Cloud, un nœud représente un flow et les arêtes relient des flows qui
partagent une IP source ou destination, en tenant compte de l'ordre temporel.
Pour l'IoT, les arêtes viennent des k voisins les plus proches dans l'espace
des huit features normalisées. La différence de topologie est assumée : elle
représente la nature différente des deux environnements.

## Modèles et évaluation

GCN, GraphSAGE et GAT sont disponibles. Chaque modèle reçoit `[N, 8]` et
produit une probabilité pour les deux classes. Nous rapportons accuracy,
precision, recall, F1 et ROC-AUC. Le F1 est essentiel car il équilibre faux
positifs et fausses alertes négatives ; le recall attaque mesure la capacité à
ne pas manquer une intrusion.

## Cross-domain

Après entraînement sur Cloud, le modèle est évalué sur le test IoT ; l'essai
inverse est également effectué. Les résultats cross-domain peuvent être plus
faibles que les résultats in-domain : c'est le domain shift, dû aux appareils,
protocoles et comportements réseau différents. L'harmonisation des huit
features rend toutefois cette comparaison techniquement valide.

## Résultats Cloud actuels

Sur le graphe Cloud harmonisé, les dernières évaluations enregistrées donnent
un F1 d'environ `0.668` pour GCN et `0.674` pour GraphSAGE. GraphSAGE a un
recall attaque très élevé (environ `0.989`), mais une précision et une accuracy
plus faibles : il détecte presque toutes les attaques, au prix de davantage de
faux positifs. Il faut présenter ce compromis honnêtement plutôt que retenir
uniquement le F1. Les résultats IoT et cross-domain doivent être régénérés avec
la nouvelle pipeline Min-Max avant d'être comparés.

## Réponses courtes aux questions probables

**Pourquoi un GNN ?** Les attaques peuvent apparaître comme des motifs de
communications liés ; le GNN agrège l'information des voisins d'un flow.

**Pourquoi Min-Max ?** C'est une convention partagée entre Cloud et IoT, qui
place les variables train sur une échelle comparable sans fuite de données.

**Pourquoi deux graphes différents ?** Les relations naturelles disponibles ne
sont pas les mêmes. Le contrat commun porte sur les features de nœuds, pas sur
la topologie imposée.

**Quelle limite ?** Le projet ne garantit pas qu'un modèle Cloud généralise
parfaitement vers l'IoT ; l'évaluation cross-domain mesure précisément cette
limite et motive des travaux futurs de domain adaptation.
