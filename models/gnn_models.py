import torch
import torch.nn as nn

from torch_geometric.nn import GCNConv, SAGEConv, GATConv


class GCN(nn.Module):
    """
    Graph Convolutional Network.
    Baseline GNN pour la classification des flows.
    """

    def __init__(self, in_channels, hidden_channels=64, num_classes=2, dropout=0.30):
        super().__init__()

        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)

        self.classifier = nn.Linear(hidden_channels, num_classes)

        self.dropout = dropout

    def forward(self, x, edge_index):

        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = torch.dropout(x, p=self.dropout, train=self.training)

        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        x = torch.dropout(x, p=self.dropout, train=self.training)

        x = self.classifier(x)

        return x


class GraphSAGE(nn.Module):
    """
    GraphSAGE pour la classification des flows.
    """

    def __init__(self, in_channels, hidden_channels=64, num_classes=2, dropout=0.30):
        super().__init__()

        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)

        self.classifier = nn.Linear(hidden_channels, num_classes)

        self.dropout = dropout

    def forward(self, x, edge_index):

        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = torch.dropout(x, p=self.dropout, train=self.training)

        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        x = torch.dropout(x, p=self.dropout, train=self.training)

        x = self.classifier(x)

        return x


class GAT(nn.Module):
    """
    Graph Attention Network.
    Utilise l'attention pour pondérer les voisins.
    """

    def __init__(
        self,
        in_channels,
        hidden_channels=32,
        num_classes=2,
        heads=2,
        dropout=0.30,
    ):
        super().__init__()

        self.gat1 = GATConv(
            in_channels,
            hidden_channels,
            heads=heads,
            dropout=dropout,
        )

        self.gat2 = GATConv(
            hidden_channels * heads,
            hidden_channels,
            heads=1,
            concat=False,
            dropout=dropout,
        )

        self.classifier = nn.Linear(hidden_channels, num_classes)

        self.dropout = dropout

    def forward(self, x, edge_index):

        x = self.gat1(x, edge_index)
        x = torch.elu(x)
        x = torch.dropout(x, p=self.dropout, train=self.training)

        x = self.gat2(x, edge_index)
        x = torch.elu(x)
        x = torch.dropout(x, p=self.dropout, train=self.training)

        x = self.classifier(x)

        return x


def get_model(name, in_channels, num_classes=2):
    """
    Fabrique le modèle demandé.
    """

    name = name.lower()

    if name == "gcn":
        return GCN(
            in_channels=in_channels,
            hidden_channels=64,
            num_classes=num_classes,
        )

    if name == "graphsage":
        return GraphSAGE(
            in_channels=in_channels,
            hidden_channels=64,
            num_classes=num_classes,
        )

    if name == "gat":
        return GAT(
            in_channels=in_channels,
            hidden_channels=32,
            num_classes=num_classes,
            heads=2,
        )

    raise ValueError(
        f"Modèle inconnu : {name}. "
        f"Choisir parmi : gcn, graphsage, gat."
    )

    