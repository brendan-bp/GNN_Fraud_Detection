"""
gnn_fraud_detection.py
======================
A prototype Graph Neural Network (GNN) for fraud detection using PyTorch Geometric.
Implements two architectures side-by-side:
  - GCN  (Graph Convolutional Network — Kipf & Welling, 2017)
  - GraphSAGE (Hamilton et al., 2017)

In this project each node represents an entity (e.g. a bank account, ABN registered company, company director).
Each edge represents a relationship between entities (e.g. a bank/crypto transaction, or financial contracts
that link companies (such as debt owed to other companies, leases, or agreements to use intellectual property).
The model learns to classify nodes as fraudulent (1) or legitimate (0) based on the structure of the graph,
which can provide much more meaningful information than simple information about the node in isolation.

Dependencies:
    pip install torch torch_geometric
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# PyG (PyTorch Geometric) convolution layers
from torch_geometric.nn import GCNConv, SAGEConv


# ===========================================================================
# 1.  GCN-based Fraud Detector
# ===========================================================================

class GCNFraudDetector(nn.Module):
    """
    A Graph Convolutional Network for node-level binary classification.

    Architecture
    ------------
    Input node features  →  GCN layer 1  →  ReLU + Dropout
                         →  GCN layer 2  →  ReLU + Dropout
                         →  GCN layer 3  →  Linear head  →  Sigmoid

    GCN convolution (Kipf & Welling 2017) computes:
        H' = σ( D̃^{-1/2} Ã D̃^{-1/2} H W )
    where
        Ã = A + I   (adjacency matrix with added self-loops)
        D̃           (degree matrix of Ã)
        H           (current node-feature matrix)
        W           (learnable weight matrix)

    In plain English: each node's new representation is a normalised,
    weighted average of its neighbours' features plus its own features,
    passed through a learnable linear transformation.

    Parameters
    ----------
    in_channels  : int   - number of input features per node
    hidden_dim   : int   - width of the hidden GCN layers
    out_channels : int   - number of output classes (2 for binary fraud)
    dropout      : float - dropout probability applied after each conv
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 64,
        out_channels: int = 2,
        dropout: float = 0.5,
    ):
        super().__init__()

        self.dropout = dropout

        # ── Layer 1: maps raw node features → hidden representation ──────────
        # GCNConv handles the normalised neighbourhood aggregation internally.
        # `add_self_loops=True` (default) ensures every node attends to itself.
        self.conv1 = GCNConv(in_channels, hidden_dim, add_self_loops=True)

        # ── Layer 2: deeper hidden representation ─────────────────────────────
        self.conv2 = GCNConv(hidden_dim, hidden_dim, add_self_loops=True)

        # ── Layer 3: compress to a smaller embedding before the head ─────────
        self.conv3 = GCNConv(hidden_dim, hidden_dim // 2, add_self_loops=True)

        # ── Classification head ───────────────────────────────────────────────
        # A plain linear layer maps the final node embedding to class logits.
        # For binary fraud detection out_channels=2 (legit / fraud).
        self.classifier = nn.Linear(hidden_dim // 2, out_channels)

        # ── Batch normalisation (optional, stabilises training) ───────────────
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x          : Tensor of shape [num_nodes, in_channels]
                     Node feature matrix.
        edge_index : LongTensor of shape [2, num_edges]
                     COO-format edge list.  edge_index[0] = source nodes,
                     edge_index[1] = target nodes.

        Returns
        -------
        Tensor of shape [num_nodes, out_channels]
            Raw logits for each node.  Apply softmax / sigmoid outside
            if you need probabilities.
        """

        # ── Conv block 1 ──────────────────────────────────────────────────────
        # GCNConv(x, edge_index) aggregates neighbour features and applies W.
        x = self.conv1(x, edge_index)
        x = self.bn1(x)             # normalise across the node batch
        x = F.relu(x)               # non-linearity
        x = F.dropout(x, p=self.dropout, training=self.training)
        # NOTE: `training=self.training` ensures dropout is disabled at eval time

        # ── Conv block 2 ──────────────────────────────────────────────────────
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # ── Conv block 3 (no BN here; smaller dim) ────────────────────────────
        x = self.conv3(x, edge_index)
        x = F.relu(x)

        # ── Classification head ───────────────────────────────────────────────
        # No activation here — raw logits are expected by CrossEntropyLoss.
        out = self.classifier(x)

        return out  # shape: [num_nodes, out_channels]

    def predict_proba(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Convenience method: returns fraud probability per node (class index 1)."""
        logits = self.forward(x, edge_index)
        # softmax converts logits to a probability distribution over classes
        return F.softmax(logits, dim=-1)[:, 1]


# ===========================================================================
# 2.  GraphSAGE-based Fraud Detector
# ===========================================================================

class SAGEFraudDetector(nn.Module):
    """
    A GraphSAGE (SAmple and aggreGatE) network for node-level classification.

    Architecture
    ------------
    Input node features  →  SAGE layer 1  →  ReLU + Dropout
                         →  SAGE layer 2  →  ReLU + Dropout
                         →  Linear head  →  Logits

    GraphSAGE key difference vs GCN
    --------------------------------
    Instead of a global, pre-computed, normalised aggregation, SAGE
    samples a fixed-size neighbourhood at each layer and concatenates
    the aggregated neighbour embedding with the node's own embedding:

        h_v' = W · CONCAT( h_v,  AGG({ h_u : u ∈ N(v) }) )

    This makes SAGE inductive — it can generalise to nodes unseen during
    training — which is important in fraud detection where new accounts
    appear continuously.

    PyG's SAGEConv uses mean aggregation by default (aggr='mean').
    You can switch to 'max' or 'lstm' for richer aggregation.

    Parameters
    ----------
    in_channels  : int   - input feature dimension
    hidden_dim   : int   - hidden layer width
    out_channels : int   - number of output classes
    dropout      : float - dropout rate
    aggr         : str   - neighbour aggregation ('mean', 'max', 'lstm')
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 64,
        out_channels: int = 2,
        dropout: float = 0.5,
        aggr: str = "mean",
    ):
        super().__init__()

        self.dropout = dropout

        # ── SAGE Layer 1 ──────────────────────────────────────────────────────
        # SAGEConv(in, out, aggr) will concatenate the node's own features
        # with the aggregated neighbour features internally, then project.
        self.conv1 = SAGEConv(in_channels, hidden_dim, aggr=aggr)

        # ── SAGE Layer 2 ──────────────────────────────────────────────────────
        self.conv2 = SAGEConv(hidden_dim, hidden_dim, aggr=aggr)

        # ── Batch norm ────────────────────────────────────────────────────────
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

        # ── MLP classification head ───────────────────────────────────────────
        # Two-layer MLP gives the model extra capacity between the GNN output
        # and the final class logits (common practice in fraud detection).
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim // 2, out_channels),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x          : Tensor [num_nodes, in_channels]
        edge_index : LongTensor [2, num_edges]

        Returns
        -------
        Tensor [num_nodes, out_channels]
        """

        # ── SAGE conv 1 ───────────────────────────────────────────────────────
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # ── SAGE conv 2 ───────────────────────────────────────────────────────
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # ── MLP head ──────────────────────────────────────────────────────────
        out = self.mlp(x)

        return out

    def predict_proba(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Returns fraud probability per node (class index 1)."""
        return F.softmax(self.forward(x, edge_index), dim=-1)[:, 1]


# ===========================================================================
# 3.  Training & Evaluation helpers
# ===========================================================================

def build_loss(class_weights: torch.Tensor | None = None) -> nn.CrossEntropyLoss:
    """
    Build a weighted cross-entropy loss.

    Fraud datasets are heavily imbalanced (e.g. 99% legitimate, 1% fraud).
    Passing class_weights upweights the minority fraud class so the model
    doesn't just learn to always predict 'legitimate'.

    Example
    -------
    # If ~1% of nodes are fraudulent:
    weights = torch.tensor([1.0, 99.0])   # [weight_legit, weight_fraud]
    criterion = build_loss(weights)
    """
    return nn.CrossEntropyLoss(weight=class_weights)


def train_one_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    data,           # a PyG Data object: has .x, .edge_index, .y, .train_mask
    device: torch.device,
) -> float:
    """
    Run one full training epoch on the node classification task.

    PyG uses *masks* (Boolean tensors over all nodes) to indicate which
    nodes belong to train / val / test splits.  We compute the loss only
    on train_mask nodes, but the GNN still uses ALL nodes and edges for
    message passing — this is the transductive semi-supervised setting.

    Returns
    -------
    float – training loss for this epoch
    """
    model.train()                           # activates dropout / BN train mode
    optimizer.zero_grad()                   # clear gradients from last step

    # Forward pass over the full graph
    logits = model(data.x.to(device), data.edge_index.to(device))
    #   logits : [num_nodes, num_classes]

    # Compute loss ONLY on labelled training nodes
    loss = criterion(
        logits[data.train_mask],            # predicted logits for train nodes
        data.y[data.train_mask].to(device), # ground-truth labels
    )

    loss.backward()                         # compute gradients
    optimizer.step()                        # update weights

    return loss.item()


@torch.no_grad()   # disables gradient tracking — saves memory during eval
def evaluate(
    model: nn.Module,
    criterion: nn.Module,
    data,
    mask: torch.Tensor,
    device: torch.device,
) -> dict:
    """
    Evaluate the model on any node split (val or test).

    Returns a dict with loss and accuracy for the given mask.
    """
    model.eval() # deactivates dropout / BN train mode

    logits = model(data.x.to(device), data.edge_index.to(device))
    loss   = criterion(logits[mask], data.y[mask].to(device)).item()

    # Convert logits → predicted class (argmax across class dimension)
    preds  = logits[mask].argmax(dim=-1)
    labels = data.y[mask].to(device)

    accuracy = (preds == labels).float().mean().item()

    return {"loss": loss, "accuracy": accuracy}


# ===========================================================================
# 4.  Factory / entry-point
# ===========================================================================

def build_model(
    arch: str,
    in_channels: int,
    hidden_dim: int = 64,
    out_channels: int = 2,
    dropout: float = 0.5,
) -> nn.Module:
    """
    Instantiate a GNN fraud detector by architecture name.

    Parameters
    ----------
    arch        : 'gcn' or 'sage'
    in_channels : number of node features
    hidden_dim  : hidden layer width
    out_channels: number of classes (default 2: legit / fraud)
    dropout     : dropout rate

    Returns
    -------
    An nn.Module ready for training.
    """
    arch = arch.lower()
    if arch == "gcn":
        return GCNFraudDetector(in_channels, hidden_dim, out_channels, dropout)
    elif arch in ("sage", "graphsage"):
        return SAGEFraudDetector(in_channels, hidden_dim, out_channels, dropout)
    else:
        raise ValueError(f"Unknown architecture '{arch}'. Choose 'gcn' or 'sage'.")


# ===========================================================================
# 5.  Minimal smoke-test (no real data needed)
# ===========================================================================

if __name__ == "__main__":
    """
    Quick sanity check with randomly generated graph data.
    Run:  python gnn_fraud_detection.py
    """

    torch.manual_seed(42)

    # ── Fake graph: 500 nodes, 10 features each, ~2000 edges ─────────────────
    num_nodes    = 500
    num_features = 10
    num_edges    = 2000

    x          = torch.randn(num_nodes, num_features)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))  # random COO edges
    y          = torch.randint(0, 2, (num_nodes,))            # random binary labels

    # ── Masks: 60% train, 20% val, 20% test ──────────────────────────────────
    perm        = torch.randperm(num_nodes)
    train_mask  = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask    = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask   = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[perm[:300]]   = True
    val_mask[perm[300:400]]  = True
    test_mask[perm[400:]]    = True

    # Wrap in a simple namespace so train/eval helpers can access .x, .y etc.
    class FakeData:
        pass

    data = FakeData()
    data.x, data.edge_index = x, edge_index
    data.y, data.train_mask = y, train_mask
    data.val_mask, data.test_mask = val_mask, test_mask

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for arch in ("gcn", "sage"):
        print(f"\n{'='*50}\nTesting {arch.upper()} architecture\n{'='*50}")

        model = build_model(arch, in_channels=num_features).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
        criterion = build_loss()  # unweighted for the smoke test

        for epoch in range(1, 6):
            loss      = train_one_epoch(model, optimizer, criterion, data, device)
            val_stats = evaluate(model, criterion, data, val_mask, device)
            print(
                f"Epoch {epoch:02d} | "
                f"Train loss: {loss:.4f} | "
                f"Val loss: {val_stats['loss']:.4f} | "
                f"Val acc: {val_stats['accuracy']:.4f}"
            )

        test_stats = evaluate(model, criterion, data, test_mask, device)
        print(f"\nTest accuracy ({arch.upper()}): {test_stats['accuracy']:.4f}")
