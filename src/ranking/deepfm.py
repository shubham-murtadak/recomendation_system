"""
DeepFM: Deep Factorization Machine for Recommendation Ranking
============================================================
Implements DeepFM (Guo et al., 2017) in PyTorch.

Architecture:
    Input Features (User, Item, Interaction, Context)
               │
       ┌───────┴───────────────┐
       ▼                       ▼
  FM Component           Deep Component
  ├── Order-1 (Linear)    └── Embedding Layer (shared)
  └── Order-2 (Pairwise)      └── Multi-Layer Perceptron (MLP)
       │                       │
       └───────┬───────────────┘
               ▼
      Sigmoid Output (Click / Interaction Probability)

Includes a scikit-learn compatible DeepFMRanker wrapper with
standard scaling, class-weighted BCE loss, and fast vectorized inference.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, ClassifierMixin
from pathlib import Path
import joblib


class DeepFMNetwork(nn.Module):
    """
    PyTorch DeepFM Neural Network.
    Combines:
        1. Linear order-1 part: w^T x
        2. Factorization Machine order-2 part: 0.5 * sum((sum v_i x_i)^2 - sum v_i^2 x_i^2)
        3. Deep multi-layer perceptron for high-order non-linear feature interactions
    """

    def __init__(
        self,
        num_features: int,
        embedding_dim: int = 16,
        hidden_dims: list = [128, 64, 32],
        dropout: float = 0.2,
    ):
        super().__init__()
        self.num_features = num_features
        self.embedding_dim = embedding_dim

        # 1. First-order linear component
        self.linear_bias = nn.Parameter(torch.zeros(1))
        self.linear = nn.Linear(num_features, 1, bias=False)

        # 2. Second-order FM embedding vectors (shared with Deep component)
        self.embeddings = nn.Parameter(
            torch.randn(num_features, embedding_dim) * 0.01
        )

        # 3. Deep Component (MLP)
        layers = []
        input_dim = num_features * embedding_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_dim = h_dim
        layers.append(nn.Linear(input_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Tensor of shape (batch_size, num_features)
        Returns:
            logits: Tensor of shape (batch_size,)
        """
        batch_size = x.size(0)

        # --- 1. Linear Order-1 Part ---
        linear_part = self.linear(x)  # (batch_size, 1)

        # --- 2. Second-Order FM Part ---
        # Scale each feature's embedding vector by feature value x_i
        # x: (B, num_features, 1), embeddings: (1, num_features, embedding_dim)
        x_embed = x.unsqueeze(-1) * self.embeddings.unsqueeze(0)  # (B, num_features, D)

        # Algebraic FM trick: sum(v_i x_i)^2 - sum(v_i^2 x_i^2)
        sum_embed = torch.sum(x_embed, dim=1)  # (B, D)
        sum_sq = sum_embed ** 2
        sq_sum = torch.sum(x_embed ** 2, dim=1)  # (B, D)
        fm_part = 0.5 * torch.sum(sum_sq - sq_sum, dim=1, keepdim=True)  # (B, 1)

        # --- 3. Deep Component ---
        deep_input = x_embed.reshape(batch_size, -1)  # (B, num_features * D)
        deep_part = self.mlp(deep_input)  # (B, 1)

        # Combined raw logits
        logits = self.linear_bias + linear_part + fm_part + deep_part
        return logits.squeeze(-1)


class DeepFMRanker(BaseEstimator, ClassifierMixin):
    """
    Scikit-learn compatible DeepFM Ranker.
    Wraps standard scaling, PyTorch training loop, and vectorized inference.
    """

    def __init__(
        self,
        embedding_dim: int = 16,
        hidden_dims: list = [128, 64, 32],
        dropout: float = 0.2,
        lr: float = 0.001,
        weight_decay: float = 1e-4,
        epochs: int = 25,
        batch_size: int = 128,
        random_state: int = 42,
    ):
        self.embedding_dim = embedding_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state

        self.scaler = StandardScaler()
        self.model = None
        self.classes_ = np.array([0, 1])

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit DeepFM on training dataset."""
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        X_scaled = self.scaler.fit_transform(X).astype(np.float32)
        y_arr = y.astype(np.float32)

        num_features = X_scaled.shape[1]
        self.model = DeepFMNetwork(
            num_features=num_features,
            embedding_dim=self.embedding_dim,
            hidden_dims=self.hidden_dims,
            dropout=self.dropout,
        )

        # Imbalance weighting for BCEWithLogitsLoss
        neg_count = float((y_arr == 0).sum())
        pos_count = float((y_arr == 1).sum())
        pos_weight = torch.tensor([neg_count / max(pos_count, 1.0)], dtype=torch.float32)

        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        dataset = TensorDataset(
            torch.from_numpy(X_scaled),
            torch.from_numpy(y_arr),
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0.0
            for bx, by in loader:
                optimizer.zero_grad()
                logits = self.model(bx)
                loss = criterion(logits, by)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

        self.model.eval()
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities [P(0), P(1)]."""
        if self.model is None:
            raise RuntimeError("DeepFMRanker is not fitted yet.")

        X_scaled = self.scaler.transform(X).astype(np.float32)
        self.model.eval()

        probs = []
        batch_size = 32768
        with torch.no_grad():
            for i in range(0, len(X_scaled), batch_size):
                bx = torch.from_numpy(X_scaled[i : i + batch_size])
                logits = self.model(bx)
                prob_1 = torch.sigmoid(logits).cpu().numpy()
                probs.append(prob_1)

        p1 = np.concatenate(probs)
        p0 = 1.0 - p1
        return np.column_stack([p0, p1])

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute training accuracy."""
        preds = (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
        return float((preds == y).mean())

    def save(self, filepath: str or Path):
        """Save the fitted model and scaler."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "scaler": self.scaler,
                "model_state": self.model.state_dict(),
                "config": {
                    "num_features": self.model.num_features,
                    "embedding_dim": self.embedding_dim,
                    "hidden_dims": self.hidden_dims,
                    "dropout": self.dropout,
                },
            },
            filepath,
        )

    @classmethod
    def load(cls, filepath: str or Path) -> "DeepFMRanker":
        """Load a saved DeepFMRanker."""
        data = joblib.load(filepath)
        instance = cls(
            embedding_dim=data["config"]["embedding_dim"],
            hidden_dims=data["config"]["hidden_dims"],
            dropout=data["config"]["dropout"],
        )
        instance.scaler = data["scaler"]
        instance.model = DeepFMNetwork(
            num_features=data["config"]["num_features"],
            embedding_dim=data["config"]["embedding_dim"],
            hidden_dims=data["config"]["hidden_dims"],
            dropout=data["config"]["dropout"],
        )
        instance.model.load_state_dict(data["model_state"])
        instance.model.eval()
        return instance
