from typing import Any

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.types import STEP_OUTPUT, TRAIN_DATALOADERS
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from . import SkillsTrainDataset


class GMFNCF(pl.LightningModule):
    """Neural Collaborative Filtering with Generalized Matrix Factorization"""

    def __init__(
        self,
        num_users: int,
        num_items: int,
        ratings: pd.DataFrame,
        all_skill_ids: np.ndarray,
    ) -> None:
        super().__init__()
        self.user_embedding_mlp = nn.Embedding(
            num_embeddings=num_users, embedding_dim=8
        )
        self.item_embedding_mlp = nn.Embedding(
            num_embeddings=num_items, embedding_dim=8
        )
        self.user_embedding_gmf = nn.Embedding(
            num_embeddings=num_users, embedding_dim=8
        )
        self.item_embedding_gmf = nn.Embedding(
            num_embeddings=num_items, embedding_dim=8
        )
        self.fc1 = nn.Linear(in_features=16, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=32)
        self.output = nn.Linear(in_features=40, out_features=1)
        self.ratings = ratings
        self.all_skill_ids = all_skill_ids

    def forward(self, user_input: Any, item_input: Any) -> Any:
        user_embedded_mlp = self.user_embedding_mlp(user_input)
        item_embedded_mlp = self.item_embedding_mlp(item_input)

        mlp_vector = torch.cat([user_embedded_mlp, item_embedded_mlp], dim=-1)

        mlp_vector = nn.ReLU()(self.fc1(mlp_vector))
        mlp_vector = nn.ReLU()(self.fc2(mlp_vector))

        user_embedded_gmf = self.user_embedding_gmf(user_input)
        item_embedded_gmf = self.item_embedding_gmf(item_input)

        gmf_vector = user_embedded_gmf * item_embedded_gmf

        vector = torch.cat([gmf_vector, mlp_vector], dim=1)

        pred = nn.Sigmoid()(self.output(vector))
        return pred

    def training_step(self, batch: Any, batch_idx: Any) -> STEP_OUTPUT:
        user_input, item_input, labels = batch
        predicted_labels = self(user_input, item_input)
        loss = nn.BCELoss()(predicted_labels, labels.view(-1, 1).float())
        self.log("loss", loss, on_step=True, on_epoch=True)
        return loss

    def configure_optimizers(self) -> Optimizer:
        return torch.optim.Adam(self.parameters())

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            SkillsTrainDataset(self.ratings, self.all_skill_ids),
            batch_size=512,
            num_workers=4,
        )
