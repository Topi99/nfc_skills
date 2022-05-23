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


class SimpleNCF(pl.LightningModule):
    """Simple Neural Collaborative Filtering with MLP"""

    def __init__(
        self,
        num_users: int,
        num_items: int,
        ratings: pd.DataFrame,
        all_skill_ids: np.ndarray,
    ) -> None:
        super().__init__()
        self.user_embedding = nn.Embedding(num_embeddings=num_users, embedding_dim=8)
        self.item_embedding = nn.Embedding(num_embeddings=num_items, embedding_dim=8)
        self.fc1 = nn.Linear(in_features=16, out_features=32)
        self.fc2 = nn.Linear(in_features=32, out_features=16)
        self.output = nn.Linear(in_features=16, out_features=1)
        self.ratings = ratings
        self.all_skill_ids = all_skill_ids

    def forward(self, user_input: Any, item_input: Any) -> Any:
        user_embedded = self.user_embedding(user_input)
        item_embedded = self.item_embedding(item_input)

        vector = torch.cat([user_embedded, item_embedded], dim=-1)

        vector = nn.ReLU()(self.fc1(vector))
        vector = nn.ReLU()(self.fc2(vector))
        # vector = nn.ReLU()(self.fc3(vector))

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
