import random
from typing import List

import numpy as np
import pandas as pd
import torch

import data
import pytorch_lightning as pl

DATA_ORIGIN_PATH = "profile_skills.csv"


def hit_ratio_10(
    all_data: pd.DataFrame,
    test_data: pd.DataFrame,
    all_skill_ids: np.ndarray,
    model: pl.LightningModule,
) -> float:
    test_user_item_set = set(
        zip(test_data["profile_id"], test_data["skill_id"])
    )
    user_interacted_items = all_data.groupby(
        "profile_id"
    )["skill_id"].apply(list).to_dict()

    hits: List[int] = []
    for (u, i) in test_user_item_set:
        interacted_items = user_interacted_items[u]
        not_interacted_items = set(all_skill_ids) - set(interacted_items)
        selected_not_interacted = list(
            np.random.choice(list(not_interacted_items), 99)
        )
        test_items = selected_not_interacted + [i]

        predicted_labels = np.squeeze(
            model(
                torch.tensor([u] * 100),
                torch.tensor(test_items)
            ).detach().numpy(),
        )

        top_10_items = [
            test_items[idx]
            for idx in np.argsort(predicted_labels)[::-1][0:10].tolist()
        ]

        if i in top_10_items:
            hits.append(1)
        else:
            hits.append(0)

    return np.average(hits)


def main() -> None:
    df = data.read_data(DATA_ORIGIN_PATH)
    train_data, test_data = data.split_train_and_test_data(df)
    num_users = df["profile_id"].max()+1
    num_items = df["skill_id"].max()+1
    all_skill_ids = df["skill_id"].unique()
    model = data.SimpleNFC(num_users, num_items, train_data, all_skill_ids)

    trainer = pl.Trainer(
        max_epochs=100,
        reload_dataloaders_every_n_epochs=True,
        progress_bar_refresh_rate=50,
        logger=False,
        checkpoint_callback=False,
    )

    trainer.fit(model)

    print(hit_ratio_10(
        df, test_data, all_skill_ids, model
    ))


if __name__ == "__main__":
    main()
