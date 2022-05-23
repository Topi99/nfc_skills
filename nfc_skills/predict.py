import sys

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch

import data
from preprocess_data import get_preprocessed_data
from train import SIMPLE_MODEL_PATH, HYBRID_MODEL_PATH


def _get_predictions(
    user_id: str,
    user2code: dict[str, int],
    code2skill: dict[int, str],
    model: pl.LightningModule,
    all_skill_ids: np.ndarray,
    user_interacted_items: dict[str, str],
) -> list[str]:
    user_id = user2code[user_id]
    interacted_items = user_interacted_items[user_id]
    not_interacted_items = list(set(all_skill_ids) - set(interacted_items))
    items_len = len(not_interacted_items)
    predicted_labels = np.squeeze(
        model(
            torch.tensor([user_id] * items_len),
            torch.tensor(not_interacted_items).to(torch.int64)
        ).detach().numpy(),
    )

    return [
        code2skill[not_interacted_items[idx]]
        for idx in np.argsort(predicted_labels)[::-1][0:10].tolist()
    ]


def main(user_id: str) -> None:
    preprocessed_data = get_preprocessed_data()
    num_users = preprocessed_data["num_users"]
    num_items = preprocessed_data["num_items"]
    all_skill_ids = preprocessed_data["all_skill_ids"]
    user2code = preprocessed_data["user2code"]
    code2skill = preprocessed_data["code2skill"]
    user_interacted_items = preprocessed_data["user_interacted_items"]

    simple_model = data.SimpleNCF(num_users, num_items, pd.DataFrame([]), all_skill_ids)
    simple_model.load_state_dict(torch.load(SIMPLE_MODEL_PATH))
    simple_model.eval()

    hybrid_model = data.GMFNCF(num_users, num_items, pd.DataFrame([]), all_skill_ids)
    hybrid_model.load_state_dict(torch.load(HYBRID_MODEL_PATH))
    hybrid_model.eval()

    print(
        "Predictions by SimpleNCF:",
        _get_predictions(
            user_id,
            user2code,
            code2skill,
            simple_model,
            all_skill_ids,
            user_interacted_items,
        )
    )

    print(
        "Predictions by GMFNCF:",
        _get_predictions(
            user_id,
            user2code,
            code2skill,
            hybrid_model,
            all_skill_ids,
            user_interacted_items,
        )
    )


if __name__ == "__main__":
    main(sys.argv[1])
