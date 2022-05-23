import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl

import data
from train import SIMPLE_MODEL_PATH, HYBRID_MODEL_PATH
from preprocess_data import get_preprocessed_data


def hit_ratio_10(
    all_skill_ids: np.ndarray,
    user_interacted_items: dict[str, str],
    test_user_item_set: set[tuple[str, str]],
    model: pl.LightningModule,
) -> float:
    hits: list[int] = []
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
    preprocessed_data = get_preprocessed_data()
    num_users = preprocessed_data["num_users"]
    num_items = preprocessed_data["num_items"]
    all_skill_ids = preprocessed_data["all_skill_ids"]

    simple_model = data.SimpleNCF(num_users, num_items, pd.DataFrame([]), all_skill_ids)
    simple_model.load_state_dict(torch.load(SIMPLE_MODEL_PATH))
    simple_model.eval()

    hybrid_model = data.GMFNCF(num_users, num_items, pd.DataFrame([]), all_skill_ids)
    hybrid_model.load_state_dict(torch.load(HYBRID_MODEL_PATH))
    hybrid_model.eval()

    user_interacted_items = preprocessed_data["user_interacted_items"]
    test_user_item_set = preprocessed_data["test_user_item_set"]

    simple_ratio = hit_ratio_10(
        all_skill_ids, user_interacted_items, test_user_item_set, simple_model
    )
    hybrid_ratio = hit_ratio_10(
        all_skill_ids, user_interacted_items, test_user_item_set, hybrid_model
    )

    print(f"Hit Ratio 10 SimpleNCF = {round(simple_ratio, 4)}")
    print(f"Hit Ratio 10 GMFNCF = {round(hybrid_ratio, 4)}")


if __name__ == "__main__":
    main()
