import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class SkillsTrainDataset(Dataset):
    """PyTorch dataset of skills for training.

    :arg dataframe: DataFrame containing the interactions.
    :arg all_skill_ids: List containing all the skill ids.
    """

    def __init__(
        self, dataframe: pd.DataFrame, all_skill_ids: list[str]
    ) -> None:
        self.users, self.items, self.labels = self.get_dataset(
            dataframe, all_skill_ids
        )

    def __len__(self) -> int:
        return len(self.users)

    def __getitem__(self, idx) -> tuple[int, int, int]:
        return self.users[idx], self.items[idx], self.labels[idx]

    @staticmethod
    def get_dataset(
        dataframe: pd.DataFrame, all_skill_ids: list[str]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Transforms a given dataframe to tuple of useful pytorch Tensors.

        :param dataframe: DataFrame containing the interactions.
        :param all_skill_ids: List containing all the skill ids.
        :return: A tuple of pytorch Tensors with users, items and labels
        """

        users, items, labels = [], [], []
        user_item_set = set(
            zip(dataframe["profile_id"], dataframe["skill_id"])
        )

        num_negatives = 4
        for u, i in user_item_set:
            users.append(u)
            items.append(i)
            labels.append(1)

            for _ in range(num_negatives):
                negative_item = np.random.choice(all_skill_ids)
                while (u, negative_item) in user_item_set:
                    negative_item = np.random.choice(all_skill_ids)
                users.append(u)
                items.append(negative_item)
                labels.append(0)

        return torch.tensor(users), torch.tensor(items), torch.tensor(labels)
