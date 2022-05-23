import pickle
from typing import TypedDict

import numpy as np
import pandas as pd

import data

DATA_ORIGIN_PATH = "profile_skills.csv"
EXPORT_DATA_PATH = "preprocessed_data.pickle"


class ExportData(TypedDict):
    train_data: pd.DataFrame
    num_users: int
    num_items: int
    all_skill_ids: np.ndarray
    user_interacted_items: dict[str, str]
    test_user_item_set: set[tuple[str, str]]
    user2code: dict[str, int]
    skill2code: dict[str, int]
    code2skill: dict[int, str]


def get_preprocessed_data() -> ExportData:
    with open(EXPORT_DATA_PATH, "rb") as handle:
        preprocessed_data: ExportData = pickle.load(handle)
        return preprocessed_data


def main() -> None:
    df = data.read_data(DATA_ORIGIN_PATH)
    (
        train_data,
        test_data,
        user2code,
        skill2code,
        code2skill,
    ) = data.split_train_and_test_data(df)
    num_users = df["profile_id"].max() + 1
    num_items = df["skill_id"].max() + 1
    all_skill_ids = df["skill_id"].unique()
    user_interacted_items = df.groupby(
        "profile_id"
    )["skill_id"].apply(list).to_dict()
    test_user_item_set = set(
        zip(test_data["profile_id"], test_data["skill_id"])
    )
    export_data: ExportData = {
        "train_data": train_data,
        "num_users": num_users,
        "num_items": num_items,
        "all_skill_ids": all_skill_ids,
        "user_interacted_items": user_interacted_items,
        "test_user_item_set": test_user_item_set,
        "user2code": user2code,
        "skill2code": skill2code,
        "code2skill": code2skill,
    }

    with open(EXPORT_DATA_PATH, "wb") as handle:
        pickle.dump(export_data, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
