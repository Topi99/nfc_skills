import pandas as pd


def read_data(path: str) -> pd.DataFrame:
    """Returns a dataset parsed into a pd.DataFrame

    :param path: The origin path of the dataset
    """
    return pd.read_csv(path, parse_dates=["updated_at"])


def split_train_and_test_data(
    data_frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int], dict[str, int], dict[int, str]]:
    """Splits the original data into test and train datasets.

    :param data_frame: The original raw dataframe

    :returns: A tuple with the train and test dataset
    """
    users_asc = data_frame.profile_id.astype("category").cat.categories
    skills_asc = data_frame.skill_id.astype("category").cat.categories
    code2skill = {code: skill for code, skill in enumerate(skills_asc)}
    user2code = {user: code for code, user in enumerate(users_asc)}
    skill2code = {skill: code for code, skill in enumerate(skills_asc)}

    data_frame["profile_id"] = data_frame["profile_id"].astype(
        "category"
    ).cat.codes
    data_frame["skill_id"] = data_frame["skill_id"].astype(
        "category"
    ).cat.codes
    data_frame["rank_latest"] = data_frame.groupby(
        ["profile_id"],
    )["updated_at"].rank(method="first", ascending=False)

    train_data = data_frame[data_frame["rank_latest"] != 1]
    test_data = data_frame[data_frame["rank_latest"] == 1]
    del data_frame

    train_data = train_data[["profile_id", "skill_id", "level"]]
    test_data = test_data[["profile_id", "skill_id", "level"]]

    return train_data, test_data, user2code, skill2code, code2skill
