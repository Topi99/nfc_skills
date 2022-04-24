import pandas as pd


def read_data(path: str) -> pd.DataFrame:
    """Returns a dataset parsed into a pd.DataFrame

    :param path: The origin path of the dataset
    """
    return pd.read_csv(path, parse_dates=["updated_at"])


def split_train_and_test_data(
    data_frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Splits the original data into test and train datasets.

    :param data_frame: The original raw dataframe

    :returns: A tuple with the train and test dataset
    """
    data_frame["rank_latest"] = data_frame.groupby(
        ["profile_id"],
    )["updated_at"].rank(method="first", ascending=False)

    train_data = data_frame[data_frame["rank_latest"] != 1]
    test_data = data_frame[data_frame["rank_latest"] == 1]

    train_data = train_data[["profile_id", "skill_id", "level"]]
    test_data = test_data[["profile_id", "skill_id", "level"]]

    return train_data, test_data
