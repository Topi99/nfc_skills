import data

DATA_ORIGIN_PATH = "profile_skills.csv"


def main() -> None:
    df = data.read_data(DATA_ORIGIN_PATH)
    train_data, test_data = data.split_train_and_test_data(df)


if __name__ == "__main__":
    main()
