import torch

import data
import pytorch_lightning as pl
from preprocess_data import get_preprocessed_data
import matplotlib.pyplot as plt

DATA_ORIGIN_PATH = "profile_skills.csv"
SIMPLE_MODEL_PATH = "simple_trained.model"
HYBRID_MODEL_PATH = "hybrid_trained.model"


def main() -> None:
    preprocessed_data = get_preprocessed_data()
    num_users = preprocessed_data["num_users"]
    num_items = preprocessed_data["num_items"]
    all_skill_ids = preprocessed_data["all_skill_ids"]
    train_data = preprocessed_data["train_data"]

    model_hybrid = data.GMFNCF(num_users, num_items, train_data, all_skill_ids)
    model_simple = data.SimpleNCF(num_users, num_items, train_data, all_skill_ids)
    logger_simple = data.MetricLogger()
    logger_hybrid = data.MetricLogger()

    trainer = pl.Trainer(
        max_epochs=100,
        reload_dataloaders_every_n_epochs=True,
        progress_bar_refresh_rate=50,
        logger=logger_simple,
        checkpoint_callback=False,
    )
    trainer.fit(model_simple)

    # Export the model
    torch.save(model_simple.state_dict(), SIMPLE_MODEL_PATH)

    trainer = pl.Trainer(
        max_epochs=100,
        reload_dataloaders_every_n_epochs=True,
        progress_bar_refresh_rate=50,
        logger=logger_hybrid,
        checkpoint_callback=False,
    )
    trainer.fit(model_hybrid)

    simple_x = [i for i, _ in enumerate(logger_simple.history["loss_step"])]
    hybrid_x = [i for i, _ in enumerate(logger_hybrid.history["loss_step"])]
    plt.plot(simple_x, logger_simple.history["loss_step"], label="SimpleNFC")
    plt.plot(hybrid_x, logger_hybrid.history["loss_step"], label="GMFNCF")
    plt.legend()
    plt.savefig("loss.png")

    # Export the model
    torch.save(model_hybrid.state_dict(), HYBRID_MODEL_PATH)

    plt.savefig("loss.png")
    print(f"Model saved at ./{SIMPLE_MODEL_PATH} and ./{HYBRID_MODEL_PATH}")


if __name__ == "__main__":
    main()
