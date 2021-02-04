import pandas as pd
import os
import numpy as np
from sklearn.manifold import TSNE


def trans_(data, d):
    return np.rollaxis(data.transform(getattr(data, d).numpy()).numpy(), axis=1)


def load_data(path="experimentation/experiment_spheres"):
    if "spheres" in path.lower():
        df_latent = pd.read_csv(os.path.join(path, "latents.csv")).values
        df_raw_train = pd.read_csv(os.path.join(path, "train_dataset.csv"), header=None)
        df_raw_test = pd.read_csv(os.path.join(path, "test_dataset.csv"), header=None)
        df_train_labels = pd.read_csv(
            os.path.join(path, "train_dataset_labels.csv"), header=None
        )
        df_train_labels.columns = ["class"]
        df_test_labels = pd.read_csv(
            os.path.join(path, "test_dataset_labels.csv"), header=None
        )
        df_test_labels.columns = ["class"]
        df_train = pd.concat([df_raw_train, df_train_labels], axis=1).values
        df_test = pd.concat([df_raw_test, df_test_labels], axis=1).values
        pass
    else:
        raise FileNotFoundError("Please specify a valid dataset path. ")
    # Assert that we are using the same data for latent and test
    assert (
        df_latent.shape[0] == df_test.shape[0]
    ), "df_latent.shape : {}, df_test.shape : {}".format(df_latent.shape, df_test.shape)

    assert df_latent.shape[1] == 3, "Latent space is not 3, but {}. ".format(
        df_latent.shape[1]
    )
    assert all(
        df_latent[:, -1] == df_test[:, -1]
    ), "Error: labels of df_test and df_latent do not match! "
    return df_latent, df_train, df_test


def load_tsne_data(df_train, df_test, seed=None):
    if os.path.exists("latents/tsne_data.csv") == False:
        data = np.concatenate([df_test, df_train[:8000]], axis=0)

        latents = TSNE(random_state=seed).fit_transform(data[:, :-1])

        pd.DataFrame(latents).to_csv(
            "latents/tsne_new_latents_spheres.csv", index=False
        )

        pd.DataFrame(data).to_csv("latents/tsne_data.csv", index=False)

    df_train_tsne = pd.read_csv("latents/tsne_data.csv")
    df_latents_tsne = pd.read_csv("latents/tsne_new_latents_spheres.csv")
    assert df_train_tsne.shape == (9000, 102) and df_latents_tsne.shape == (9000, 2)
    return df_train_tsne.values, df_latents_tsne.values
