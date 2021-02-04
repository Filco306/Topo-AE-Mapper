from scipy.stats import gaussian_kde
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted
from sklearn.decomposition import PCA, TruncatedSVD as SVD
from sklearn.manifold import TSNE
from scipy.spatial.distance import cdist
from scipy.special import entr
import numpy as np
import pandas as pd
import os
from src.metadata import N_COLS_DATASETS, N_CLASSES_DATASETS
import logging

WRKDIRPATH = os.getcwd()


class CustomTSNE(BaseEstimator, TransformerMixin):
    """
        # noqa: E501
        Will account for the train data only and not account for the test data when constructing a graph
        using eccentricity.
    """

    def __init__(
        self, train_data, dataset, n_components=2, metric="euclidean", seed=125342
    ):
        assert train_data is not None
        assert (
            seed is not None
        ), "Seed is not set. Please set a seed to ensure reproducability"
        super().__init__()
        self.dataset = dataset
        self.train_data = train_data
        self.metric = metric
        self.seed = seed
        self.n_components = n_components
        self.tsne = TSNE(
            n_components=self.n_components, metric=self.metric, random_state=seed
        )
        latent_path = os.path.join(
            WRKDIRPATH, "latents", "tsne_new_latents_spheres.csv"
        )  # os.path.join(WRKDIRPATH, "latents", "tsne_train_spheres.csv")
        if os.path.exists(latent_path) == False:
            self.train_data_transformed = self.tsne.fit_transform(
                train_data[:, : N_COLS_DATASETS[self.dataset]]
            )
            pd.DataFrame(self.train_data_transformed).to_csv(
                os.path.join(WRKDIRPATH, "latents", "tsne_train_spheres.csv")
            )
        else:
            self.train_data_transformed = pd.read_csv(latent_path)
        assert self.train_data_transformed.shape == (9000, 2)

    def fit(self, X):
        check_array(X)
        self._is_fitted = True
        return self

    def transform(self, X):
        check_is_fitted(self, "_is_fitted")
        ret = self.train_data_transformed[:1000]
        assert ret.shape == (1000, 2)
        assert (
            np.unique(np.array(ret)[:, -1]).shape[0] > N_CLASSES_DATASETS[self.dataset]
        )
        return self.train_data_transformed[:1000]

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)


class CustomEccentricity(BaseEstimator, TransformerMixin):
    """
        # noqa: E501
        Will account for the train data only and not account for the test data when constructing a graph
        using eccentricity.
    """

    def __init__(
        self, train_data, dataset, exponent=2, metric="euclidean", seed=125342
    ):
        assert train_data is not None
        assert (
            seed is not None
        ), "Seed is not set. Please set a seed to ensure reproducability"
        self.train_data = train_data
        self.dataset = dataset
        self.exponent = 2
        self.seed = seed
        self.metric = metric
        self.latent_path = os.path.join(WRKDIRPATH, "latents", "ecc_spheres.csv")
        if os.path.exists(self.latent_path):
            self.latents = pd.read_csv(self.latent_path)

    def fit(self, X):
        check_array(X)
        self._is_fitted = True

    def transform(self, X):
        check_is_fitted(self, "_is_fitted")
        if os.path.exists(self.latent_path) == False:
            Xt = check_array(X)
            if X.shape[1] == N_COLS_DATASETS[self.dataset] + 1:
                X = X[:, :-1]
            # We will only check for one test point at a time. This is complicated man.

            # In each loop, we need to append one data point, calculate the eccentricity and detach

            Xt_ = np.linalg.norm(
                cdist(
                    self.train_data[:, : N_COLS_DATASETS[self.dataset]],
                    Xt[:, : N_COLS_DATASETS[self.dataset]],
                    metric=self.metric,
                ).T,
                axis=1,
                ord=2,
                keepdims=True,
            )
            pd.DataFrame(Xt_).to_csv(self.latent_path, index=False)
            assert Xt_.shape in [(1000, 1), (1000,)]
            assert np.unique(Xt_[:, -1]).shape[0] > N_CLASSES_DATASETS[self.dataset]
            return Xt_[:, 0]
        else:
            assert self.latents.shape in [(1000, 1), (1000,)]
            assert (
                np.unique(np.array(self.latents)[:, -1]).shape[0]
                > N_CLASSES_DATASETS[self.dataset]
            )
            return self.latents

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)


class CustomEntropy(BaseEstimator, TransformerMixin):
    """
        # noqa: E501
        Will account for the train data only and not account for the test data when constructing a graph
        using eccentricity.
    """

    def __init__(
        self, train_data, dataset, seed=125342, exponent=2, metric="euclidean"
    ):
        self.train_data = train_data
        self.dataset = dataset
        self.exponent = exponent
        self.metric = metric
        self.seed = seed
        self.latent_path = os.path.join(WRKDIRPATH, "latents", "entropy_spheres.csv")
        if os.path.exists(self.latent_path):
            self.latent = pd.read_csv(self.latent_path)
        else:
            self.latent = None

    def fit(self, X):
        check_array(X)
        self._is_fitted = True

    def transform(self, X):
        check_is_fitted(self, "_is_fitted")
        if X.shape[1] == N_COLS_DATASETS[self.dataset] + 1:
            X = X[:, : N_COLS_DATASETS[self.dataset]]
        if self.latent is None:
            Xt = check_array(X)

            entrs = []
            for i in range(X.shape[0]):
                entrs.append(
                    entr(
                        np.abs(
                            np.concatenate(
                                [
                                    self.train_data[:, : N_COLS_DATASETS[self.dataset]],
                                    Xt[i, : N_COLS_DATASETS[self.dataset]].reshape(
                                        (1, -1)
                                    ),
                                ],
                                axis=0,
                            )
                        )
                    ).sum(axis=1)[-1]
                )
            entrs = np.array(entrs)
            pd.DataFrame(entrs).to_csv(self.latent_path, index=False)
            assert entrs.shape in [(1000, 1), (1000,)]
            return entrs
        else:
            assert self.latent.shape in [(1000, 1), (1000,)]
            return self.latent

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)


class CustomPCA(BaseEstimator, TransformerMixin):
    def __init__(self, train_data, dataset, n_components=2, seed=125342):
        assert train_data is not None
        assert (
            seed is not None
        ), "Seed is not set. Please set a seed to ensure reproducability"
        self.train_data = train_data
        self.dataset = dataset
        self.n_components = n_components
        self.pca = PCA(n_components=2, random_state=seed)
        self.seed = seed
        self.pca.fit(self.train_data[:, : N_COLS_DATASETS[self.dataset]])
        self.latent_path = os.path.join(WRKDIRPATH, "latents", "pca_spheres.csv")
        if os.path.exists(self.latent_path):
            self.latent = pd.read_csv(self.latent_path)
        else:
            self.latent = None

    def fit(self, X):
        check_array(X)
        self._is_fitted = True
        return True

    def transform(self, X, y=None):
        check_is_fitted(self, "_is_fitted")
        if self.latent is None:
            Xt = check_array(X)
            if X.shape[1] == N_COLS_DATASETS[self.dataset] + 1:
                X = X[:, :-1]
            Xt = self.pca.transform(X)
            pd.DataFrame(Xt).to_csv(self.latent_path, index=False)
            assert Xt.shape == (1000, 2)
            return Xt
        else:
            assert self.latent.shape == (1000, 2)
            return self.latent

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X[:, : N_COLS_DATASETS[self.dataset]])


class CustomSVD(BaseEstimator, TransformerMixin):
    def __init__(self, train_data, dataset, n_components=2, seed=125342):
        assert (
            seed is not None
        ), "Seed is not set. Please set a seed to ensure reproducability"
        self.train_data = train_data
        self.dataset = dataset
        self.n_components = n_components
        self.seed = seed
        self.svd = SVD(n_components=2, random_state=seed)
        self.svd.fit(self.train_data[:, : N_COLS_DATASETS[self.dataset]])
        self.latent_path = os.path.join(WRKDIRPATH, "latents", "svd_spheres.csv")
        if os.path.exists(self.latent_path):
            self.latent = pd.read_csv(self.latent_path)
        else:
            self.latent = None

    def fit(self, X):
        check_array(X)
        self._is_fitted = True
        return True

    def transform(self, X, y=None):
        check_is_fitted(self, "_is_fitted")
        Xt = check_array(X)
        if self.latent is None:
            if X.shape[1] == N_COLS_DATASETS[self.dataset] + 1:
                X = X[:, :-1]
            Xt = self.svd.transform(X)
            pd.DataFrame(Xt).to_csv(self.latent_path, index=False)
            assert Xt.shape == (1000, 2)
            return Xt
        else:
            assert self.latent.shape == (1000, 2)
            return self.latent

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X[:, : N_COLS_DATASETS[self.dataset]])


class CustomGaussianKDE(BaseEstimator, TransformerMixin):
    def __init__(self, train_data, dataset, seed=125342):
        assert (
            seed is not None
        ), "Seed is not set. Please set a seed to ensure reproducability"
        self.train_data = train_data
        self.dataset = dataset
        self.seed = seed
        self.gkde = gaussian_kde(train_data.T)
        self.latent_path = os.path.join(WRKDIRPATH, "latents", "gkde_spheres.csv")
        if os.path.exists(self.latent_path):
            self.latent = pd.read_csv(self.latent_path)
        else:
            self.latent = None

    def fit(self, X):
        check_array(X)
        self._is_fitted = True
        return self

    def transform(self, X, y=None):
        check_is_fitted(self, "_is_fitted")
        if self.latent is None:
            Xt = check_array(X)
            if X.shape[1] == N_COLS_DATASETS[self.dataset] + 1:
                X = X[:, : N_COLS_DATASETS[self.dataset]]
            Xt = self.gkde(X.T)
            pd.DataFrame(Xt).to_csv(self.latent_path, index=False)
            assert Xt.shape in [(1000, 1), (1000, 1)]
            assert np.unique(Xt[:, -1]).shape[0] > N_CLASSES_DATASETS[self.dataset]
            return Xt
        else:
            assert self.latent.shape in [(1000, 1), (1000, 1)]
            assert (
                np.unique(np.array(self.latent)[:, -1]).shape[0]
                > N_CLASSES_DATASETS[self.dataset]
            )
            return self.latent

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)


class CustomTAE(BaseEstimator, TransformerMixin):
    def __init__(self, train_data, dataset, seed=125342):
        super().__init__()
        if seed is None:
            logging.warn("WARNING: Seed is not set! Setting seed to 125342. ")
            seed = 125342
        self.seed = seed
        self.dataset = dataset
        self.latents_path = os.path.join(WRKDIRPATH, "latents", "tae_spheres.csv")
        self.train_data = train_data
        assert os.path.exists(self.latents_path)
        self.fitted_data = pd.read_csv(self.latents_path)

    def fit(self, X):
        check_array(X)
        self._is_fitted = True
        return self

    def transform(self, X, y=None):
        if self.fitted_data.shape[1] == 3:
            ret = self.fitted_data.iloc[:, :-1]
        elif self.fitted_data.shape[1] == 2:
            ret = self.fitted_data
        else:
            raise Exception("Something weird going on. ")
        assert ret.shape == (1000, 2)
        assert (
            np.unique(np.array(ret)[:, -1]).shape[0] > N_CLASSES_DATASETS[self.dataset]
        )
        return ret

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)
