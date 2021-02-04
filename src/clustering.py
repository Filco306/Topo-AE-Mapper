from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted
from gtda.utils.validation import validate_params
from src.metadata import N_COLS_DATASETS, N_CLASSES_DATASETS
from sklearn.preprocessing import StandardScaler
import numpy as np

IS_IMAGE_DATA = {"mnist": True, "spheres": False, "fashionmnist": True, "cifar": True}

CURRENT_DATASET = None


class ClusterPreprocess(BaseEstimator, TransformerMixin):

    _hyperparameters = {
        "dataset": {"type": str, "in": ["spheres", "mnist", "fashionmnist", "cifar"]},
        "scaling": {"type": bool},
    }

    def __init__(self, dataset, scaling: bool = True):
        global CURRENT_DATASET
        if CURRENT_DATASET is None:
            CURRENT_DATASET = dataset
        assert CURRENT_DATASET is not None
        self.dataset = CURRENT_DATASET
        self.scaling = scaling

    def fit(self, X):
        check_array(X)
        validate_params(self.get_params(), self._hyperparameters)
        self._is_fitted = True
        return self

    def transform(self, X):
        check_is_fitted(self, "_is_fitted")

        n_cols = N_COLS_DATASETS[self.dataset]
        assert X.shape[1] in [n_cols, n_cols + 1], "Wrong shape: {}".format(X.shape)
        if X.shape[1] == (n_cols + 1):
            X_to_scale = X[:, :-1]
        elif X.shape[1] == n_cols:
            X_to_scale = X
        else:
            raise ValueError("Wrong shape of data: {}".format(X.shape))
        scaler = StandardScaler()
        ret = scaler.fit_transform(X_to_scale)
        assert ret.shape[1] == N_COLS_DATASETS[self.dataset], "{} != {}".format(
            ret.shape, N_COLS_DATASETS[self.dataset]
        )
        assert (
            np.unique(ret[:, -1]).shape[0] > N_CLASSES_DATASETS[self.dataset] + 10
        ), "np.unique : {}".format(np.unique(ret[:, -1]))
        return ret
