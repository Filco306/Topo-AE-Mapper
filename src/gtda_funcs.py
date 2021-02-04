from src.custom_filter_functions import (
    CustomTSNE,
    CustomGaussianKDE,
    CustomPCA,
    CustomSVD,
    CustomEccentricity,
    CustomEntropy,
    CustomTAE,
)
from gtda.mapper import (
    CubicalCover,
    OneDimensionalCover,
    make_mapper_pipeline,
    plot_static_mapper_graph,
)
from sklearn.cluster import DBSCAN
import numpy as np
from src.evaluation_functions import get_statistics
from src.clustering import ClusterPreprocess
from src.load_data import load_data
from src.metadata import N_COLS_DATASETS, N_CLASSES_DATASETS
from typing import Union
import pandas as pd
from src.load_data import load_tsne_data


def assert_shape(data, dataset="spheres"):
    if data.shape[1] == (N_COLS_DATASETS[dataset] + 1):
        input_ = data[:, :-1]
    elif data.shape[1] == N_COLS_DATASETS[dataset]:
        input_ = data[:, :-1]
    else:
        raise Exception("Wrong shape of data! ")
    # Ensure we do not have a data leak.
    assert (
        np.unique(input_[:, -1]).shape[0] > N_CLASSES_DATASETS[dataset]
        and input_.shape[1] == N_COLS_DATASETS[dataset]
    )
    return input_


def get_funcs(
    df_train: Union[pd.DataFrame, np.ndarray],
    df_train_tsne: Union[pd.DataFrame, np.ndarray],
    dataset: str = "spheres",
    seed: int = None,
):
    assert df_train.shape == (9000, N_COLS_DATASETS[dataset] + 1), "{}".format(
        df_train.shape
    )
    assert df_train_tsne.shape == (9000, N_COLS_DATASETS[dataset] + 1), "{}".format(
        df_train_tsne.shape
    )
    return [
        ("pca", CustomPCA(train_data=df_train[:, :-1], dataset="spheres", seed=seed),),
        ("svd", CustomSVD(train_data=df_train[:, :-1], dataset="spheres", seed=seed),),
        (
            "gkde",
            CustomGaussianKDE(
                train_data=df_train[:, :-1], dataset="spheres", seed=seed
            ),
        ),
        (
            "ecc",
            CustomEccentricity(
                train_data=df_train[:, :-1], dataset="spheres", seed=seed
            ),
        ),
        (
            "entropy",
            CustomEntropy(train_data=df_train[:, :-1], dataset="spheres", seed=seed),
        ),
        (
            "tsne",
            CustomTSNE(train_data=df_train_tsne[:, :-1], dataset="spheres", seed=seed),
        ),
        ("tae", CustomTAE(train_data=df_train[:, :-1], dataset="spheres", seed=seed),),
    ]


def plot_graph(
    experiment_path,
    kpi,
    method: str,
    overlap_frac: float,
    n_intervals: int,
    funcs: list = None,
    seed=None,
):
    # experiment_path = "experimentation/experiment_spheres"
    _, df_train, df_test = load_data(path=experiment_path)
    df_train_tsne, _ = load_tsne_data(df_train=df_train, df_test=df_test, seed=seed)
    if funcs is None:
        funcs = get_funcs(df_train=df_train, df_train_tsne=df_train_tsne, seed=seed)
    if method == "tsne":
        data_ = df_train_tsne[:1000]
    else:
        data_ = df_test
    clusterer = DBSCAN(4)
    for func in funcs:
        if method == func[0]:
            filter_func = func[1]
            break
    s_ = filter_func.fit_transform(data_)
    if len(s_.shape) == 1 or s_.shape[1] == 1:
        cover = OneDimensionalCover(n_intervals=n_intervals, overlap_frac=overlap_frac)
    else:
        cover = CubicalCover(n_intervals=n_intervals, overlap_frac=overlap_frac)
    c_p = ClusterPreprocess(dataset="spheres", scaling=False)

    pipe = make_mapper_pipeline(
        filter_func=filter_func,
        cover=cover,
        clustering_preprocessing=c_p,
        clusterer=clusterer,
        verbose=False,
        n_jobs=6,
    )

    fig = plot_static_mapper_graph(pipe, data_, color_variable=(data_.shape[1] - 1))
    fig.update_layout(title="{} : {}".format(method, kpi))
    fig.show(config={"scrollZoom": True})


def piping_plotting(
    data, filter_func, overlap_frac=0.3, n_intervals=10, plotting=True,
):
    # Define filter function – can be any scikit-learn transformer
    # Define cover
    assert data.shape[0] == 1000, "{} != 1000".format(data.shape)
    data_ = data
    s_ = filter_func.fit_transform(data_)[:10]
    if len(s_.shape) == 1 or s_.shape[1] == 1:
        cover = OneDimensionalCover(n_intervals=n_intervals, overlap_frac=overlap_frac)
    else:
        cover = CubicalCover(n_intervals=n_intervals, overlap_frac=overlap_frac)
    # Choose clustering algorithm – default is DBSCAN
    clusterer = DBSCAN(4)

    # Configure parallelism of clustering step
    n_jobs = 6
    c_p = ClusterPreprocess(dataset="spheres", scaling=False)
    pipe = make_mapper_pipeline(
        filter_func=filter_func,
        cover=cover,
        clustering_preprocessing=c_p,
        clusterer=clusterer,
        verbose=False,
        n_jobs=n_jobs,
    )
    input_ = assert_shape(data=data_, dataset="spheres")
    graph = pipe.fit_transform(input_)

    node_elements = graph.vs["node_elements"]
    stats = get_statistics(data_[:, -1], node_elements)
    if plotting is True:
        fig = plot_static_mapper_graph(pipe, data_, color_by_columns_dropdown=True)
        fig.update_layout(title="method : {}".format(type(filter_func)))
        fig.show(config={"scrollZoom": True})
    return pipe, graph, stats
