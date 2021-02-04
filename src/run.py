from src.custom_filter_functions import CustomTSNE
from src.load_data import load_data, load_tsne_data
from src.gtda_funcs import piping_plotting, get_funcs
from src.evaluation_functions import get_statistics_, get_best_worst
import numpy as np
import pandas as pd
import logging


def run_experiment(experiment_path, plotting=False, funcs=None, seed=None):
    """
        Run the experiments.
    """
    logging.info("Loading data. ")
    _, df_train, df_test = load_data(path=experiment_path)
    df_train_tsne, _ = load_tsne_data(df_train=df_train, df_test=df_test, seed=seed)
    assert df_train.shape[1] == 102

    if funcs is None:
        funcs = get_funcs(df_train, df_train_tsne, seed=seed)
    assert funcs is not None, "Error, no specified filtering functions. "

    # PLOTTING ######
    if plotting is True:
        pipes = []
        for method, filter_func in funcs:
            logging.info(method)
            if isinstance(filter_func, CustomTSNE):
                _, _, stats = piping_plotting(
                    df_train_tsne[:1000], overlap_frac=0.3, filter_func=filter_func
                )
            else:
                _, _, stats = piping_plotting(
                    df_test, overlap_frac=0.3, filter_func=filter_func
                )

            pipes.append((method, stats))
    else:
        logging.info("Plotting is set to False. Not plotting. ")

    # LARGE EXPERIMENT ######

    n_bins = np.arange(5, 45, 5)
    overlap_fracs = np.arange(0.025, 0.425, 0.025)

    pipes = []
    for n_intervals in n_bins:
        for overlap_frac in overlap_fracs:
            for method, filter_func in funcs:
                logging.info(
                    """Creating mapper graph with:
                        overlap_frac={overlap_frac}
                        n_intervals={n_intervals}
                        filter_func={filter_func}\n""".format(
                        overlap_frac=overlap_frac,
                        n_intervals=int(n_intervals),
                        filter_func=str(type(filter_func)),
                    )
                )
                if isinstance(filter_func, CustomTSNE):
                    _, _, stats = piping_plotting(
                        df_train_tsne[:1000],
                        overlap_frac=float(overlap_frac),
                        n_intervals=int(n_intervals),
                        filter_func=filter_func,
                        plotting=False,
                    )
                else:
                    _, _, stats = piping_plotting(
                        df_test,
                        overlap_frac=float(overlap_frac),
                        n_intervals=int(n_intervals),
                        filter_func=filter_func,
                        plotting=False,
                    )
                pipes.append(
                    {
                        "method": method,
                        "stats": stats,
                        "n_intervals": n_intervals,
                        "overlap_frac": overlap_frac,
                    }
                )

    for i, pipe in enumerate(pipes):
        pipes[i]["statistics"] = get_statistics_(pipes[i]["stats"])
        pipes[i]["statistics"]["method"] = pipes[i]["method"]
        pipes[i]["statistics"]["n_intervals"] = pipes[i]["n_intervals"]
        pipes[i]["statistics"]["overlap_frac"] = pipes[i]["overlap_frac"]

    statistics_df = pd.DataFrame([pipe["statistics"] for pipe in pipes])
    logging.info(
        statistics_df.loc[statistics_df["method"] == "tsne"]
        .describe()
        .drop("count", axis=0)
    )
    logging.info(
        statistics_df.loc[statistics_df["method"] == "tae"]
        .describe()
        .drop("count", axis=0)
    )
    logging.info(
        statistics_df.loc[statistics_df["method"] == "pca"]
        .describe()
        .drop("count", axis=0)
    )
    logging.info(
        statistics_df.loc[statistics_df["method"] == "svd"]
        .describe()
        .drop("count", axis=0)
    )
    logging.info(
        statistics_df.loc[statistics_df["method"] == "eccentricity"]
        .describe()
        .drop("count", axis=0)
    )
    logging.info(
        statistics_df.loc[statistics_df["method"] == "entropy"]
        .describe()
        .drop("count", axis=0)
    )
    logging.info(
        statistics_df.loc[statistics_df["method"] == "gkde"]
        .describe()
        .drop("count", axis=0)
    )

    for method in statistics_df.method.unique().tolist():
        logging.info(
            "\n ------------------------- METHOD : {} ------------- \n ".format(method)
        )
        get_best_worst(statistics_df, method)
        logging.info("\n --------------------------------------------------- \n ")
    return statistics_df
