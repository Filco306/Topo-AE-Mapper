from collections import Counter
from statistics import median
import logging
from typing import List
import numpy as np
import pandas as pd


def get_statistics(labels: np.ndarray, node_metadata: List[np.ndarray]):
    nodes = []
    for arr in node_metadata:
        c = Counter()
        for r in arr:
            c[labels[r]] += 1
        nodes.append(c)
    return nodes


def get_statistics_(counters: List[Counter]):
    n_different_classes = [len(x.keys()) for x in counters]
    tot_per_counter = [sum(x.values()) for x in counters]
    avg_shares = []
    for i, c in enumerate(counters):
        avg_share = 0
        for key in c:
            avg_share += c[key] / tot_per_counter[i]
        avg_shares.append(avg_share)
    d = {}
    d["n_nodes"] = len(counters)
    d["min_diff_classes"] = min(n_different_classes)
    d["max_diff_classes"] = max(n_different_classes)
    d["median_diff_classes"] = median(n_different_classes)
    d["avg_diff_classes"] = sum(n_different_classes) / len(n_different_classes)
    d["min_avg_share"] = min(avg_shares)
    d["median_avg_shae"] = median(avg_shares)
    d["min_avg_share"] = max(avg_shares)
    d["avg_avg_share"] = sum(avg_shares) / len(n_different_classes)
    return d


def get_best_worst(df: pd.DataFrame, method: str):
    desc = df.loc[df["method"] == method].describe()
    logging.info("\n ---------------- BEST ----------------- \n ")
    logging.info(
        df.loc[
            (df["avg_diff_classes"] == desc.loc["min", "avg_diff_classes"])
            & (df["method"] == method)
        ].head(1)
    )
    logging.info("\n ---------------- WORST ----------------- \n ")
    logging.info(
        df.loc[
            (df["avg_diff_classes"] == desc.loc["max", "avg_diff_classes"])
            & (df["method"] == method)
        ].head(1)
    )

    logging.info(
        "N maps with best result: {}".format(
            df.loc[
                (df["avg_diff_classes"] == desc.loc["min", "avg_diff_classes"])
                & (df["method"] == method)
            ].shape[0]
        )
    )
    logging.info(
        "N maps with worst result: {}".format(
            df.loc[
                (df["avg_diff_classes"] == desc.loc["max", "avg_diff_classes"])
                & (df["method"] == method)
            ].shape[0]
        )
    )
