from src.run import run_experiment
from src.utils import str2bool
import argparse
import logging
from src.gtda_funcs import plot_graph
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def main(args):
    logging.info(
        "Running experiment with data from experiment run {}".format(
            args.experiment_path
        )
    )
    df = run_experiment(
        experiment_path=args.experiment_path,
        plotting=args.plotting,
        seed=int(args.seed),
    )
    df.to_csv("res/results.csv", index=False)
    df = pd.read_csv("res/results.csv")
    for method in df["method"].unique().tolist():
        if method != "entropy":
            best_res = df.loc[
                (df["method"] == method)
                & (
                    df["avg_diff_classes"]
                    == df.loc[df["method"] == method, "avg_diff_classes"].min()
                )
            ]
            for i, row in best_res.iterrows():
                plot_graph(
                    experiment_path=args.experiment_path,
                    kpi=row["avg_diff_classes"],
                    method=row["method"],
                    overlap_frac=row["overlap_frac"],
                    n_intervals=row["n_intervals"],
                    seed=int(args.seed),
                )

    logging.info("----- AVERAGE DIFF CLASSES -----")
    for method in df["method"].unique().tolist():
        logging.info(method)
        logging.info(df.loc[df["method"] == method, "avg_diff_classes"].mean())

    legend = ["PCA", "SVD", "Kernel density", "Eccentricity", "T-SNE", "TAE"]

    for method in df["method"].unique().tolist():
        if method != "entropy":
            sns.distplot(df.loc[df["method"] == method, "avg_diff_classes"], hist=False)
            plt.xlabel("Distribution of average different classes ")
            plt.legend(legend)
            plt.savefig("res/dists.png")


parser = argparse.ArgumentParser()


parser.add_argument(
    "--experiment_path",
    default="experimentation/experiment_spheres",
    help="Path to the experiments run for the topological autoencoders.",
)
parser.add_argument(
    "--plotting",
    type=str2bool,
    nargs="?",
    const=True,
    default=False,
    help="Whether one wants to plot an example",
)
parser.add_argument(
    "--logging_level",
    choices=["INFO", "DEBUG", "WARN", "ERROR", "CRITICAL"],
    default="INFO",
    help="Logging level when running experiments",
)

parser.add_argument(
    "--seed", type=int, default=125342, help="Seed used to run experiments. "
)

parser.add_argument(
    "--output_file",
    default=None,
    help="File to output results in. Default None; results will be outputted in the terminal. ",
)

args = parser.parse_args()
if args.output_file is not None:
    logging.basicConfig(
        filename=args.output_file, level=getattr(logging, args.logging_level)
    )
else:
    logging.basicConfig(level=getattr(logging, args.logging_level))

if __name__ == "__main__":
    main(args)
