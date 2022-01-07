# Public repository for "Using Topological Autoencoders for global and local topology"

## Hello!

Hello! If you have come here, it means you might be interested in my extended abstract "Using topological autoencoders as a filtering function for global and local topology". I am looking for collaborators on this project to get this project into an actual paper, and if you are interested, do not hesitate to email me at c.filip.cornell@gmail.com. It would be great to have a discussion!

Questions of interest:

- How do we best validate a Mapper graph?
- If we can validate the Mapper graph, can we prove which configuration of the Mapper is the best (choice of overlap, filtering function etc.)?

Best, Filip

## General info

This repository is for the experiments in the submission accepted to the workshop *TDA and Beyond* at NeurIps2020.

The experiments were run using __Python 3.8.7__.

## Install

Use your favorite virtual environment, and run

```bash
pip install -r requirements.txt
```

## Run

To run the experiments with default settings and simply reproduce the experiments, do:

```bash
python3 index.py
```

To output the results into a file, do:

```bash
python3 index.py --output_file FILEPATH
```

This will output your log output into a file which can then be used for analysis.

The experiments will take about an hour or so to with a computer with the following specifications.
- Macbook Pro
- MacOS Catalina 10.15.7 (19H2)
- 1,4 GHz Quad-Core Intel Core i5
- 16 GB 2133 MHz LPDDR3
- Intel Iris Plus Graphics 645 1536 MB


### Arguments

There are some possible arguments to use.

| Argument            | Default                                       | Description                                                                                                                                                                                                                                                                                                                                                                                                                           |
|---------------------|-----------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `--experiment_path` | experimentation/experiment_spheres | Location of the folder with experiments containing data to use. |
| `--plotting`        | `False`                                       | Whether to plot an example during the experiment or not. The experiment will then halt and not continue until you close the plots.                                                                                                                                                                                                                                                                                                    |
| `--logging_level`   | `INFO`                                        | Options: `INFO`, `DEBUG`, `WARN`, `ERROR` and `CRITICAL`. In order to obtain the output of the results, `DEBUG` or `INFO` needs to be set (the experiments are printed on the `INFO`-level).                                                                                                                                                                                                                                          |
| `--seed`            | 125342                                        | The seed used to run the experiments. Mainly important to ensure some of the non-deterministic dimensionality reduction functions (e.g., `T-SNE`) reproduces the same results.                                                                                                                                                                                                                                                        |
| `--output_file`     | `None`                                        | File to output results in. If None, results will be outputted in the terminal.                                                                                                                                                                                                                                                                                                                                                        |

## Seeds

The seeds for these experiments are nested into the code, and the default seed is `125342`.

## Data

The dataset used can be found in the folder `experimentation/experiment_spheres`, which is a dataset consisting of 11 different 100-dimensional hyper-spheres, 10 of them encompassed by another larger one. This dataset was created using [functions available in the Topological Autoencoders repository](https://github.com/BorgwardtLab/topological-autoencoders).

### Use another dataset

To run these experiments on another dataset, your data should be divided as follows.

- `train_dataset.csv` : the training data points
- `train_dataset_labels.csv` : the manifold label groupings
- `test_dataset.csv` : the test data points
- `test_dataset_labels.csv` : the manifold label groupings of the test set
- `latents.csv` : latents of the *test* data points from inference of a topological autoencoder
- `latents/tae_spheres.csv` - tae *test* points but without the labels (`latents.csv` also contains the manifold labels).

Additionally, the `latents/` folder which will be filled up with csv-files during the experiments. It contains `tae_spheres.csv`, which is the `latents.csv` without the labels. 

## TAE-model

The Topological Autoencoder model trained is available in `models/topoAE_model.pth`. The model was trained using __python 3.8.5__ and the versions:
- `torch==1.4.0`
- `torchvision==0.5.0`
- `CUDA Version: 10.1`

The code used to train the model is available in [this Borgwardt lab repository](https://github.com/BorgwardtLab/topological-autoencoders).

## Cite

If you'd like to cite this article, please cite it using this BibTex code:

```
@inproceedings{
cornell2020using,
title={Using topological autoencoders as a filtering function for global and local topology},
author={Filip Cornell},
booktitle={NeurIPS 2020 Workshop on Topological Data Analysis and Beyond},
year={2020},
url={https://openreview.net/forum?id=0V6WLosuIfJ}
}
```

## Acknowledgements

This work was partially supported by [Wallenberg Autonomous Systems Program](https://wasp-sweden.org/) ([WASP](https://wasp-sweden.org/)).
