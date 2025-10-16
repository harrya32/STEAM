"""Real data generic vs. STEAM comparisons."""

import argparse
import os
import random
from typing import Callable, Dict, Iterable, Tuple

import pandas as pd

from src.generation import generate_sequentially, generate_standard
from src.metrics import evaluate_average_u_pehe, evaluate_c, evaluate_f, evaluate_jsd


DatasetLoader = Callable[[], Tuple[pd.DataFrame, str, str, bool]]


def compare_steam_and_generic(
    real: pd.DataFrame,
    gen: str,
    treatment_col: str,
    outcome_col: str,
    n_iter: int,
    private: bool = False,
    epsilon=None,
    delta=None,
    binary_y: bool = False,
    classifier: str = "logistic",
    save: bool = False,
    fp: str = "",
):
    """
    Compares the performance of generic and STEAM synthetic data generation methods.

    This function generates synthetic data using a generic generative model, and its STEAM counterpart. 
    It then evaluates the data using our metrics, and aggregates the results over different seeds. Optionally, results can be saved to a CSV file.

    Parameters
    ----------
    real : pandas.DataFrame
        The real dataset.
    gen : str
        The name of the generic generative model.
    treatment_col : str
        The name of the treatment column.
    outcome_col : str
        The name of the outcome column.
    n_iter : int
        The number of iterations to run.
    private : bool, optional
        Indicates whether to generate using differential privacy. Default is `False`.
    epsilon : list of float, optional
        Privacy budget for each STEAM component, if using differential privacy. If `private=True`, this should be provided as a list. The sum of the list will be the budget for the generic model. Default is `None`.
    delta : list of float, optional
        Privacy slack parameters for each STEAM component, if using differential privacy. If `private=True`, this should be provided as a list. The sum of the list will be the slack param for the generic model. Default is `None`.
    binary_y : bool, optional
        Indicates whether the outcome variable (`outcome_col`) is binary. Default is `False`.
    classifier : str, optional
        The classifier used for JSD_\pi evaluation. Options are `'logistic'` (logistic regression) or `'rf'` (Random Forest). Default is `'logistic'`.
    save : bool, optional
        Indicates whether to save the results to a CSV file. Default is `False`.
    fp : str, optional
        The results file path, if `save=True`. Default is an empty string.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the evaluation results for both the generic and STEAM methods across the specified metrics.
        The DataFrame has the following columns:
        - `'method'`: The generation method (`'generic'` or `'STEAM'`).
        - `'p_a,x'`: Covariate alpha precision metric.
        - `'r_b,x'`: Covariate beta recall metric.
        - `'jsd_pi'`: JSD_\pi metric.
        - `'u_pehe'`: U_PEHE metric.
    """

    results = pd.DataFrame(columns=["method", "p_a,x", "r_b,x", "jsd_pi", "u_pehe"])
    for _ in range(n_iter):
        standard = generate_standard(real, gen, private=private, epsilon=epsilon, delta=delta)
        if standard[treatment_col].nunique() == 1 and len(standard) >= 2:
            indices_to_flip = random.sample(range(len(standard)), 2)
            standard.loc[indices_to_flip, treatment_col] = 1 - standard.loc[indices_to_flip, treatment_col]

        steam = generate_sequentially(
            real,
            gen,
            treatment_col,
            outcome_col,
            private=private,
            epsilon=epsilon,
            delta=delta,
            binary_y=binary_y,
        )
        n_units = len(real.drop([treatment_col, outcome_col], axis=1).columns)

        results.loc[len(results)] = [
            "generic",
            evaluate_f(real, standard, treatment_col, outcome_col),
            evaluate_c(real, standard, treatment_col, outcome_col),
            evaluate_jsd(real, standard, treatment_col, outcome_col, classifier),
            evaluate_average_u_pehe(real, standard, treatment_col, outcome_col, n_units, binary_y),
        ]
        results.loc[len(results)] = [
            "STEAM",
            evaluate_f(real, steam, treatment_col, outcome_col),
            evaluate_c(real, steam, treatment_col, outcome_col),
            evaluate_jsd(real, steam, treatment_col, outcome_col, classifier),
            evaluate_average_u_pehe(real, steam, treatment_col, outcome_col, n_units, binary_y),
        ]

        if save and fp:
            results.to_csv(fp, index=False)

    return results


def load_actg() -> Tuple[pd.DataFrame, str, str, bool]:
    actg = pd.read_csv("../data/aids_preprocessed.csv")
    return actg, "t", "y", False


def load_ihdp() -> Tuple[pd.DataFrame, str, str, bool]:
    ihdp_full = pd.read_csv("../data/ihdp.csv")
    ihdp = ihdp_full.drop(["y_cfactual", "mu0", "mu1"], axis=1)
    ihdp["treatment"] = ihdp["treatment"].astype(int)
    return ihdp, "treatment", "y_factual", False


def load_acic() -> Tuple[pd.DataFrame, str, str, bool]:
    acic_full = pd.read_csv("../data/acic.csv")
    acic_full["y"] = acic_full["y0"]
    acic_full.loc[acic_full["z"] == 1, "y"] = acic_full.loc[acic_full["z"] == 1, "y1"]
    acic = acic_full.drop(["y0", "y1", "mu0", "mu1"], axis=1)
    return acic, "z", "y", False


def load_jobs() -> Tuple[pd.DataFrame, str, str, bool]:
    jobs = pd.read_csv("../data/jobs_small.csv")
    jobs["training"] = jobs["training"].astype(int)
    return jobs, "training", "re78", False


DATASET_LOADERS: Dict[str, DatasetLoader] = {
    "actg": load_actg,
    "ihdp": load_ihdp,
    "acic": load_acic,
    "jobs": load_jobs,
}


AVAILABLE_MODELS: Tuple[str, ...] = ("tvae", "arf", "ctgan", "nflow", "ddpm")


def _ensure_output_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run STEAM vs. generic comparisons on real-world datasets."
    )
    parser.add_argument(
        "-d",
        "--datasets",
        choices=list(DATASET_LOADERS.keys()),
        nargs="+",
        default=list(DATASET_LOADERS.keys()),
        help="Datasets to evaluate. Defaults to all available datasets.",
    )
    parser.add_argument(
        "-m",
        "--models",
        choices=list(AVAILABLE_MODELS),
        nargs="+",
        default=list(AVAILABLE_MODELS),
        help="Generative models to compare. Defaults to all available models.",
    )
    parser.add_argument(
        "-n",
        "--iterations",
        type=int,
        default=20,
        help="Number of iterations to run for each dataset/model pair. Defaults to 20.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default="../results",
        help="Directory where CSV results will be stored.",
    )
    parser.add_argument(
        "-c",
        "--classifier",
        choices=["logistic", "rf"],
        default="logistic",
        help="Classifier to use for JSD_\\pi evaluation. Defaults to logistic regression.",
    )
    return parser.parse_args()


def run_experiments(
    datasets: Iterable[str],
    models: Iterable[str],
    iterations: int,
    output_dir: str,
    classifier: str,
) -> None:
    output_dir = _ensure_output_dir(output_dir)

    for dataset_name in datasets:
        loader = DATASET_LOADERS[dataset_name]
        real, treatment_col, outcome_col, binary_y = loader()
        print(f"Running {dataset_name.upper()} comparisons")

        for model in models:
            print(f"  - Model: {model}")
            output_path = os.path.join(output_dir, f"{dataset_name}_{model}.csv")
            compare_steam_and_generic(
                real,
                model,
                treatment_col,
                outcome_col,
                iterations,
                binary_y=binary_y,
                classifier=classifier,
                save=True,
                fp=output_path,
            )


def main() -> None:
    args = parse_args()
    run_experiments(args.datasets, args.models, args.iterations, args.output_dir, args.classifier)


if __name__ == "__main__":
    main()
