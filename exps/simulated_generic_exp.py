"""Simulated data insight experiments."""

import argparse
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from catenets.experiment_utils.simulation_utils import simulate_treatment_setup
from src.generation import generate_sequentially, generate_standard
from src.metrics import evaluate_average_u_pehe, evaluate_c, evaluate_f, evaluate_jsd

def compare_steam_and_generic(
    real,
    gen,
    treatment_col,
    outcome_col,
    n_iter,
    private=False,
    epsilon=None,
    delta=None,
    binary_y=False,
    classifier="logistic",
    save=False,
    fp="",
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
        # Ensure `standard[treatment_col]` has at least 2 of each class for stability reasons with U_PEHE evaluation
        if standard[treatment_col].nunique() == 1 and len(standard) >= 2:  # All generated samples are treated or untreated
            indices_to_flip = random.sample(range(len(standard)), 2)  # Randomly select 2 indices
            standard.loc[indices_to_flip, treatment_col] = 1 - standard.loc[indices_to_flip, treatment_col]  # Flip values
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

def num_cov_insight_exp(n, ds, n_t, n_c, gen, n_iter, save=False, fp=''):
    """
    Compares the performance of generic and STEAM synthetic data generation methods across simulated datasets with different numbers of covariates.

    This function simulates datasets with varying numbers of covariates, generates synthetic data using both a generic generative model and its STEAM counterpart,
    evaluates the data using specified metrics, and aggregates the results. Optionally, results can be saved to a CSV file.

    Parameters
    ----------
    n : int
        The number of samples in each simulated dataset.
    ds : list of int
        A list of different numbers of covariates to simulate datasets with.
    n_t : int
        The number of covariates that influence outcomes only.
    n_c : int
        The number of confounding covariates, influencing both treatment and outcome.
    gen : str
        The name of the generic generative model.
    n_iter : int
        The number of iterations to run for each dataset configuration.
    save : bool, optional
        Indicates whether to save the results to a CSV file. Default is `False`.
    fp : str, optional
        The results file path, if `save=True`. Default is an empty string.
    """

    results = pd.DataFrame(columns=["method", "p_a,x", "r_b,x", "jsd_pi", "u_pehe", "num_cov"])

    for d in ds:
        X, y, w, p, t = simulate_treatment_setup(n, d=d, n_c=n_c, n_t=n_t)
        X_df = pd.DataFrame(X)
        y_df = pd.DataFrame(y, columns=["y"])
        w_df = pd.DataFrame(w, columns=["w"])
        d_real = pd.concat([X_df, w_df, y_df], axis=1)

        r = compare_steam_and_generic(d_real, gen, "w", "y", n_iter)
        r["num_cov"] = d
        results = pd.concat([results, r])
        print(f"Tested num. covariates = {d}")

    if save and fp:
        results.to_csv(fp, index=False)
        
    return results

def confounding_insight_exp(n, d, n_t, n_cs, gen, n_iter, save=False, fp=''):
    """
    Compares the performance of generic and STEAM synthetic data generation methods across simulated datasets with different numbers of confounding covariates.

    This function simulates datasets with varying numbers of confounding covariates, generates synthetic data using both a generic generative model and its STEAM counterpart,
    evaluates the data using specified metrics, and aggregates the results. Optionally, results can be saved to a CSV file.

    Parameters
    ----------
    n : int
        The number of samples in each simulated dataset.
    d : int
        The total number of covariates in the dataset.
    n_t : int
        The number of covariates that influence outcomes only.
    n_cs : list of int
        A list of different numbers of confounding covariates to simulate datasets with.
    gen : str
        The name of the generic generative model.
    n_iter : int
        The number of iterations to run for each dataset configuration.
    save : bool, optional
        Indicates whether to save the results to a CSV file. Default is `False`.
    fp : str, optional
        The results file path, if `save=True`. Default is an empty string.
    """

    results = pd.DataFrame(columns=["method", "p_a,x", "r_b,x", "jsd_pi", "u_pehe", "n_c"])

    for n_c in n_cs:
        X, y, w, p, t = simulate_treatment_setup(n, d=d, n_c=n_c, n_t=n_t)
        X_df = pd.DataFrame(X)
        y_df = pd.DataFrame(y, columns=["y"])
        w_df = pd.DataFrame(w, columns=["w"])
        d_real = pd.concat([X_df, w_df, y_df], axis=1)

        r = compare_steam_and_generic(d_real, gen, "w", "y", n_iter)
        r["n_c"] = n_c
        results = pd.concat([results, r])
        print(f"Tested n_c = {n_c}")

    if save and fp:
        results.to_csv(fp, index=False)
        
    return results



def predictive_insight_exp(n, d, n_c, n_ts, gen, n_iter, save=False, fp=''):
    """
    Compares the performance of generic and STEAM synthetic data generation methods across simulated datasets with different numbers of predictive covariates.

    This function simulates datasets with varying numbers of predictive covariates, generates synthetic data using both a generic generative model and its STEAM counterpart,
    evaluates the data using specified metrics, and aggregates the results. Optionally, results can be saved to a CSV file.

    Parameters  
    ----------
    n : int
        The number of samples in each simulated dataset.
    d : int
        The total number of covariates in the dataset.
    n_c : int
        The number of confounding covariates, influencing both treatment and outcome.
    n_ts : list of int
        A list of different numbers of predictive covariates to simulate datasets with.
    gen : str
        The name of the generic generative model.
    n_iter : int
        The number of iterations to run for each dataset configuration.
    save : bool, optional
        Indicates whether to save the results to a CSV file. Default is `False`.
    fp : str, optional
        The results file path, if `save=True`. Default is an empty string.
    """

    results = pd.DataFrame(columns=["method", "p_a,x", "r_b,x", "jsd_pi", "u_pehe", "n_t"])

    for n_t in n_ts:
        X, y, w, p, t = simulate_treatment_setup(n, d=d, n_c=n_c, n_t=n_t)
        X_df = pd.DataFrame(X)
        y_df = pd.DataFrame(y, columns=["y"])
        w_df = pd.DataFrame(w, columns=["w"])
        d_real = pd.concat([X_df, w_df, y_df], axis=1)

        r = compare_steam_and_generic(d_real, gen, "w", "y", n_iter)
        r["n_t"] = n_t
        results = pd.concat([results, r])

        print(f"Tested n_t = {n_t}")

    if save and fp:
        results.to_csv(fp, index=False)
        
    return results



def _ensure_output_dir(output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def _mean_and_ci_by_group(df: pd.DataFrame, group_key: str, metric: str) -> tuple[pd.Series, pd.Series]:
    grouped = df.groupby(group_key)[metric]
    mean = grouped.mean()
    std = grouped.std()
    count = grouped.count()
    denom = np.sqrt(count.replace(0, np.nan))
    ci = std.divide(denom).fillna(0.0) * 1.96
    return mean, ci


def run_covariate_insight(output_dir: str, n_iter: int) -> pd.DataFrame:
    results_path = os.path.join(output_dir, "covariate_insight.csv")
    results = num_cov_insight_exp(
        1000,
        [5, 10, 20, 50],
        2,
        2,
        "ddpm",
        n_iter,
        save=True,
        fp=results_path,
    )

    x_values = sorted(results["num_cov"].unique())
    steam_mean, steam_ci = _mean_and_ci_by_group(
        results[results["method"] == "STEAM"], "num_cov", "u_pehe"
    )
    generic_mean, generic_ci = _mean_and_ci_by_group(
        results[results["method"] == "generic"], "num_cov", "u_pehe"
    )

    plt.rcParams.update({"font.size": 30})

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(x_values, generic_mean.reindex(x_values), "o-", label="Generic")
    ax.plot(x_values, steam_mean.reindex(x_values), "o-", label="STEAM")

    ax.fill_between(
        x_values,
        generic_mean.reindex(x_values) - generic_ci.reindex(x_values, fill_value=0.0),
        generic_mean.reindex(x_values) + generic_ci.reindex(x_values, fill_value=0.0),
        alpha=0.2,
    )
    ax.fill_between(
        x_values,
        steam_mean.reindex(x_values) - steam_ci.reindex(x_values, fill_value=0.0),
        steam_mean.reindex(x_values) + steam_ci.reindex(x_values, fill_value=0.0),
        alpha=0.2,
    )
    ax.set_ylabel(r"$U_{PEHE}$")
    ax.set_xticks(x_values)
    fig.savefig(os.path.join(output_dir, "covariate_dimension_insight_U.pdf"), bbox_inches="tight")
    plt.close(fig)

    steam_mean_jsd, steam_ci_jsd = _mean_and_ci_by_group(
        results[results["method"] == "STEAM"], "num_cov", "jsd_pi"
    )
    generic_mean_jsd, generic_ci_jsd = _mean_and_ci_by_group(
        results[results["method"] == "generic"], "num_cov", "jsd_pi"
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(x_values, generic_mean_jsd.reindex(x_values), "o-", label="Generic")
    ax.plot(x_values, steam_mean_jsd.reindex(x_values), "o-", label="STEAM")

    ax.fill_between(
        x_values,
        generic_mean_jsd.reindex(x_values) - generic_ci_jsd.reindex(x_values, fill_value=0.0),
        generic_mean_jsd.reindex(x_values) + generic_ci_jsd.reindex(x_values, fill_value=0.0),
        alpha=0.2,
    )
    ax.fill_between(
        x_values,
        steam_mean_jsd.reindex(x_values) - steam_ci_jsd.reindex(x_values, fill_value=0.0),
        steam_mean_jsd.reindex(x_values) + steam_ci_jsd.reindex(x_values, fill_value=0.0),
        alpha=0.2,
    )
    ax.legend()
    ax.set_xlabel(r"$d$")
    ax.set_ylabel(r"$JSD_\pi$")
    ax.set_xticks(x_values)
    fig.savefig(os.path.join(output_dir, "covariate_dimension_insight_d.pdf"), bbox_inches="tight")
    plt.close(fig)

    return results


def run_confounding_insight(output_dir: str, n_iter: int) -> pd.DataFrame:
    results_path = os.path.join(output_dir, "treatment_assignment_insight.csv")
    results = confounding_insight_exp(
        1000,
        10,
        2,
        [1, 2, 3, 4, 5],
        "ddpm",
        n_iter,
        save=True,
        fp=results_path,
    )

    x_values = sorted(results["n_c"].unique())
    steam_mean_jsd, steam_ci_jsd = _mean_and_ci_by_group(
        results[results["method"] == "STEAM"], "n_c", "jsd_pi"
    )
    generic_mean_jsd, generic_ci_jsd = _mean_and_ci_by_group(
        results[results["method"] == "generic"], "n_c", "jsd_pi"
    )

    plt.rcParams.update({"font.size": 30})

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(x_values, generic_mean_jsd.reindex(x_values), "o-", label="Generic")
    ax.plot(x_values, steam_mean_jsd.reindex(x_values), "o-", label="STEAM")

    ax.fill_between(
        x_values,
        generic_mean_jsd.reindex(x_values) - generic_ci_jsd.reindex(x_values, fill_value=0.0),
        generic_mean_jsd.reindex(x_values) + generic_ci_jsd.reindex(x_values, fill_value=0.0),
        alpha=0.2,
    )
    ax.fill_between(
        x_values,
        steam_mean_jsd.reindex(x_values) - steam_ci_jsd.reindex(x_values, fill_value=0.0),
        steam_mean_jsd.reindex(x_values) + steam_ci_jsd.reindex(x_values, fill_value=0.0),
        alpha=0.2,
    )
    ax.legend()
    ax.set_xlabel("K")
    ax.set_ylabel(r"$JSD_\pi$")
    ax.set_ylim([0.8, 1])
    ax.set_xticks(x_values)
    fig.savefig(os.path.join(output_dir, "treatment_assignment_insight.pdf"), bbox_inches="tight")
    plt.close(fig)

    return results


def run_predictive_insight(output_dir: str, n_iter: int) -> pd.DataFrame:
    results_path = os.path.join(output_dir, "predictive_insight.csv")
    results = predictive_insight_exp(
        1000,
        10,
        2,
        [1, 2, 3, 4, 5],
        "ddpm",
        n_iter,
        save=True,
        fp=results_path,
    )

    adjusted_results = results.copy()
    adjusted_results["n_t"] = adjusted_results["n_t"] + 2
    x_values = sorted(adjusted_results["n_t"].unique())

    steam_mean, steam_ci = _mean_and_ci_by_group(
        adjusted_results[adjusted_results["method"] == "STEAM"], "n_t", "u_pehe"
    )
    generic_mean, generic_ci = _mean_and_ci_by_group(
        adjusted_results[adjusted_results["method"] == "generic"], "n_t", "u_pehe"
    )

    plt.rcParams.update({"font.size": 30})

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(x_values, generic_mean.reindex(x_values), "o-", label="Generic")
    ax.plot(x_values, steam_mean.reindex(x_values), "o-", label="STEAM")

    ax.fill_between(
        x_values,
        generic_mean.reindex(x_values) - generic_ci.reindex(x_values, fill_value=0.0),
        generic_mean.reindex(x_values) + generic_ci.reindex(x_values, fill_value=0.0),
        alpha=0.2,
    )
    ax.fill_between(
        x_values,
        steam_mean.reindex(x_values) - steam_ci.reindex(x_values, fill_value=0.0),
        steam_mean.reindex(x_values) + steam_ci.reindex(x_values, fill_value=0.0),
        alpha=0.2,
    )
    ax.legend()
    ax.set_xlabel("$K$")
    ax.set_ylabel(r"$U_{PEHE}$")
    ax.set_xticks(x_values)
    fig.savefig(os.path.join(output_dir, "predictive_insight.pdf"), bbox_inches="tight")
    plt.close(fig)

    return adjusted_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run STEAM vs. generic simulated insight experiments."
    )
    parser.add_argument(
        "-e",
        "--experiments",
        choices=["covariate", "confounding", "predictive"],
        nargs="+",
        default=["covariate", "confounding", "predictive"],
        help="List of insight experiments to execute. Defaults to all experiments.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default="../results",
        help="Directory where CSV results and figures will be stored.",
    )
    parser.add_argument(
        "-n",
        "--iterations",
        type=int,
        default=10,
        help="Number of iterations for each experiment. Defaults to 10.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = _ensure_output_dir(args.output_dir)

    experiment_map = {
        "covariate": run_covariate_insight,
        "confounding": run_confounding_insight,
        "predictive": run_predictive_insight,
    }

    for experiment in args.experiments:
        print(f"Running {experiment} insight experiment...")
        experiment_map[experiment](output_dir, args.iterations)

if __name__ == "__main__":
    main()