"""Examples of existing metric failures."""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import numpy as np
import pandas as pd
from catenets.models.torch import DragonNet, SLearner, TARNet, TLearner
from synthcity.metrics.eval_statistical import (
    AlphaPrecision,
    InverseKLDivergence,
    JensenShannonDistance,
    KolmogorovSmirnovTest,
    MaximumMeanDiscrepancy,
    WassersteinDistance,
)
from synthcity.plugins.core.dataloader import GenericDataLoader

from catenets.experiment_utils.simulation_utils import simulate_treatment_setup
from src.metrics import evaluate_average_u_pehe, evaluate_jsd
from sklearn.metrics import root_mean_squared_error
from scipy.special import expit


def _evaluate_metrics(real: pd.DataFrame, synth: pd.DataFrame) -> dict[str, float]:
    alpha = AlphaPrecision()
    kl = InverseKLDivergence()
    mmd = MaximumMeanDiscrepancy()

    loader_real = GenericDataLoader(real)
    loader_synth = GenericDataLoader(synth)

    alpha_result = alpha.evaluate(loader_real, loader_synth)
    kl_result = kl.evaluate(loader_real, loader_synth)
    mmd_result = mmd.evaluate(loader_real, loader_synth)

    return {
        "alpha_precision": alpha_result["delta_precision_alpha_OC"],
        "beta_recall": alpha_result["delta_coverage_beta_OC"],
        "kl_marginal": kl_result["marginal"],
        "mmd_joint": mmd_result["joint"],
    }


def _print_metrics(title: str, results: dict[str, float]) -> None:
    print(title)
    for key, value in results.items():
        print(f"{key.replace('_', ' ')}: {value}")


def _simulate_dataframe(d: int, *, seed: int | None = None, n_t: int | None = None) -> pd.DataFrame:
    kwargs = {"seed": seed} if seed is not None else {}
    if n_t is not None:
        kwargs["n_t"] = n_t
    X, y, w, p, t = simulate_treatment_setup(1000, d, **kwargs)
    X_df = pd.DataFrame(X)
    y_df = pd.DataFrame(y, columns=["y"])
    w_df = pd.DataFrame(w, columns=["w"])
    return pd.concat([X_df, w_df, y_df], axis=1)


def experiment_covariate_distribution() -> None:
    real = _simulate_dataframe(1)
    synth = _simulate_dataframe(1, seed=1)

    covariate_cols = real.drop(["w", "y"], axis=1).columns
    synth.loc[:, covariate_cols] = 0

    results = _evaluate_metrics(real, synth)
    _print_metrics("Modelling P_x", results)


def experiment_treatment_assignment() -> None:
    real = _simulate_dataframe(1)
    synth = _simulate_dataframe(1, seed=1)

    synth.loc[:, "w"] = 0

    results = _evaluate_metrics(real, synth)
    _print_metrics("Modelling P_w|x", results)


def experiment_outcome_generation() -> None:
    real = _simulate_dataframe(1, n_t=1)
    synth = _simulate_dataframe(1, n_t=1, seed=1)

    synth.loc[:, "y"] = np.random.normal(loc=0, scale=1, size=len(synth))

    results = _evaluate_metrics(real, synth)
    _print_metrics("Modelling P_y|x,w", results)

def _make_generator_factories() -> Dict[str, Callable[[int], object]]:
    return {
        "t": lambda d: TLearner(n_unit_in=d, binary_y=False, batch_norm=False),
        "s": lambda d: SLearner(n_unit_in=d, binary_y=False, batch_norm=False),
        "dr": lambda d: DragonNet(n_unit_in=d, binary_y=False, batch_norm=False),
        "tar": lambda d: TARNet(n_unit_in=d, binary_y=False, batch_norm=False),
    }


def _assign_outcomes(
    generator_factory: Callable[[int], object],
    dimension: int,
    X_real: np.ndarray,
    y_real: np.ndarray,
    w_real: np.ndarray,
    X_synth: pd.DataFrame,
    base_synth: pd.DataFrame,
) -> pd.DataFrame:
    generator = generator_factory(dimension)
    generator.fit(X_real, y_real, w_real)
    _, y0, y1 = generator.predict(X_synth.to_numpy(), return_po=True)

    treated = base_synth["w"].to_numpy().astype(int)
    y0_np = np.array([val.item() if hasattr(val, "item") else val for val in y0])
    y1_np = np.array([val.item() if hasattr(val, "item") else val for val in y1])
    outcomes = np.where(treated == 0, y0_np, y1_np)

    synth = base_synth.copy()
    synth["y"] = outcomes
    return synth


def _oracle_rmse(
    cate: np.ndarray,
    X_real: np.ndarray,
    synth: pd.DataFrame,
    dimension: int,
) -> float:
    learner = TLearner(n_unit_in=dimension, binary_y=False)
    features = synth.drop(["w", "y"], axis=1).to_numpy()
    outcomes = synth["y"].to_numpy()
    treatments = synth["w"].to_numpy()
    learner.fit(features, outcomes, treatments)
    predictions = learner.predict(X_real)
    return root_mean_squared_error(cate, predictions.detach().cpu().numpy())


def _evaluate_selection_metrics(
    real: pd.DataFrame,
    synth: pd.DataFrame,
) -> dict[str, float]:
    loader_real = GenericDataLoader(real)
    loader_synth = GenericDataLoader(synth)

    alpha = AlphaPrecision()
    alpha_metrics = alpha.evaluate(loader_real, loader_synth)
    kl = InverseKLDivergence()
    kl_metrics = kl.evaluate(loader_real, loader_synth)
    was = WassersteinDistance()
    was_metrics = was.evaluate(loader_real, loader_synth)
    ks = KolmogorovSmirnovTest()
    ks_metrics = ks.evaluate(loader_real, loader_synth)
    jsd = JensenShannonDistance()
    jsd_metrics = jsd.evaluate(loader_real, loader_synth)

    return {
        "alpha": alpha_metrics["delta_precision_alpha_OC"],
        "beta": alpha_metrics["delta_coverage_beta_OC"],
        "kl": kl_metrics["marginal"],
        "was": was_metrics["joint"],
        "ks": ks_metrics["marginal"],
        "jsd": jsd_metrics["marginal"],
    }


def selection_test_outcome_generation(
    n_iter: int = 1,
    save: bool = False,
    fp: str = "",
) -> pd.DataFrame:
    """Compare outcome generators by metric performance and oracle RMSE."""

    records = []
    generator_factories = _make_generator_factories()

    for _ in range(n_iter):
        n = 1000
        d = 10
        n_c = 0
        n_t = 1
        X, y, w, _, cate = simulate_treatment_setup(n, d=d, n_c=n_c, n_t=n_t)
        real_df = pd.concat(
            [pd.DataFrame(X), pd.DataFrame(w, columns=["w"]), pd.DataFrame(y, columns=["y"])],
            axis=1,
        )

        X_syn, _, w_syn, _, _ = simulate_treatment_setup(n, d=d, n_c=n_c, n_t=n_t)
        X_syn_df = pd.DataFrame(X_syn)
        base_synth = pd.concat([X_syn_df, pd.DataFrame(w_syn, columns=["w"])], axis=1)

        for name, factory in generator_factories.items():
            synth_with_outcomes = _assign_outcomes(
                factory,
                d,
                X,
                y,
                w,
                X_syn_df,
                base_synth,
            )

            metrics = _evaluate_selection_metrics(real_df, synth_with_outcomes)
            record = {
                "outcome_learner": name,
                **metrics,
                "u": evaluate_average_u_pehe(real_df, synth_with_outcomes, "w", "y", d),
                "oracle": _oracle_rmse(cate, X, synth_with_outcomes, d),
            }
            records.append(record)

    columns = [
        "outcome_learner",
        "alpha",
        "beta",
        "kl",
        "was",
        "ks",
        "jsd",
        "u",
        "oracle",
    ]
    results = pd.DataFrame(records, columns=columns)
    if results.empty:
        results = pd.DataFrame(columns=columns)
    else:
        results = results[columns]

    if save and fp:
        results.to_csv(fp, index=False)

    return results

def prop_test(
    X: np.ndarray,
    n_c: int = 0,
    n_w: int = 0,
    xi: float = 0.5,
    nonlinear: bool = True,
    offset: Any = 0,
    target_prop: Optional[np.ndarray] = None,
) -> np.ndarray:
    if n_c + n_w == 0:
        # constant propensity
        return xi * np.ones(X.shape[0])
    else:
        coefs = np.ones(n_c + n_w)

        if nonlinear:
            z = np.dot(X[:, : (n_c + n_w)] , coefs) / (n_c + n_w)
        else:
            z = np.dot(X[:, : (n_c + n_w)], coefs) / (n_c + n_w)

        if type(offset) is float or type(offset) is int:
            prop = expit(xi * z + offset)
            if target_prop is not None:
                avg_prop = np.average(prop)
                prop = target_prop / avg_prop * prop
            return prop
        elif offset == "center":
            # center the propensity scores to median 0.5
            prop = expit(xi * (z - np.median(z)))
            if target_prop is not None:
                avg_prop = np.average(prop)
                prop = target_prop / avg_prop * prop
            return prop
        else:
            raise ValueError("Not a valid value for offset")

def _simulate_propensity_dataframe(
    n: int,
    d: int,
    n_w: int,
    *,
    seed: int,
) -> pd.DataFrame:
    X, y, w, _, _ = simulate_treatment_setup(
        n,
        d=d,
        n_t=0,
        n_w=n_w,
        seed=seed,
        propensity_model=prop_test,
    )
    X_df = pd.DataFrame(X)
    y_df = pd.DataFrame(y, columns=["y"])
    w_df = pd.DataFrame(w, columns=["w"])
    return pd.concat([X_df, w_df, y_df], axis=1)


def selection_test_treatment_assignment(
    n_iter: int = 1,
    save: bool = False,
    fp: str = "",
) -> pd.DataFrame:
    """Evaluate assignment metrics as confounders are removed."""

    records: list[dict[str, float | int]] = []
    rng = np.random.default_rng()

    n = 1000
    d = 5
    max_confounders = 5
    confounder_counts = [5,3,1]

    for _ in range(n_iter):
        base_seed = int(rng.integers(1, 1_000_000))
        real = _simulate_propensity_dataframe(n, d, max_confounders, seed=base_seed)

        for idx, count in enumerate(confounder_counts):
            synth = _simulate_propensity_dataframe(
                n,
                d,
                count,
                seed=base_seed + idx + 1,
            )

            metrics = _evaluate_selection_metrics(real, synth)
            records.append(
                {
                    "num_correct_confounders": count,
                    **metrics,
                    "JSD_pi": evaluate_jsd(real, synth, "w", "y"),
                }
            )

    columns = [
        "num_correct_confounders",
        "alpha",
        "beta",
        "kl",
        "was",
        "ks",
        "jsd",
        "JSD_pi",
    ]

    results = pd.DataFrame(records, columns=columns)
    if results.empty:
        results = pd.DataFrame(columns=columns)
    else:
        results = results[columns]

    if save and fp:
        results.to_csv(fp, index=False)

    return results

def main() -> None:
    experiment_covariate_distribution()
    experiment_treatment_assignment()
    experiment_outcome_generation()
    n_iter = 5

    selection_results = selection_test_outcome_generation(n_iter=n_iter, save=True, fp='../results/outcome_generation_metrics.csv')
    if not selection_results.empty:
        means = round(selection_results.groupby('outcome_learner').mean(), 3)
        print("Mean selection test outcome generation metrics:")
        print(means)

        cis = round(selection_results.groupby('outcome_learner').std() * 1.96 / np.sqrt(n_iter), 3)
        print("95% confidence intervals for selection test outcome generation metrics:")
        print(cis)

    treatment_results = selection_test_treatment_assignment(n_iter=n_iter, save=True, fp='../results/treatment_assignment_metrics.csv')
    if not treatment_results.empty:
        means = round(treatment_results.groupby('num_correct_confounders').mean(), 3)
        print("Mean selection test treatment assignment metrics:")
        print(means)

        cis = round(treatment_results.groupby('num_correct_confounders').std() * 1.96 / np.sqrt(n_iter), 3)
        print("95% confidence intervals for selection test treatment assignment metrics:")
        print(cis)

if __name__ == "__main__":
    main()