"""Utility functions for generating synthetic datasets for experiments."""

import random
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from catenets.models.torch import SLearner
from src.catenets_dp.models.torch import TLearner as TLearnerDP
from diffprivlib.models import LogisticRegression as LogisticRegressionDP
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from synthcity.plugins import Plugins
import torch


def _random_state() -> int:
    return random.randint(0, 1_000_000)


def _validate_privacy_params(
    epsilon: Optional[Sequence[float]],
    delta: Optional[Sequence[float]],
    expected_length: int,
) -> Tuple[Sequence[float], Sequence[float]]:
    if epsilon is None or delta is None:
        raise ValueError("epsilon and delta must be provided when private=True.")
    if len(epsilon) < expected_length or len(delta) < expected_length:
        raise ValueError(
            f"epsilon and delta must each contain at least {expected_length} elements when private=True."
        )
    return epsilon, delta

def generate_sequentially_to_w(
    real: pd.DataFrame,
    gen: str,
    treatment_col: str,
    outcome_col: str,
    private: bool = False,
    epsilon: Optional[Sequence[float]] = None,
    delta: Optional[Sequence[float]] = None,
    classifier: str = "logistic",
) -> pd.DataFrame:
    """
    Generates synthetic covariate and treatment data sequentially.

    This function first generates synthetic covariates using a specified generative model, then creates
    treatment assignments based on the real data. The synthetic dataset includes the
    generated covariates, treatment assignments, and a placeholder for outcomes.

    Parameters
    ----------
    real : pandas.DataFrame
        The real dataset.
    gen : str
        The name of the generative model to use for synthesizing covariates.
    treatment_col : str
        The name of the treatment column.
    outcome_col : str
        The name of the outcome column.
    private : bool, optional
        Indicates whether to use differential privacy for generating covariates and treatments. Default is `False`.
    epsilon : list of float, optional
        Privacy budget parameters for differential privacy, if `private=True`. The first value is used for covariates,
        and the second for treatments.
    delta : list of float, optional
        Privacy slack parameters for differential privacy, if `private=True`. The first value is used for covariates,
        and the second for treatment.

    Returns
    -------
    pandas.DataFrame
        A synthetic dataset containing generated covariates, treatments, and a placeholder outcome column 
        (initialized to 0).
    """
    random.seed()

    if private:
        epsilon, delta = _validate_privacy_params(epsilon, delta, expected_length=2)
        cov_epsilon, treat_epsilon = epsilon[0], epsilon[1]
        cov_delta = delta[0]
    else:
        cov_epsilon = treat_epsilon = cov_delta = None

    real_cov = real.drop([treatment_col, outcome_col], axis=1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if private:
        g = Plugins().get(gen, random_state=_random_state(), epsilon=cov_epsilon, delta=cov_delta, device=device)
    else:
        g = Plugins().get(gen, random_state=_random_state(), device=device)
    print(f'Fitting {gen} covariate model')
    g.fit(real_cov)
    print(f'Generating {gen} synthetic covariates')
    synth_cov = g.generate(count = len(real)).dataframe()

    #generate propensities
    X = np.array(real_cov)
    y = np.array(real[treatment_col])

    if private:
        classifier = LogisticRegressionDP(random_state=_random_state(), epsilon=treat_epsilon)
    else:
        if classifier == "logistic":
            classifier = LogisticRegression(random_state=_random_state())
        elif classifier == "rf":
            classifier = RandomForestClassifier(random_state=_random_state())
    print('Fitting propensity model')
    classifier.fit(X, y)
    print('Generating propensities')
    probabilities = classifier.predict_proba(np.array(synth_cov))
    prob_class_1 = probabilities[:, 1]
    binary_outcomes = np.random.binomial(n=1, p=prob_class_1)

    # Ensure both treatment classes have at least two instances, for stability reasons
    if np.all(binary_outcomes == 0):  # All outcomes are 0
        top_2_idx = np.argsort(prob_class_1)[-2:]  # Indices of top 2 probabilities for class 1
        binary_outcomes[top_2_idx] = 1  # Set top 2 to class 1
    elif np.all(binary_outcomes == 1):  # All outcomes are 1
        bottom_2_idx = np.argsort(prob_class_1)[:2]  # Indices of lowest 2 probabilities for class 1
        binary_outcomes[bottom_2_idx] = 0  # Set bottom 2 to class 0

    synth_cov_with_prop = synth_cov.copy()
    synth_cov_with_prop[treatment_col] = pd.Series(binary_outcomes, index=synth_cov.index)
    synth_cov_with_prop[outcome_col] = 0

    return synth_cov_with_prop

def generate_sequentially(
    real: pd.DataFrame,
    gen: str,
    treatment_col: str,
    outcome_col: str,
    private: bool = False,
    epsilon: Optional[Sequence[float]] = None,
    delta: Optional[Sequence[float]] = None,
    binary_y: bool = False,
    classifier: str = "logistic",
) -> pd.DataFrame:
    """
    Generates a STEAM synthetic dataset.

    This function combines covariate generation, treatment assignment, and outcome generation in a sequential manner:
    1. Covariates are synthesized using a generative model.
    2. Treatment assignments are generated based on the synthetic covariates.
    3. Outcomes are predicted using a potential outcome model trained on the real data.

    Parameters
    ----------
    real : pandas.DataFrame
        The real dataset.
    gen : str
        The name of the generative model to use for synthesizing covariates.
    treatment_col : str
        The name of the treatment column.
    outcome_col : str
        The name of the outcome column.
    private : bool, optional
        Indicates whether to use differential privacy for all steps of generation. Default is `False`.
    epsilon : list of float, optional
        Privacy budget parameters for differential privacy, if `private=True`. The values are:
        - `epsilon[0]`: Covariate generation.
        - `epsilon[1]`: Treatment generation.
        - `epsilon[2]`: Outcome generation.
    delta : list of float, optional
        Privacy slack parameters for differential privacy, if `private=True`. The values correspond to the same stages 
        as `epsilon`.
    binary_y : bool, optional
        Indicates whether the outcome is binary. Default is `False`.

    Returns
    -------
    pandas.DataFrame
        A STEAM synthetic dataset containing generated covariates, treatment assignments, and outcomes. 
    """
    random.seed()
    synth = generate_sequentially_to_w(
        real,
        gen,
        treatment_col,
        outcome_col,
        private=private,
        epsilon=epsilon,
        delta=delta,
        classifier=classifier
    )

    features = real.drop([treatment_col, outcome_col], axis=1)
    X = np.array(features)
    y = np.array(real[outcome_col])
    w = np.array(real[treatment_col])
    n_units = len(features.columns)
    
    if private:
        epsilon, delta = _validate_privacy_params(epsilon, delta, expected_length=3)
        l = TLearnerDP(
            n_unit_in=n_units,
            binary_y=binary_y,
            seed=_random_state(),
            batch_norm=False,
        )
        print('Fitting private CATE learner')
        l.fit(X, y, w, epsilon=epsilon[2], delta=delta[2])
    else:
        l = SLearner(n_unit_in=n_units, binary_y=binary_y, seed=_random_state(), )
        print('Fitting CATE learner')
        l.fit(X, y, w)

    seq_X = np.array(synth.drop([treatment_col, outcome_col], axis=1))
    print('Generating POs')
    cate, y0, y1 = l.predict(seq_X, return_po=True)

    outcomes = []
    for index, value in synth[treatment_col].items():
        if value == 0:
            outcomes.append(y0[index].item()) 
        else:
            outcomes.append(y1[index].item()) 
    synth[outcome_col] = outcomes
    return synth

def generate_standard(
    real: pd.DataFrame,
    gen: str,
    private: bool = False,
    epsilon: Optional[Sequence[float]] = None,
    delta: Optional[Sequence[float]] = None,
) -> pd.DataFrame:
    """
    Generates a synthetic dataset using a generative model, optionally incorporating differential privacy.

    This function fits a specified generative model on the real dataset and generates a synthetic dataset of the
    same size. If `private=True`, differential privacy is applied to the model, with privacy parameters `epsilon` and `delta`.

    Parameters
    ----------
    real : pandas.DataFrame
        The real dataset.
    gen : str
        The name of the generative model to use for synthesizing the dataset.
    private : bool, optional
        Indicates whether to use differential privacy for the generative model. Default is `False`.
    epsilon : list of float, optional
        Privacy budget parameters for differential privacy. If `private=True`, the sum of this list is used as the
        privacy budget for the model. Default is `None`.
    delta : list of float, optional
        Privacy slack parameters for differential privacy. If `private=True`, the sum of this list is used as the
        privacy slack for the model. Default is `None`.

    Returns
    -------
    pandas.DataFrame
        A synthetic dataset generated by the specified generative model. The dataset has the same structure and 
        size as the input `real` dataset.
    """
    random.seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if private:
        epsilon, delta = _validate_privacy_params(epsilon, delta, expected_length=1)
        epsilon_sum = sum(epsilon)
        delta_sum = sum(delta)
        g = Plugins().get(
            gen,
            random_state=_random_state(),
            epsilon=epsilon_sum,
            delta=delta_sum,
            device=device,
        )
    else:
        g = Plugins().get(gen, random_state=_random_state(), device=device)

    print(f'Fitting {gen} model')
    assert isinstance(real, pd.DataFrame)

    g.fit(real)
    print(f'Generating {gen} synthetic dataset')
    synth = g.generate(count=len(real)).dataframe()
    return synth