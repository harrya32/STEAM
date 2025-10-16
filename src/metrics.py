"""Evaluation metrics for comparing real and synthetic datasets."""

from __future__ import annotations

import random
from typing import Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from captum.attr import IntegratedGradients
from catenets.models.torch import DRLearner, RALearner, SLearner, TLearner
from scipy.spatial.distance import jensenshannon
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, root_mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
from synthcity.metrics.eval_statistical import AlphaPrecision
from synthcity.plugins.core.dataloader import GenericDataLoader


def _random_seed() -> int:
    return random.randint(0, 1_000_000)


def _split_train_test(
    df: pd.DataFrame,
    test_fraction: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if df.empty:
        raise ValueError("Input dataframe must not be empty.")

    n_test = round(len(df) * test_fraction)
    test = df.iloc[:n_test].copy()
    train = df.iloc[n_test:].copy()
    return train, test


def _covariates(df: pd.DataFrame, treatment_col: str, outcome_col: str) -> pd.DataFrame:
    return df.drop([treatment_col, outcome_col], axis=1)


def _feature_matrix(df: pd.DataFrame, treatment_col: str, outcome_col: str) -> np.ndarray:
    return _covariates(df, treatment_col, outcome_col).to_numpy()


BASE_LEARNERS: Tuple[type, ...] = (TLearner, SLearner, DRLearner, RALearner)


def evaluate_f(
    real: pd.DataFrame,
    synth: pd.DataFrame,
    treatment_col: str,
    outcome_col: str,
) -> float:
    """
    Evaluates the covariate precision metric between real and synthetic datasets using synthcity's AlphaPrecision.

    Parameters
    ----------
    real : pandas.DataFrame
        The real dataset.
    synth : pandas.DataFrame
        The synthetic dataset.
    treatment_col : str
        The name of the treatment column.
    outcome_col : str
        The name of the outcome column.

    Returns
    -------
    float
        The alpha precision metric computed between the real and synthetic datasets.

    """
    alpha = AlphaPrecision(random_state=_random_seed())
    real_cov = _covariates(real, treatment_col, outcome_col)
    synth_cov = _covariates(synth, treatment_col, outcome_col)
    result = alpha.evaluate(GenericDataLoader(real_cov), GenericDataLoader(synth_cov))
    return result['delta_precision_alpha_OC']


def evaluate_c(
    real: pd.DataFrame,
    synth: pd.DataFrame,
    treatment_col: str,
    outcome_col: str,
) -> float:
    """
    Evaluates the covariate recall metric between real and synthetic datasets using synthcity's AlphaPrecision.

    Parameters
    ----------
    real : pandas.DataFrame
        The real dataset.
    synth : pandas.DataFrame
        The synthetic dataset.
    treatment_col : str
        The name of the treatment column.
    outcome_col : str
        The name of the outcome column.

    Returns
    -------
    float
        The beta recall metric computed between the real and synthetic datasets.

    """
    alpha = AlphaPrecision(random_state=_random_seed())
    real_cov = _covariates(real, treatment_col, outcome_col)
    synth_cov = _covariates(synth, treatment_col, outcome_col)
    result = alpha.evaluate(GenericDataLoader(real_cov), GenericDataLoader(synth_cov))
    return result['delta_coverage_beta_OC']



def train_propensity_function(
    data: pd.DataFrame,
    treatment_col: str,
    outcome_col: str,
    clf,
):
    """
    Trains a propensity model to predict treatment assignment.

    This function trains a classifier on the covariates of the input dataset to predict the treatment assignment.

    Parameters
    ----------
    data : pandas.DataFrame
        The input dataset.
    treatment_col : str
        The name of the treatment column.
    outcome_col : str
        The name of the outcome column.
    clf : sklearn.base.BaseEstimator
        A classifier to be trained on the covariates for predicting the treatment.

    Returns
    -------
    sklearn.base.BaseEstimator
        The trained classifier.
    """
    X = _feature_matrix(data, treatment_col, outcome_col)
    y = data[treatment_col].to_numpy()
    clf.fit(X, y)
    return clf

def get_jsd(real: Sequence[Sequence[float]], synth: Sequence[Sequence[float]]) -> float:
    """
    Computes the average Jensen-Shannon Distance (JSD) between two lists of binary probability distributions.

    This function calculates the JSD for each corresponding pair of binary probability
    distributions in `real` and `synth`, then returns the average JSD over all pairs.

    Parameters
    ----------
    real : list of float
        A list where each element is a binary probability value between 0 and 1 representing the real distribution.
    synth : list of float
        A list where each element is a binary probability value between 0 and 1 representing the synthetic distribution.
        Must have the same length as `real`.

    Returns
    -------
    float
        The average Jensen-Shannon Divergence (JSD) between the `real` and `synth` distributions.

    Notes
    -----
    - The Jensen-Shannon Divergence is computed using `scipy.spatial.distance.jensenshannon`.
    - Each element in the `real` and `synth` lists is treated as a binary probability distribution.
    - The base of the logarithm used in the JSD calculation is 2, resulting in values between 0 and 1.
    """
    if len(real) != len(synth):
        raise ValueError("real and synth must have the same length.")

    total = 0.0
    for r, s in zip(real, synth):
        total += jensenshannon(r, s, base=2)
    return total / len(real)

def evaluate_jsd(
    real: pd.DataFrame,
    synth: pd.DataFrame,
    treatment_col: str,
    outcome_col: str,
    classifier: str = 'logistic',
) -> float:
    """
    Evaluates the JSD_pi metric, the similarity between real and synthetic treatment assignment mechanisms using Jensen-Shannon Distance (JSD).

    This function trains propensity score models on the real and synthetic datasets, then computes 
    the JSD between their predicted probabilities on a test set derived from the real dataset.
    The returned value is `1 - JSD`, which represents the similarity (higher values indicate higher similarity).

    Parameters
    ----------
    real : pandas.DataFrame
        The real dataset.
    synth : pandas.DataFrame
        The synthetic dataset.
    treatment_col : str
        The name of the treatment column.
    outcome_col : str
        The name of the outcome column.
    classifier : str, optional
        The type of classifier to use for propensity score estimation. Options are:
        - `'logistic'` (default): Uses `LogisticRegression`.
        - `'rf'`: Uses `RandomForestClassifier`.

    Returns
    -------
    float
        JSD_pi metric, a similarity score between the real and synthetic treatment assignment mechanisms.

    Raises
    ------
    ValueError
        If the specified `classifier` is not supported.
    """
    real_train, test = _split_train_test(real)

    if classifier == 'logistic':
        pi_real = LogisticRegression()
        pi_synth = LogisticRegression()
    elif classifier == 'rf':
        pi_real = RandomForestClassifier()
        pi_synth = RandomForestClassifier()
    else:
        raise ValueError(f'Classifier type {classifier} is not supported.')

    pi_real = train_propensity_function(real_train, treatment_col, outcome_col, pi_real)
    pi_synth = train_propensity_function(synth, treatment_col, outcome_col, pi_synth)

    test_features = _feature_matrix(test, treatment_col, outcome_col)
    probabilities_real = pi_real.predict_proba(test_features)
    probabilities_synth = pi_synth.predict_proba(test_features)

    return 1 - get_jsd(probabilities_real, probabilities_synth)

def evaluate_average_u_pehe(
    real: pd.DataFrame,
    synth: pd.DataFrame,
    treatment_col: str,
    outcome_col: str,
    n_units: int,
    binary_y: bool = False,
) -> float:
    """
    Evaluates the U_PEHE metric, the similarity between real and synthetic outcome generation mechanisms.

    This function computes U_PEHE by comparing the predictions of CATE learners trained on the real and synthetic datasets. 
    The comparison is made on a test set derived from the real dataset.

    Parameters
    ----------
    real : pandas.DataFrame
        The real dataset.
    synth : pandas.DataFrame
        The synthetic dataset.
    treatment_col : str
        The name of the treatment column.
    outcome_col : str
        The name of the outcome column.
    n_units : int
        The number of units (features) for the learners.
    binary_y : bool, optional
        Indicates whether the outcome variable is binary. Default is `False`.

    Returns
    -------
    float
        The U_PEHE metric, a similarity score between real and synthetic outcome generation mechanisms.
    """
    real_train, test = _split_train_test(real)
    X_test = _feature_matrix(test, treatment_col, outcome_col)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

    X_real = _feature_matrix(real_train, treatment_col, outcome_col)
    y_real = real_train[outcome_col].to_numpy()
    w_real = real_train[treatment_col].to_numpy()

    X_synth = _feature_matrix(synth, treatment_col, outcome_col)
    y_synth = synth[outcome_col].to_numpy()
    w_synth = synth[treatment_col].to_numpy()

    total_pehe = 0.0
    for learner_cls in BASE_LEARNERS:
        real_learner = learner_cls(n_unit_in=n_units, binary_y=binary_y, seed=_random_seed())
        synth_learner = learner_cls(n_unit_in=n_units, binary_y=binary_y, seed=_random_seed())

        real_learner.fit(X_real, y_real, w_real)
        pred_real = real_learner.predict(X_test_tensor)

        synth_learner.fit(X_synth, y_synth, w_synth)
        pred_synth = synth_learner.predict(X_test_tensor)

        total_pehe += root_mean_squared_error(
            pred_real.cpu().detach().numpy(),
            pred_synth.cpu().detach().numpy(),
        )

    return total_pehe / len(BASE_LEARNERS)


def cosine_average_first(v1: torch.Tensor, v2: torch.Tensor) -> float:
    similarity = cosine_similarity(
        v1.sum(axis=0).reshape(1, -1),
        v2.sum(axis=0).reshape(1, -1),
    )
    return float(similarity[0][0])

def evaluate_u_int(
    real: pd.DataFrame,
    synth: pd.DataFrame,
    treatment_col: str,
    outcome_col: str,
    n_units: int,
    binary_y: bool = False,
) -> float:
    real_train, test = _split_train_test(real)

    learner_real = TLearner(n_unit_in=n_units, binary_y=binary_y, seed=_random_seed())
    X_real = _feature_matrix(real_train, treatment_col, outcome_col)
    y_real = real_train[outcome_col].to_numpy()
    w_real = real_train[treatment_col].to_numpy()
    learner_real.fit(X_real, y_real, w_real)

    learner_synth = TLearner(n_unit_in=n_units, binary_y=binary_y, seed=_random_seed())
    X_synth = _feature_matrix(synth, treatment_col, outcome_col)
    y_synth = synth[outcome_col].to_numpy()
    w_synth = synth[treatment_col].to_numpy()
    learner_synth.fit(X_synth, y_synth, w_synth)

    X_test = _feature_matrix(test, treatment_col, outcome_col)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

    attr_real = IntegratedGradients(learner_real).attribute(X_test_tensor)
    attr_synth = IntegratedGradients(learner_synth).attribute(X_test_tensor)

    return cosine_average_first(attr_real, attr_synth)

def evaluate_u_policy(
    real: pd.DataFrame,
    synth: pd.DataFrame,
    treatment_col: str,
    outcome_col: str,
    n_units: int,
    binary_y: bool = False,
) -> float:
    real_train, test = _split_train_test(real)

    learner_real = TLearner(n_unit_in=n_units, binary_y=binary_y, seed=_random_seed())
    X_real = _feature_matrix(real_train, treatment_col, outcome_col)
    y_real = real_train[outcome_col].to_numpy()
    w_real = real_train[treatment_col].to_numpy()
    learner_real.fit(X_real, y_real, w_real)

    learner_synth = TLearner(n_unit_in=n_units, binary_y=binary_y, seed=_random_seed())
    X_synth = _feature_matrix(synth, treatment_col, outcome_col)
    y_synth = synth[outcome_col].to_numpy()
    w_synth = synth[treatment_col].to_numpy()
    learner_synth.fit(X_synth, y_synth, w_synth)

    X_test_tensor = torch.tensor(
        _feature_matrix(test, treatment_col, outcome_col),
        dtype=torch.float32,
    )

    pred_real = learner_real.predict(X_test_tensor)
    pred_synth = learner_synth.predict(X_test_tensor)

    product = (pred_real * pred_synth).cpu().detach().numpy()
    correct_rate = float(np.count_nonzero(product > 0) / max(len(product), 1))

    return correct_rate