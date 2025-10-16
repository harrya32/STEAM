from synthcity.plugins import Plugins
import random
import numpy as np
import pandas as pd
from src.metrics import evaluate_f, evaluate_c, evaluate_jsd, evaluate_average_u_pehe
from catenets.models.torch import SLearner
from sklearn.linear_model import LogisticRegression

DEFAULT_GENERATOR_CONFIG = {
    'generator_n_layers_hidden': 2,
    'generator_n_units_hidden': 500,
    'generator_nonlin': 'relu',
}


def _random_state():
    return random.randint(0, 1_000_000)


def _get_generator(gen, hyperparam_overrides=None):
    config = {**DEFAULT_GENERATOR_CONFIG}
    if hyperparam_overrides:
        config.update(hyperparam_overrides)
    return Plugins().get(gen, random_state=_random_state(), **config)

def generate_stand_with_hyperparams(real, gen, hyperparam_overrides=None):
    g = _get_generator(gen, hyperparam_overrides)

    print(f'Fitting {gen} model')
    g.fit(real)
    print(f'Generating {gen} synthetic dataset')
    synth = g.generate(count = len(real)).dataframe()
    return synth

def generate_steam_with_hyperparams(real, gen, treatment_col, outcome_col, hyperparam_overrides=None):
    g = _get_generator(gen, hyperparam_overrides)
    feature_cols = [col for col in real.columns if col not in {treatment_col, outcome_col}]
    real_cov = real[feature_cols]
    print(f'Fitting {gen} covariate model')
    g.fit(real_cov)
    print(f'Generating {gen} synthetic covariates')
    synth_cov = g.generate(count=len(real)).dataframe()

    # generate propensities
    X = real_cov.to_numpy()
    y = real[treatment_col].to_numpy()

    classifier = LogisticRegression(random_state=_random_state(), max_iter=1_000)
    print('Fitting propensity model')
    classifier.fit(X, y)
    print('Generating propensities')
    probabilities = classifier.predict_proba(synth_cov.to_numpy())
    prob_class_1 = probabilities[:, 1]
    binary_outcomes = np.random.binomial(n=1, p=prob_class_1)

    synth = synth_cov.copy()
    synth[treatment_col] = pd.Series(binary_outcomes, index=synth_cov.index)

    synth[outcome_col] = 0.0

    y = real[outcome_col].to_numpy()
    w = real[treatment_col].to_numpy()
    n_units = len(feature_cols)

    learner = SLearner(n_unit_in=n_units, binary_y=False, seed=_random_state())
    print('Fitting CATE learner')
    learner.fit(X, y, w)

    seq_X = synth[feature_cols].to_numpy()
    print('Generating POs')
    _, y0, y1 = learner.predict(seq_X, return_po=True)

    y0 = np.asarray(y0.detach().cpu()).squeeze()
    y1 = np.asarray(y1.detach().cpu()).squeeze()
    treatment = synth[treatment_col].to_numpy()
    outcomes = np.where(treatment == 0, y0, y1)
    synth[outcome_col] = outcomes
    return synth

def hyperparam_exp(real, gen, treatment_col, outcome_col, hyperparams, n_iter, hyperparam_name='generator_n_layers_hidden', binary_y=False, save=False, fp=''):
    feature_cols = [col for col in real.columns if col not in {treatment_col, outcome_col}]
    n_units = len(feature_cols)
    records = []
    for h in hyperparams:
        for _ in range(n_iter):
            overrides = {hyperparam_name: h}
            stand = generate_stand_with_hyperparams(real, gen, overrides)
            steam = generate_steam_with_hyperparams(real, gen, treatment_col, outcome_col, overrides)

            records.extend([
                {
                    'method': 'generic',
                    'hyperparam_name': hyperparam_name,
                    'hyperparam': h,
                    'p_a,x': evaluate_f(real, stand, treatment_col, outcome_col),
                    'r_b,x': evaluate_c(real, stand, treatment_col, outcome_col),
                    'jsd_pi': evaluate_jsd(real, stand, treatment_col, outcome_col),
                    'u_pehe': evaluate_average_u_pehe(real, stand, treatment_col, outcome_col, n_units, binary_y),
                },
                {
                    'method': 'steam',
                    'hyperparam_name': hyperparam_name,
                    'hyperparam': h,
                    'p_a,x': evaluate_f(real, steam, treatment_col, outcome_col),
                    'r_b,x': evaluate_c(real, steam, treatment_col, outcome_col),
                    'jsd_pi': evaluate_jsd(real, steam, treatment_col, outcome_col),
                    'u_pehe': evaluate_average_u_pehe(real, steam, treatment_col, outcome_col, n_units, binary_y),
                },
            ])

    results = pd.DataFrame.from_records(
        records,
        columns=['method', 'hyperparam_name', 'hyperparam', 'p_a,x', 'r_b,x', 'jsd_pi', 'u_pehe'],
    )
    if save and fp:
        results.to_csv(fp, index=False)
    return results

def main():
    ihdp_full = pd.read_csv('../data/ihdp.csv')
    ihdp = ihdp_full.drop(['y_cfactual', 'mu0', 'mu1'], axis=1)
    ihdp['treatment'] = ihdp['treatment'].astype(int)

    experiments = [
        {
            'hyperparams': [3, 4, 5],
            'hyperparam_name': 'generator_n_layers_hidden',
            'fp': '../results/hyperparam_ihdp_ctgan_hidden_layers.csv',
        },
        {
            'hyperparams': [5, 50, 100, 300],
            'hyperparam_name': 'generator_n_units_hidden',
            'fp': '../results/hyperparam_ihdp_ctgan_hidden_units.csv',
        },
        {
            'hyperparams': ['leaky_relu', 'selu'],
            'hyperparam_name': 'generator_nonlin',
            'fp': '../results/hyperparam_ihdp_ctgan_nonlin.csv',
        },
    ]

    for exp in experiments:
        hyperparam_exp(
            ihdp,
            'ctgan',
            'treatment',
            'y_factual',
            exp['hyperparams'],
            5,
            hyperparam_name=exp['hyperparam_name'],
            save=True,
            fp=exp['fp'],
        )

if __name__ == "__main__":
    main()