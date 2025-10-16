from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from catenets.models.torch import TLearner
from src.metrics import (
    evaluate_average_u_pehe,
    evaluate_c,
    evaluate_f,
    evaluate_jsd,
)
from synthcity.plugins import Plugins

SEED_UPPER_BOUND = 1_000_000
DEFAULT_SEED = 0


@dataclass(frozen=True)
class DatasetConfig:
    """Container describing the configuration for a dataset."""

    loader: Callable[[Path], pd.DataFrame]
    treatment_col: str
    outcome_col: str
    output_stub: str
    binary_y: bool = False


def load_acic(data_dir: Path) -> pd.DataFrame:
    """Load and prepare the ACIC dataset."""

    acic_full = pd.read_csv(data_dir / 'acic.csv')
    acic_full['y'] = acic_full['y0']
    acic_full.loc[acic_full['z'] == 1, 'y'] = acic_full.loc[acic_full['z'] == 1, 'y1']
    return acic_full.drop(['y0', 'y1', 'mu0', 'mu1'], axis=1)


def load_ihdp(data_dir: Path) -> pd.DataFrame:
    """Load and prepare the IHDP dataset."""

    ihdp_full = pd.read_csv(data_dir / 'ihdp.csv')
    ihdp = ihdp_full.drop(['y_cfactual', 'mu0', 'mu1'], axis=1)
    ihdp['treatment'] = ihdp['treatment'].astype(int)
    return ihdp


def load_aids(data_dir: Path) -> pd.DataFrame:
    """Load the AIDS dataset."""

    return pd.read_csv(data_dir / 'aids_preprocessed.csv')


DATASET_CONFIGS: dict[str, DatasetConfig] = {
    'acic': DatasetConfig(
        loader=load_acic,
        treatment_col='z',
        outcome_col='y',
        output_stub='ablation_acic',
    ),
    'ihdp': DatasetConfig(
        loader=load_ihdp,
        treatment_col='treatment',
        outcome_col='y_factual',
        output_stub='ablation_ihdp',
    ),
    'aids': DatasetConfig(
        loader=load_aids,
        treatment_col='t',
        outcome_col='y',
        output_stub='ablation_aids',
    ),
}


def make_rng(seed: int | None = None) -> np.random.Generator:
    """Return a NumPy random generator with an optional seed."""

    return np.random.default_rng(seed)


def next_seed(
    rng: np.random.Generator, *, low: int = 0, high: int = SEED_UPPER_BOUND
) -> int:
    """Draw an integer seed from the provided generator."""

    return int(rng.integers(low, high))


def ensure_parent_dir(path: Path) -> None:
    """Create the parent directories for the provided path if needed."""

    path.parent.mkdir(parents=True, exist_ok=True)


def generate_sequential_no_prop(
    real: pd.DataFrame,
    gen: str,
    treatment_col: str,
    outcome_col: str,
    *,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Generate sequential synthetic data without propensity matching."""

    rng = rng or make_rng()

    feature_cols = [col for col in real.columns if col not in (treatment_col, outcome_col)]
    g = Plugins().get(gen, random_state=next_seed(rng))
    real_X_W = real.drop([outcome_col], axis=1)
    print(f'Fitting {gen} X and W model')
    g.fit(real_X_W)
    print(f'Generating {gen} synthetic X and W')
    synth = g.generate(count=len(real)).dataframe()
    synth[outcome_col] = 0

    # generate the outcome using counterfactual predictions
    X = np.array(real[feature_cols])
    y = np.array(real[outcome_col])
    w = np.array(real[treatment_col])
    n_units = len(feature_cols)

    l = TLearner(n_unit_in=n_units, binary_y=False, seed=next_seed(rng))
    print('Fitting CATE learner')
    l.fit(X, y, w)

    seq_X = np.array(synth[feature_cols])
    print('Generating POs')
    _, y0, y1 = l.predict(seq_X, return_po=True)

    treatment = synth[treatment_col].to_numpy()
    y0 = np.asarray(y0.detach().cpu()).reshape(-1)
    y1 = np.asarray(y1.detach().cpu()).reshape(-1)
    synth[outcome_col] = np.where(treatment == 0, y0, y1)
    return synth


def ablation_exp(
    real: pd.DataFrame,
    gen: str,
    treatment_col: str,
    outcome_col: str,
    n_iter: int,
    *,
    binary_y: bool = False,
    seed: int | None = None,
) -> pd.DataFrame:
    """Run the ablation experiment and return evaluation metrics."""

    rng = make_rng(seed)
    results = pd.DataFrame(columns=['method', 'p_a,x', 'r_b,x', 'jsd_pi', 'u_pehe'])
    feature_cols = [col for col in real.columns if col not in (treatment_col, outcome_col)]
    n_units = len(feature_cols)
    for _ in range(n_iter):
        ablation = generate_sequential_no_prop(
            real,
            gen,
            treatment_col,
            outcome_col,
            rng=rng,
        )

        results.loc[len(results)] = [
            'ablation',
            evaluate_f(real, ablation, treatment_col, outcome_col),
            evaluate_c(real, ablation, treatment_col, outcome_col),
            evaluate_jsd(real, ablation, treatment_col, outcome_col),
            evaluate_average_u_pehe(
                real,
                ablation,
                treatment_col,
                outcome_col,
                n_units,
                binary_y,
            ),
        ]

    return results


def positive_int(value: str) -> int:
    """Parse a strictly positive integer from the CLI."""

    parsed_value = int(value)
    if parsed_value < 1:
        raise argparse.ArgumentTypeError('value must be a positive integer')
    return parsed_value


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the ablation experiments."""

    parser = argparse.ArgumentParser(description='Run ablation experiments.')
    parser.add_argument(
        '--dataset',
        choices=sorted(DATASET_CONFIGS.keys()) + ['all'],
        default='all',
        help="Dataset to evaluate; use 'all' to process every dataset.",
    )
    parser.add_argument(
        '--n-iter',
        type=positive_int,
        default=5,
        help='Number of iterations per dataset.',
    )
    parser.add_argument(
        '--generator',
        default='tvae',
        help='Name of the SynthCity generator to use.',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=DEFAULT_SEED,
        help='Seed for the experiment RNG.',
    )
    parser.add_argument(
        '--data-dir',
        type=Path,
        default=None,
        help='Optional path to override the default data directory.',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=None,
        help='Optional path to override the default results directory.',
    )
    parser.add_argument(
        '--no-save',
        dest='save',
        action='store_false',
        help='Skip writing results to disk.',
    )
    parser.set_defaults(save=True)
    return parser.parse_args()


def run_dataset(
    dataset_name: str,
    config: DatasetConfig,
    *,
    generator: str,
    n_iter: int,
    seed: int,
    binary_y: bool,
    data_dir: Path,
    results_dir: Path,
    save: bool,
) -> pd.DataFrame:
    """Run the ablation experiment for a single dataset."""

    print(f'Running {dataset_name} for {n_iter} iterations with generator {generator}.')
    real = config.loader(data_dir)
    results = ablation_exp(
        real,
        generator,
        config.treatment_col,
        config.outcome_col,
        n_iter,
        binary_y=binary_y,
        seed=seed,
    )

    if save:
        output_path = results_dir / f'{config.output_stub}_{generator}.csv'
        ensure_parent_dir(output_path)
        results.to_csv(output_path, index=False)
        print(f'Saved results to {output_path}')

    return results


def main() -> None:
    """Execute ablation experiments based on CLI arguments."""

    args = parse_args()
    base_dir = Path(__file__).resolve().parent
    data_dir = args.data_dir or (base_dir.parent / 'data')
    results_dir = args.output_dir or (base_dir / 'results')

    selected_datasets = (
        DATASET_CONFIGS.keys() if args.dataset == 'all' else [args.dataset]
    )

    for name in selected_datasets:
        config = DATASET_CONFIGS[name]
        run_dataset(
            name,
            config,
            generator=args.generator,
            n_iter=args.n_iter,
            seed=args.seed,
            binary_y=config.binary_y,
            data_dir=data_dir,
            results_dir=results_dir,
            save=args.save,
        )


if __name__ == '__main__':
    main()