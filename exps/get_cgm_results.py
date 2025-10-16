"""Reads in data from /cgms/ and computes metrics, compared to real datasets."""
import argparse
from typing import Tuple
import pandas as pd
from src.metrics import (
    evaluate_average_u_pehe,
    evaluate_c,
    evaluate_f,
    evaluate_jsd,
)

def load_ihdp() -> Tuple[pd.DataFrame, str, str, bool]:
    """Load IHDP dataset."""
    ihdp_full = pd.read_csv("../data/ihdp.csv")
    ihdp = ihdp_full.drop(["y_cfactual", "mu0", "mu1"], axis=1)
    ihdp["treatment"] = ihdp["treatment"].astype(int)
    return ihdp, "treatment", "y_factual", False


def load_acic() -> Tuple[pd.DataFrame, str, str, bool]:
    """Load ACIC dataset."""
    acic_full = pd.read_csv("../data/acic.csv")
    acic_full["y"] = acic_full["y0"]
    acic_full.loc[acic_full["z"] == 1, "y"] = acic_full.loc[acic_full["z"] == 1, "y1"]
    acic = acic_full.drop(["y0", "y1", "mu0", "mu1"], axis=1)
    return acic, "z", "y", False


def load_actg() -> Tuple[pd.DataFrame, str, str, bool]:
    """Load ACTG dataset."""
    actg = pd.read_csv("../data/aids_preprocessed.csv")
    return actg, "t", "y", False


def compare_dataset(
    dataset_name: str,
    model: str,
    graph: str,
    n_iter: int = 20,
    classifier: str = "logistic",
) -> pd.DataFrame:
    """
    Compare synthetic data to real data for a given dataset.
    
    Args:
        dataset_name: Name of the dataset ('ihdp', 'acic', or 'actg')
        model: Model name (e.g., 'ANM', 'DCM')
        graph: Graph type (e.g., 'pc', 'naive', 'pruned')
        n_iter: Number of iterations to evaluate
        classifier: Classifier type for JSD evaluation
        
    Returns:
        DataFrame with evaluation results
    """
    # Load the appropriate dataset
    loaders = {
        "ihdp": load_ihdp,
        "acic": load_acic,
        "actg": load_actg,
    }
    
    if dataset_name not in loaders:
        raise ValueError(f"Unknown dataset: {dataset_name}. Must be one of {list(loaders.keys())}")
    
    real_data, treatment_col, outcome_col, binary_y = loaders[dataset_name]()
    folder = f"./cgms/{dataset_name}_{model}/"
    results = pd.DataFrame(columns=["model", "graph", "p_a,x", "r_b,x", "jsd_pi", "u_pehe"])

    for i in range(n_iter):
        fp = f"{folder}{dataset_name}_{model}_{graph}_{i}.csv"
        try:
            synth_data = pd.read_csv(fp).drop(["y_cfactual", "mu0", "mu1"], axis=1)
        except FileNotFoundError:
            print(f"Warning: File not found: {fp}")
            continue
        print("Shape of synthetic data:", synth_data.shape)
        print("Shape of real data:", real_data.shape)
        n_units = len(real_data.drop([treatment_col, outcome_col], axis=1).columns)
        results.loc[len(results)] = [
            model,
            graph,
            evaluate_f(real_data, synth_data, treatment_col, outcome_col),
            evaluate_c(real_data, synth_data, treatment_col, outcome_col),
            evaluate_jsd(real_data, synth_data, treatment_col, outcome_col, classifier),
            evaluate_average_u_pehe(real_data, synth_data, treatment_col, outcome_col, n_units, binary_y),
        ]
    
    return results

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate synthetic CGM data against real datasets"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["ihdp", "acic", "actg"],
        choices=["ihdp", "acic", "actg"],
        help="Datasets to evaluate (default: all)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["ANM", "DCM"],
        help="Models to evaluate (default: ANM DCM)",
    )
    parser.add_argument(
        "--graphs",
        nargs="+",
        default=["pc", "naive", "pruned"],
        help="Graph types to evaluate (default: pc naive pruned)",
    )
    parser.add_argument(
        "--n-iter",
        type=int,
        default=20,
        help="Number of iterations to evaluate (default: 20)",
    )
    parser.add_argument(
        "--classifier",
        default="logistic",
        help="Classifier type for JSD evaluation (default: logistic)",
    )
    parser.add_argument(
        "--output-dir",
        default="../results",
        help="Output directory for results (default: ../results)",
    )
    return parser.parse_args()


def main():
    """Main function to run CGM results evaluation."""
    args = parse_args()
    
    print(f"Evaluating datasets: {args.datasets}")
    print(f"Models: {args.models}")
    print(f"Graphs: {args.graphs}")
    print(f"Iterations: {args.n_iter}")
    
    # Dictionary to store results for each dataset
    all_results = {dataset: pd.DataFrame() for dataset in args.datasets}

    for dataset in args.datasets:
        print(f"\nProcessing {dataset.upper()} dataset...")
        for model in args.models:
            for graph in args.graphs:
                print(f"  Evaluating {model} with {graph} graph...")
                results = compare_dataset(
                    dataset_name=dataset,
                    model=model,
                    graph=graph,
                    n_iter=args.n_iter,
                    classifier=args.classifier,
                )
                all_results[dataset] = pd.concat(
                    [all_results[dataset], results], ignore_index=True
                )

    # Save results
    for dataset in args.datasets:
        output_path = f"{args.output_dir}/{dataset}_cgm.csv"
        all_results[dataset].to_csv(output_path, index=False)
        print(f"\nSaved {dataset} results to {output_path}")


if __name__ == "__main__":
    main()