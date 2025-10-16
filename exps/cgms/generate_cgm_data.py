"""
Graph discovery and CGM synthetic data generation. This module will:
* load the three real-world datasets (AIDS, IHDP, ACIC),
* construct the corresponding causal graphs (naïve, pruned, and PC-derived), and
* generate synthetic datasets using both the ANM and DCM approaches.
"""
from __future__ import annotations

import argparse
import logging
import random
from itertools import combinations
from pathlib import Path
from typing import Callable, Dict, Iterable, Mapping, MutableMapping, Optional, Sequence

import networkx as nx
import numpy as np
import pandas as pd

import dowhy.gcm as cy
from dowhy.gcm import InvertibleStructuralCausalModel, draw_samples

from causallearn.search.ConstraintBased.PC import pc
from model.diffusion import create_model_from_graph

LOGGER = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
DATA_DIR = REPO_ROOT / "data"

DEFAULT_DIFFUSION_PARAMS: Dict[str, object] = {
    "num_epochs": 500,
    "lr": 1e-4,
    "batch_size": 64,
    "hidden_dim": 64,
    "use_positional_encoding": False,
    "weight_decay": 0,
    "lambda_loss": 0,
    "clip": False,
    "verbose": False,
}

AIDS_COVARIATES = [
    "age",
    "wtkg",
    "cd40",
    "karnof",
    "cd80",
    "gender",
    "homo",
    "race",
    "drugs",
    "symptom",
    "str2",
    "hemo",
]
AIDS_CONTINUOUS = {"age", "wtkg", "cd40", "karnof", "cd80", "y"}

IHDP_COVARIATES = [f"x{i}" for i in range(1, 26)]
IHDP_CONTINUOUS = {"y_factual", "x1", "x2", "x3", "x4", "x5", "x6"}

ACIC_BINARY_COLUMNS = {"z"}

DATASET_ROLES = {
    "aids": {
        "treatment": "t",
        "outcome": "y",
        "covariates": AIDS_COVARIATES,
    },
    "ihdp": {
        "treatment": "treatment",
        "outcome": "y_factual",
        "covariates": IHDP_COVARIATES,
    },
    "acic": {
        "treatment": "z",
        "outcome": "y",
        # Covariates determined dynamically (all other columns).
        "covariates": None,
    },
}


# ---------------------------------------------------------------------------
# Dataset loading helpers
# ---------------------------------------------------------------------------

def load_aids_dataset() -> pd.DataFrame:
    """Load the pre-processed AIDS dataset."""
    csv_path = DATA_DIR / "aids_preprocessed.csv"
    LOGGER.info("Loading AIDS dataset from %s", csv_path)
    return pd.read_csv(csv_path)


def load_ihdp_dataset() -> pd.DataFrame:
    """Load the IHDP dataset, mirroring the notebook preprocessing."""
    csv_path = DATA_DIR / "ihdp.csv"
    LOGGER.info("Loading IHDP dataset from %s", csv_path)
    df = pd.read_csv(csv_path)
    df = df.drop(["y_cfactual", "mu0", "mu1"], axis=1)
    df["treatment"] = df["treatment"].astype(int)
    return df


def load_acic_dataset() -> pd.DataFrame:
    """Load the encoded ACIC dataset, checking both likely locations."""
    local_path = SCRIPT_DIR / "acic.csv"
    if local_path.exists():
        LOGGER.info("Loading ACIC dataset from %s", local_path)
        return pd.read_csv(local_path)

    csv_path = DATA_DIR / "acic.csv"
    LOGGER.info("Loading ACIC dataset from %s", csv_path)
    return pd.read_csv(csv_path)


DATASET_LOADERS: Mapping[str, Callable[[], pd.DataFrame]] = {
    "aids": load_aids_dataset,
    "ihdp": load_ihdp_dataset,
    "acic": load_acic_dataset,
}


# ---------------------------------------------------------------------------
# Graph construction utilities
# ---------------------------------------------------------------------------

def build_aids_graphs(data: pd.DataFrame, alpha: float) -> Mapping[str, nx.DiGraph]:
    """Create naïve, PC-discovered, and pruned AIDS graphs."""
    variables = AIDS_COVARIATES

    pairs_with_t = [(var, "t") for var in variables]
    pairs_with_y = [(var, "y") for var in variables]
    pair_t_y = [("t", "y")]
    sorted_vars = list(variables)
    unique_pairs = [
        (var1, var2)
        for idx, var1 in enumerate(sorted_vars)
        for var2 in sorted_vars[idx + 1 :]
    ]

    naive_graph = nx.DiGraph(pairs_with_t + pairs_with_y + pair_t_y + unique_pairs)
    naive_path = SCRIPT_DIR / "aids_naive_graph.gpickle"
    nx.write_gpickle(naive_graph, naive_path)
    LOGGER.info("Saved AIDS naïve graph to %s", naive_path)

    pc_graph = run_pc_algorithm(data, alpha=alpha, dataset_name="AIDS")
    pc_path = SCRIPT_DIR / "aids_pc_graph.gpickle"
    nx.write_gpickle(pc_graph, pc_path)
    legacy_pc_path = SCRIPT_DIR / "aids_graph.gpickle"
    nx.write_gpickle(pc_graph, legacy_pc_path)
    LOGGER.info("Saved AIDS PC graph to %s (legacy copy at %s)", pc_path, legacy_pc_path)

    roles = DATASET_ROLES["aids"]
    pruned_graph = prune_pc_graph(
        pc_graph,
        treatment=roles["treatment"],
        outcome=roles["outcome"],
        covariates=set(roles["covariates"]),
    )
    pruned_path = SCRIPT_DIR / "aids_pruned_graph.gpickle"
    nx.write_gpickle(pruned_graph, pruned_path)
    legacy_pruned_path = SCRIPT_DIR / "aids_graph_removed.gpickle"
    nx.write_gpickle(pruned_graph, legacy_pruned_path)
    LOGGER.info(
        "Saved pruned AIDS graph to %s (legacy copy at %s)",
        pruned_path,
        legacy_pruned_path,
    )

    return {"naive": naive_graph, "pc": pc_graph, "pruned": pruned_graph}


def build_ihdp_graphs(data: pd.DataFrame, alpha: float) -> Mapping[str, nx.DiGraph]:
    """Create naïve, PC-discovered, and pruned IHDP graphs."""
    variables = IHDP_COVARIATES
    unique_pairs = list(combinations(variables, 2))

    treatment_edges = [(var, "treatment") for var in variables]
    outcome_edges = [(var, "y_factual") for var in variables]
    pair_t_y = [("treatment", "y_factual")]

    naive_graph = nx.DiGraph(unique_pairs + treatment_edges + outcome_edges + pair_t_y)
    naive_path = SCRIPT_DIR / "ihdp_naive_graph.gpickle"
    nx.write_gpickle(naive_graph, naive_path)
    LOGGER.info("Saved IHDP naïve graph to %s", naive_path)

    pc_graph = run_pc_algorithm(data, alpha=alpha, dataset_name="IHDP")
    pc_path = SCRIPT_DIR / "ihdp_pc_graph.gpickle"
    nx.write_gpickle(pc_graph, pc_path)
    LOGGER.info("Saved IHDP PC graph to %s", pc_path)

    roles = DATASET_ROLES["ihdp"]
    pruned_graph = prune_pc_graph(
        pc_graph,
        treatment=roles["treatment"],
        outcome=roles["outcome"],
        covariates=set(roles["covariates"]),
    )
    pruned_path = SCRIPT_DIR / "ihdp_pruned_graph.gpickle"
    nx.write_gpickle(pruned_graph, pruned_path)
    LOGGER.info("Saved IHDP pruned graph to %s", pruned_path)

    return {"naive": naive_graph, "pc": pc_graph, "pruned": pruned_graph}


def run_pc_algorithm(
    data: pd.DataFrame,
    alpha: float = 0.05,
    dataset_name: Optional[str] = None,
) -> nx.DiGraph:
    """Run the PC causal discovery algorithm and convert the result to NetworkX."""
    label = dataset_name or "dataset"
    LOGGER.info("Running PC algorithm with alpha=%s on %s", alpha, label)
    data_np = data.to_numpy()
    pc_result = pc(data_np, alpha=alpha)

    # Map generic node labels (X1, X2, ...) back to original column names when needed.
    column_dict = {f"X{i + 1}": col for i, col in enumerate(data.columns)}

    graph = nx.DiGraph()
    for edge in pc_result.G.get_graph_edges():
        source_raw = edge.node1.get_name()
        target_raw = edge.node2.get_name()
        source = column_dict.get(source_raw, source_raw)
        target = column_dict.get(target_raw, target_raw)
        if source == target:
            continue
        graph.add_edge(source, target)

    return graph


def make_dag(graph: nx.DiGraph) -> nx.DiGraph:
    """Convert a directed graph to a DAG by removing edges that form cycles."""
    result = graph.copy()
    if nx.is_directed_acyclic_graph(result):
        LOGGER.info("Graph already a DAG; no cycles detected.")
        return result

    LOGGER.info("Cycles detected; removing edges to enforce DAG structure.")
    while not nx.is_directed_acyclic_graph(result):
        cycles = list(nx.simple_cycles(result))
        if not cycles:
            break
        cycle = cycles[0]
        edge_to_remove = (cycle[0], cycle[1])
        result.remove_edge(*edge_to_remove)
        LOGGER.debug("Removed edge %s to break cycle", edge_to_remove)

    LOGGER.info("DAG conversion complete.")
    return result


def prune_pc_graph(
    graph: nx.DiGraph,
    *,
    treatment: str,
    outcome: str,
    covariates: Iterable[str],
) -> nx.DiGraph:
    """Remove edges starting at the outcome or from treatment to covariates."""
    covariate_set = set(covariates)
    pruned = graph.copy()
    to_remove = [
        (source, target)
        for source, target in pruned.edges()
        if source == outcome or (source == treatment and target in covariate_set)
    ]
    if to_remove:
        pruned.remove_edges_from(to_remove)
        LOGGER.debug("Pruned edges: %s", to_remove)
    return pruned


def build_acic_graphs(data: pd.DataFrame, alpha: float = 0.05) -> Mapping[str, nx.DiGraph]:
    """Construct ACIC graphs: PC result and pruned PC graph."""
    roles = DATASET_ROLES["acic"]
    treatment = roles["treatment"]
    outcome = roles["outcome"]
    covariates = [col for col in data.columns if col not in {treatment, outcome}]

    pc_graph = run_pc_algorithm(data, alpha=alpha, dataset_name="ACIC")
    pc_path = SCRIPT_DIR / "acic_pc_graph.gpickle"
    nx.write_gpickle(pc_graph, pc_path)
    legacy_pc_path = SCRIPT_DIR / "acic_graph.gpickle"
    nx.write_gpickle(pc_graph, legacy_pc_path)
    LOGGER.info(
        "Saved ACIC PC graph to %s (legacy copy at %s)", pc_path, legacy_pc_path
    )

    pruned_graph = prune_pc_graph(
        pc_graph,
        treatment=treatment,
        outcome=outcome,
        covariates=set(covariates),
    )
    pruned_path = SCRIPT_DIR / "acic_pruned_graph.gpickle"
    nx.write_gpickle(pruned_graph, pruned_path)
    legacy_pruned_path = SCRIPT_DIR / "acic_graph_removed.gpickle"
    nx.write_gpickle(pruned_graph, legacy_pruned_path)
    LOGGER.info(
        "Saved ACIC pruned graph to %s (legacy copy at %s)",
        pruned_path,
        legacy_pruned_path,
    )

    return {"pc": pc_graph, "pruned": pruned_graph}


GRAPH_BUILDERS: Mapping[str, Callable[[pd.DataFrame, float], Mapping[str, nx.DiGraph]]] = {
    "aids": build_aids_graphs,
    "ihdp": build_ihdp_graphs,
    "acic": build_acic_graphs,
}


# ---------------------------------------------------------------------------
# Post-processing utilities
# ---------------------------------------------------------------------------

def _threshold_all_but(df: pd.DataFrame, exclude: Iterable[str]) -> pd.DataFrame:
    result = df.copy()
    binary_cols = [col for col in result.columns if col not in set(exclude)]
    if binary_cols:
        result[binary_cols] = (result[binary_cols] >= 0.5).astype(int)
    return result


def _threshold_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    result = df.copy()
    for column in columns:
        if column in result.columns:
            result[column] = (result[column] >= 0.5).astype(int)
    return result


# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------

def generate_anm_datasets(
    data: pd.DataFrame,
    graph: nx.DiGraph,
    output_dir: Path,
    file_pattern: str,
    count: int,
    start_index: int,
    postprocess: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info(
        "Generating %s ANM datasets in %s with pattern %s",
        count,
        output_dir,
        file_pattern,
    )

    for offset in range(count):
        idx = start_index + offset
        causal_model = InvertibleStructuralCausalModel(graph.copy())
        cy.auto.assign_causal_mechanisms(causal_model, data, override_models=True)
        cy.fit(causal_model, data)
        synth = draw_samples(causal_model, num_samples=len(data))
        synth = synth[data.columns]
        if postprocess is not None:
            synth = postprocess(synth)
        output_path = output_dir / file_pattern.format(i=idx)
        synth.to_csv(output_path, index=False)
        LOGGER.debug("Saved ANM dataset to %s", output_path)


def generate_dcm_datasets(
    data: pd.DataFrame,
    graph: nx.DiGraph,
    output_dir: Path,
    file_pattern: str,
    count: int,
    start_index: int,
    params: Optional[MutableMapping[str, object]] = None,
    postprocess: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    params = dict(DEFAULT_DIFFUSION_PARAMS if params is None else params)
    LOGGER.info(
        "Generating %s DCM datasets in %s with pattern %s",
        count,
        output_dir,
        file_pattern,
    )

    for offset in range(count):
        idx = start_index + offset
        diffusion_model = create_model_from_graph(graph, params)
        cy.fit(diffusion_model, data)
        synth = draw_samples(diffusion_model, num_samples=len(data))
        synth = synth[data.columns]
        if postprocess is not None:
            synth = postprocess(synth)
        output_path = output_dir / file_pattern.format(i=idx)
        synth.to_csv(output_path, index=False)
        LOGGER.debug("Saved DCM dataset to %s", output_path)


# ---------------------------------------------------------------------------
# Pipeline configuration
# ---------------------------------------------------------------------------

POSTPROCESSORS: Mapping[str, Callable[[pd.DataFrame], pd.DataFrame]] = {
    "aids": lambda df: _threshold_all_but(df, AIDS_CONTINUOUS),
    "ihdp": lambda df: _threshold_all_but(df, IHDP_CONTINUOUS),
    "acic": lambda df: _threshold_columns(df, ACIC_BINARY_COLUMNS),
}

ANM_CONFIG: Mapping[str, Mapping[str, Dict[str, object]]] = {
    "aids": {
        "naive": {
            "output_subdir": "aids_ANM",
            "file_pattern": "aids_ANM_naive_{i}.csv",
            "count": 20,
            "start": 0,
        },
        "pc": {
            "output_subdir": "aids_ANM",
            "file_pattern": "aids_ANM_pc_{i}.csv",
            "count": 20,
            "start": 0,
        },
        "pruned": {
            "output_subdir": "aids_ANM",
            "file_pattern": "aids_ANM_pruned_{i}.csv",
            "count": 20,
            "start": 0,
        },
    },
    "ihdp": {
        "naive": {
            "output_subdir": "ihdp_ANM",
            "file_pattern": "ihdp_ANM_naive_{i}.csv",
            "count": 20,
            "start": 0,
        },
        "pc": {
            "output_subdir": "ihdp_ANM",
            "file_pattern": "ihdp_ANM_pc_{i}.csv",
            "count": 20,
            "start": 0,
        },
        "pruned": {
            "output_subdir": "ihdp_ANM",
            "file_pattern": "ihdp_ANM_pruned_{i}.csv",
            "count": 20,
            "start": 0,
        },
    },
    "acic": {
        "pc": {
            "output_subdir": "acic_ANM",
            "file_pattern": "acic_ANM_pc_{i}.csv",
            "count": 20,
            "start": 0,
        },
        "pruned": {
            "output_subdir": "acic_ANM",
            "file_pattern": "acic_ANM_pruned_{i}.csv",
            "count": 20,
            "start": 0,
        },
    },
}

DCM_CONFIG: Mapping[str, Mapping[str, Dict[str, object]]] = {
    "aids": {
        "naive": {
            "output_subdir": "aids_DCM",
            "file_pattern": "aids_DCM_naive_{i}.csv",
            "count": 20,
            "start": 0,
        },
        "pc": {
            "output_subdir": "aids_DCM",
            "file_pattern": "aids_DCM_pc_{i}.csv",
            "count": 20,
            "start": 0,
        },
        "pruned": {
            "output_subdir": "aids_DCM",
            "file_pattern": "aids_DCM_pruned_{i}.csv",
            "count": 20,
            "start": 0,
        },
    },
    "ihdp": {
        "naive": {
            "output_subdir": "ihdp_DCM",
            "file_pattern": "ihdp_DCM_naive_{i}.csv",
            "count": 20,
            "start": 0,
        },
        "pc": {
            "output_subdir": "ihdp_DCM",
            "file_pattern": "ihdp_DCM_pc_{i}.csv",
            "count": 20,
            "start": 0,
        },
        "pruned": {
            "output_subdir": "ihdp_DCM",
            "file_pattern": "ihdp_DCM_pruned_{i}.csv",
            "count": 20,
            "start": 0,
        },
    },
    "acic": {
        "pc": {
            "output_subdir": "acic_DCM",
            "file_pattern": "acic_DCM_pc_{i}.csv",
            "count": 20,
            "start": 0,
        },
        "pruned": {
            "output_subdir": "acic_DCM",
            "file_pattern": "acic_DCM_pruned_{i}.csv",
            "count": 20,
            "start": 0,
        },
    },
}


def set_global_seed(seed: Optional[int]) -> None:
    if seed is None:
        return
    LOGGER.info("Setting global random seed to %s", seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        LOGGER.debug("torch not available; skipping torch seed setup.")


def run_dataset_pipeline(
    dataset: str,
    enabled_models: Sequence[str],
    anm_override: Optional[int],
    dcm_override: Optional[int],
    alpha: float,
) -> None:
    LOGGER.info("Starting pipeline for dataset: %s", dataset.upper())
    data_loader = DATASET_LOADERS[dataset]
    builder = GRAPH_BUILDERS[dataset]
    data = data_loader()
    graphs = builder(data, alpha)

    postprocess = POSTPROCESSORS[dataset]

    if "anm" in enabled_models:
        for graph_name, config in ANM_CONFIG[dataset].items():
            if graph_name not in graphs:
                LOGGER.warning(
                    "Graph '%s' not available for dataset %s; skipping ANM generation.",
                    graph_name,
                    dataset,
                )
                continue
            count = int(anm_override if anm_override is not None else config["count"])
            output_dir = SCRIPT_DIR / str(config["output_subdir"])
            generate_anm_datasets(
                data=data,
                graph=graphs[graph_name],
                output_dir=output_dir,
                file_pattern=str(config["file_pattern"]),
                count=count,
                start_index=int(config["start"]),
                postprocess=postprocess,
            )

    if "dcm" in enabled_models:
        for graph_name, config in DCM_CONFIG[dataset].items():
            if graph_name not in graphs:
                LOGGER.warning(
                    "Graph '%s' not available for dataset %s; skipping DCM generation.",
                    graph_name,
                    dataset,
                )
                continue
            count = int(dcm_override if dcm_override is not None else config["count"])
            output_dir = SCRIPT_DIR / str(config["output_subdir"])
            generate_dcm_datasets(
                data=data,
                graph=graphs[graph_name],
                output_dir=output_dir,
                file_pattern=str(config["file_pattern"]),
                count=count,
                start_index=int(config["start"]),
                postprocess=postprocess,
            )


def resolve_datasets(selection: Sequence[str]) -> Sequence[str]:
    if "all" in selection:
        return list(DATASET_LOADERS.keys())
    return selection


def resolve_models(selection: Sequence[str]) -> Sequence[str]:
    if "all" in selection:
        return ["anm", "dcm"]
    return selection


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run graph discovery and synthetic generation experiments."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["all"],
        choices=["all", "aids", "ihdp", "acic"],
        help="Datasets to process (default: all).",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["all"],
        choices=["all", "anm", "dcm"],
        help="Model types to run (default: both).",
    )
    parser.add_argument(
        "--anm-runs",
        type=int,
        default=None,
        help="Override the number of ANM datasets to generate.",
    )
    parser.add_argument(
        "--dcm-runs",
        type=int,
        default=None,
        help="Override the number of DCM datasets to generate.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance level for the PC algorithm.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducibility.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level (default: INFO).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level))
    LOGGER.debug("Parsed arguments: %s", args)

    set_global_seed(args.seed)

    datasets = resolve_datasets(args.datasets)
    models = resolve_models(args.models)

    for dataset in datasets:
        run_dataset_pipeline(
            dataset=dataset,
            enabled_models=models,
            anm_override=args.anm_runs,
            dcm_override=args.dcm_runs,
            alpha=args.alpha,
        )


if __name__ == "__main__":
    main()
