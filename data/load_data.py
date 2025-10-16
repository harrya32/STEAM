from io import BytesIO

import numpy as np
import pandas as pd
import requests
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder


JOBS_COLUMNS = [
    "training",  # Treatment assignment indicator
    "age",  # Age of participant
    "education",  # Years of education
    "black",  # Indicate whether individual is black
    "hispanic",  # Indicate whether individual is hispanic
    "married",  # Indicate whether individual is married
    "no_degree",  # Indicate if individual has no high-school diploma
    "re75",  # Real earnings in 1975, prior to study participation
    "re78",  # Real earnings in 1978, after study end
]

IHDP_COLUMNS = [
    "treatment",  # Treatment assignment indicator
    "y_factual",  # Observed outcome
    "y_cfactual",  # Counterfactual outcome (not observed)
    "mu0",  # True potential outcome under control (not observed)
    "mu1",  # True potential outcome under treatment (not observed)
    "x1",
    "x2",
    "x3",
    "x4",
    "x5",
    "x6",
    "x7",
    "x8",
    "x9",
    "x10",
    "x11",
    "x12",
    "x13",
    "x14",
    "x15",
    "x16",
    "x17",
    "x18",
    "x19",
    "x20",
    "x21",
    "x22",
    "x23",
    "x24",
    "x25",
]

JOBS_FILE_URLS = (
    "http://www.nber.org/~rdehejia/data/nsw_treated.txt",
    "http://www.nber.org/~rdehejia/data/nsw_control.txt",
)

ACIC_X_URL = "https://raw.githubusercontent.com/BiomedSciAI/causallib/master/causallib/datasets/data/acic_challenge_2016/x.csv"
ACIC_Y_URL = "https://raw.githubusercontent.com/BiomedSciAI/causallib/master/causallib/datasets/data/acic_challenge_2016/zymu_1.csv"

ACTG_SOURCE_URL = "https://raw.githubusercontent.com/tobhatt/CorNet/main/data/actg175.csv"
ACTG_RAW_OUTPUT = "actg175.csv"
ACTG_RCT_COLUMNS = [
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
    "t",
    "y",
]

IHDP_SOURCE_URL = (
    "https://raw.githubusercontent.com/AMLab-Amsterdam/CEVAE/master/datasets/IHDP/csv/ihdp_npci_1.csv"
)
IHDP_RAW_OUTPUT = "ihdp_npci_1.csv"

def encode(real: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode ACIC categorical columns."""

    encoder = OneHotEncoder()
    categorical = ["x_2", "x_21", "x_24"]
    encoded = encoder.fit_transform(real[categorical])
    encoded_df = pd.DataFrame(encoded.toarray(), columns=encoder.get_feature_names_out(categorical))
    real_encoded = pd.concat([real, encoded_df], axis=1)
    real_encoded.drop(categorical, axis=1, inplace=True)
    return real_encoded

def prepare_jobs_dataset(output_csv: str = "jobs_small.csv") -> pd.DataFrame:
    """Download, shuffle, and persist the Lalonde jobs dataset."""

    frames = [
        pd.read_csv(url, delim_whitespace=True, header=None, names=JOBS_COLUMNS)
        for url in JOBS_FILE_URLS
    ]
    lalonde = pd.concat(frames, ignore_index=True)
    lalonde = lalonde.sample(frac=1.0, random_state=42).reset_index(drop=True)
    lalonde.to_csv(output_csv, index=False)
    return lalonde


def prepare_acic_dataset(output_csv: str = "acic.csv") -> pd.DataFrame:
    """Download the ACIC challenge data and persist the combined frame."""

    features = pd.read_csv(ACIC_X_URL)
    targets = pd.read_csv(ACIC_Y_URL)
    dataset = pd.concat([features, targets], axis=1)
    dataset = encode(dataset)
    dataset.to_csv(output_csv, index=False)
    return dataset


def sample_rct(actg175: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the ACTG175 trial to a binary treatment RCT sample."""

    # AIDS dataset preprocessing steps from
    # https://github.com/tobhatt/CorNet/blob/main/real_world_exp/load_rct_data.py

    cd4_baseline = actg175["cd40"]
    cd4_20 = actg175["cd420"]

    outcome = (cd4_20 - cd4_baseline).values
    outcome_norm = preprocessing.scale(outcome)

    cov_cont = actg175[["age", "wtkg", "cd40", "karnof", "cd80"]]
    cov_cont_norm = preprocessing.scale(cov_cont)

    cov_bin = actg175[["gender", "homo", "race", "drugs", "symptom", "str2", "hemo"]]
    cov_bin_val = cov_bin.values
    t = actg175[["arms"]].values

    data = np.concatenate(
        (cov_cont_norm, cov_bin_val, t.reshape(-1, 1), outcome_norm.reshape(-1, 1)),
        axis=1,
    )

    # Only focus on one arm (0=zidovudine, 1=zidovudine and didanosine,
    #  2=zidovudine and zalcitabine, 3=didanosine)
    treatment_arm = 2
    control_arm = 0
    treatment_mask = (t == control_arm) + (t == treatment_arm)

    data_rct = data[treatment_mask.flatten()]
    data_rct[:, -2] = np.where(data_rct[:, -2] == treatment_arm, 1, 0)

    return pd.DataFrame(data_rct, columns=ACTG_RCT_COLUMNS)


def prepare_actg_dataset(
    source_url: str = ACTG_SOURCE_URL,
    raw_output_path: str = ACTG_RAW_OUTPUT,
    processed_output_csv: str = "aids_preprocessed.csv",
) -> pd.DataFrame:
    """Download the ACTG175 trial data, preprocess, and persist outputs."""

    response = requests.get(source_url)
    response.raise_for_status()

    with open(raw_output_path, "wb") as file:
        file.write(response.content)

    actg175 = pd.read_csv(raw_output_path, header=0, index_col=0)
    aids_preprocessed = sample_rct(actg175)
    aids_preprocessed.to_csv(processed_output_csv, index=False)
    return aids_preprocessed

def prepare_ihdp_dataset(
    source_url: str = IHDP_SOURCE_URL,
    raw_output_path: str | None = None,
    output_csv: str = "ihdp.csv",
) -> pd.DataFrame:
    """Download the IHDP dataset, apply column labels, and persist output."""

    response = requests.get(source_url)
    response.raise_for_status()

    content = response.content

    if raw_output_path:
        with open(raw_output_path, "wb") as file:
            file.write(content)

    ihdp_full = pd.read_csv(BytesIO(content), header=None, names=IHDP_COLUMNS)
    ihdp_full.to_csv(output_csv, index=False)
    return ihdp_full


def main() -> None:
    """Produce standardized datasets for downstream experiments."""

    prepare_jobs_dataset()
    prepare_acic_dataset()
    prepare_actg_dataset()
    prepare_ihdp_dataset(raw_output_path=IHDP_RAW_OUTPUT)


if __name__ == "__main__":
    main()
