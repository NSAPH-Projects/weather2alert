import itertools
import json

import numpy as np
import pandas as pd
import torch

# import matplotlib.pyplot as plt

WESTERN_STATES = [
    "AZ",
    "CA",
    "CO",
    "ID",
    "MT",
    "NM",
    "NV",
    "OR",
    "WA",
    "ND",
    "SD",
    "NE",
    "KS",
] # ND, SD, NE and KS together only add 10 counties to the cold western group

SOUTHERN_STATES = [
    "TX", "OK", "AR", "LA", "MS", "AL", "GA", "FL", "TN", "KY", "SC", "NC", 
    "VA", "WV", "VA", "MD", "DE", 
    "NM", "AZ", "CA" # including CA in South based on the specific counties in our final 30
]


def get_similar_counties(dir: str):
    """Returns a dictionary that assigns similar counties to each county based
    on climate zones
    """
    spatial_feats = pd.read_parquet(f"{dir}/spatial_feats.parquet")
    with open(f"{dir}/fips2state.json", "r") as io:
        fips2state = json.load(io)
    fips2state = {int(k):v.strip() for k,v in fips2state.items()}
    # zones = ["Cold", "Hot-Dry", "Marine", "Mixed-Dry", "Mixed-Humid", "Very Cold"] # missing "Hot-Humid" due to one-hot encoding 
    spatial_feats = (
        spatial_feats.rename(
            {
                "Cold": "cold",
                "Hot-Dry": "hot_dry",
                "Marine": "marine",
                "Mixed-Dry": "mixed_dry",
                "Mixed-Humid": "mixed_humid",
                "Very Cold": "very_cold",
            },
            axis=1,
        )
        .assign(state=spatial_feats.index.map(fips2state))
        .assign(western=lambda x: x["state"].isin(WESTERN_STATES))
        .assign(cold_west=lambda x: x["cold"] * x["western"])
        .assign(cold_east=lambda x: x["cold"] * ~x["western"])
        .drop(columns="cold")
    )
    zones = [
        "cold_west",
        "cold_east",
        "hot_dry",
        "marine",
        "mixed_dry",
        "mixed_humid",
        "very_cold",
    ]
    similar_counties = {}
    # groupby climate vars and collect indices in array
    grouped = spatial_feats.reset_index().groupby(zones).agg({"fips": list})
    for i, row in grouped.iterrows():
        for fips in row["fips"]:
            similar_counties[fips] = row["fips"]
    return similar_counties


def load_rl_states_data(dir: str, HI_restriction: float, incorp_forecasts: bool):
    """Loads states RL training data for all counties and years.

    Args:
        dir (str): path to directory where the data processed for the bayesian model is stored

    Returns:
        base_dict (dict): dictionary where each key is a features used for the base hospitalizations
            model and values are dataframes with index (fips, year) and columns corresponding
            to day of summer.
        eff_dict (dict): similar as base_dict but the features for the alert effectiveness.
        extra_dict (dict): similar structure as previous dicts but contains the features that
            are neither part of the baseline or effectiveness models.
    """
    # we could use the HeatAlertDataModule here,
    # but used the appraoch of only loading what's needed
    states = pd.read_parquet(f"{dir}/states.parquet").drop(columns="intercept")
    states = states.rename({"quant_HI_county": "heat_qi"}, axis=1)
    states["hi_mean"] = (states.HI_mean - states.HI_mean.mean()) / states.HI_mean.std()
    states["future"] = pd.read_parquet(f"{dir}/abs_HI.parquet") # using absolute HI here so we can apply known forecast accuracy
    states = states.rename({"quant_HI_county": "heat_qi"}, axis=1)
    quarters = pd.read_parquet(f"{dir}/future_quarters.parquet")
    states = pd.concat([states, quarters], axis=1)
    for q in range(5,11):
        states["q" + str(q) + "0"] = pd.read_parquet(f"{dir}/future_q" + str(q) + "0.parquet")
    nws_alerts = pd.read_parquet(f"{dir}/actions.parquet")
    budgets = pd.read_parquet(f"{dir}/budget.parquet")
    st_ab = pd.read_parquet(f"{dir}/state.parquet").state
    sind = pd.read_parquet(f"{dir}/location_indicator.parquet").sind
    year = pd.read_parquet(f"{dir}/year.parquet").year
    with open(f"{dir}/fips2idx.json", "r") as io:
        fips2idx = json.load(io)
    idx2fips = {v: k for k, v in fips2idx.items()}
    # dos2index = {x: i for i, x in enumerate(sorted(states.dos_0.unique()))}
    n_counties = len(sind.unique())
    n_years = len(year.unique())
    n_days = int(sum(year == 2006)/n_counties)
    dos_index = list(itertools.chain(*[np.arange(0,n_days) for i in np.arange(0,n_years*n_counties)]))
    base_feat_names = [
        "heat_qi",
        "heat_qi_above_25",
        "heat_qi_above_75",
        "excess_heat",
        "weekend",
        "dos_0",
        "dos_1",
        "dos_2",
    ]
    base = (
        states[base_feat_names]
        .assign(fips=sind.map(idx2fips).astype(int))
        # .assign(dos_index=states["dos_0"].map(dos2index))
        .assign(dos_index=dos_index)
        .assign(year=year)
    )
    eff_feat_names = [
        "heat_qi",
        "excess_heat",
        "weekend",
        "dos_0",
        "dos_1",
        "dos_2",
    ]
    eff = (
        states[eff_feat_names]
        .assign(fips=sind.map(idx2fips).astype(int))
        # .assign(dos_index=states["dos_0"].map(dos2index))
        .assign(dos_index=dos_index)
        .assign(year=year)
    )
    extra_feats = [
        "hi_mean", "future", 'T4_1', 'T4_2', 'T4_3', 'T4_4', 
        "q50", "q60", "q70", "q80", "q90", "q100"
    ]
    extra = (
        states[extra_feats]
        .assign(fips=sind.map(idx2fips).astype(int))
        # .assign(dos_index=states["dos_0"].map(dos2index))
        .assign(dos_index=dos_index)
        .assign(year=year)
    )
    other = pd.concat([nws_alerts, budgets, year], axis=1)
    other.columns = ["nws_alert", "budget", "y"]
    other["south"] = [s.strip() in SOUTHERN_STATES for s in st_ab]
    other_vars = (
        other.assign(fips=sind.map(idx2fips).astype(int))
        .assign(dos_index=dos_index)
        .assign(year=year)
    )
    base_dict = {}
    for f in base_feat_names:
        D = base[[f, "fips", "year", "dos_index"]]
        # reshape where columns are dos_index and "loc, year" are the indices
        D = D.pivot(index=["fips", "year"], columns="dos_index", values=f)
        base_dict[f"baseline_{f}"] = D
    eff_dict = {}
    for f in eff_feat_names:
        D = eff[[f, "fips", "year", "dos_index"]]
        D = D.pivot(index=["fips", "year"], columns="dos_index", values=f)
        eff_dict[f"effectiveness_{f}"] = D
    extra_dict = {}
    for f in extra_feats:
        D = extra[[f, "fips", "year", "dos_index"]]
        D = D.pivot(index=["fips", "year"], columns="dos_index", values=f)
        extra_dict[f] = D
    other_dict = {}
    for f in other.columns:
        D = other_vars[[f, "fips", "year", "dos_index"]]
        D = D.pivot(index=["fips", "year"], columns="dos_index", values=f)
        other_dict[f] = D
    if incorp_forecasts:
        ## Get "forecasts": already have abs_HI in extra_dict["future"]
        qhi = base_dict['baseline_heat_qi']
        # Just the number of eligible days:
        eligible = (qhi >= HI_restriction).astype("int")
        eligible_sum = np.cumsum(eligible, axis=1)
        extra_dict["future_eligible"] = np.subtract(np.broadcast_to(np.max(eligible_sum, axis=1), (n_days, qhi.shape[0])).T, eligible_sum)
        # Number of repeated eligible days:
        a = eligible.loc[:,0:(n_days-2)]
        b = eligible.loc[:,1:]
        b.columns = a.columns
        rep_elig = (2*b - a) == 1
        RE_sum = np.cumsum(rep_elig, axis=1)
        RE_sum.columns = RE_sum.columns + 1
        RE_sum[0] = 0
        cols = RE_sum.columns.tolist()
        RE_sum = RE_sum[cols[-1:] + cols[:-1]]
        extra_dict["future_rep_elig"] = np.subtract(np.broadcast_to(np.max(RE_sum, axis=1), (n_days, qhi.shape[0])).T, RE_sum)
    return base_dict, eff_dict, extra_dict, other_dict


def subset_rl_states(
    county: int,
    counties: list[int],
    years: list[int] | None,
    base_dict: dict,
    eff_dict: dict,
    extra_dict: dict,
    other_dict: dict,
) -> tuple[dict, dict, dict, dict]:
    if years is None:
        index_vals = next(iter(base_dict.values())).index.get_level_values("year")
        years = sorted(index_vals.unique())
    idxs = list(itertools.product(counties, years))
    base_dict = {k: v.loc[idxs] for k, v in base_dict.items()}
    eff_dict = {k: v.loc[idxs] for k, v in eff_dict.items()}
    extra_dict = {k: v.loc[idxs] for k, v in extra_dict.items()}
    other_dict = {k: v.loc[idxs] for k, v in other_dict.items()}
    # ## Ensure all the budgets are just for the county of interest:
    # counties = [county]*len(counties)
    # new_idxs = list(itertools.product(counties, years))
    # other_dict = {k: v.loc[new_idxs] for k, v in other_dict.items()}
    return base_dict, eff_dict, extra_dict, other_dict


def dict_as_tensor(dict):
    """Converts a dictionary of dataframes to a dictionary of tensors"""
    return {k: torch.FloatTensor(v.values) for k, v in dict.items()}


def load_rl_states_by_county(
    county: int,
    dir: str,
    years: list[int] | None = None,
    match_similar: bool = False,
    include_COI: bool = True,
    as_tensors: bool = False,
    incorp_forecasts: bool = False,
    HI_restriction: float = 0.8,
) -> tuple[dict, dict, dict, dict]:
    """Loads states RL training data for a single county and years"""
    base_dict, eff_dict, extra_dict, other_dict = load_rl_states_data(dir, HI_restriction, incorp_forecasts)

    if match_similar:
        similar_counties = get_similar_counties(dir)
        counties = similar_counties[county]
        if not include_COI:
            counties.remove(county)
    else:
        counties = [county]

    base_dict, eff_dict, extra_dict, other_dict = subset_rl_states(
        county, counties, years, base_dict, eff_dict, extra_dict, other_dict
    )

    if as_tensors:
        base_dict = dict_as_tensor(base_dict)
        eff_dict = dict_as_tensor(eff_dict)
        extra_dict = dict_as_tensor(extra_dict)
        other_dict = dict_as_tensor(other_dict)

    return base_dict, eff_dict, extra_dict, other_dict


if __name__ == "__main__":
    dir = "data/processed"
    county = 48453
    years = None
    base_dict, eff_dict, extra_dict = load_rl_states_data(dir)
    similar_counties = get_similar_counties(dir)
    counties = similar_counties[county]
    base_dict, eff_dict, extra_dict = subset_rl_states(
        counties, years, base_dict, eff_dict, extra_dict
    )

# %%
