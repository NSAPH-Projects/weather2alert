import pandas as pd
import logging
import numpy as np
from scipy.special import expit


LOGGER = logging.getLogger(__name__)


def load_hosps(
    data_path: str,
    **kwargs,
    # confounders: pd.DataFrame,
    # exogenous_states: pd.DataFrame,
    # endogenous_states_actions: pd.DataFrame
):
    """Loads hospitalizations from data_path"""
    LOGGER.info("Loading hospitalizations from data_path")
    hosps = pd.read_parquet(data_path)

    # rename total_count to eligible_pop
    hosps = hosps.rename(
        columns={"other_hosps": "hospitalizations", "total_count": "eligible_pop"}
    )

    # # make an right join to index by pd
    # merged = pd.merge(exogenous_states, endogenous_states_actions, on=["fips", "date"])
    # hosps = pd.merge(merged, hosps, on=["fips", "date"], how="right")
    # hosps = hosps[["fips", "date", "hospitalizations", "eligible_pop"]]

    return hosps


def sim_hosps(
    sim_coefs: dict,
    confounders: pd.DataFrame,
    exogenous_states: pd.DataFrame,
    endogenous_states_actions: pd.DataFrame,
):
    """Simulates or loads hospitalizations"""
    confounders = confounders.set_index("fips")
    merged = pd.merge(exogenous_states, endogenous_states_actions, on=["fips", "date"])

    # split cases to obtain hostpitalizations
    LOGGER.info("Simulating hospitalizations")
    baseline = np.zeros(merged.shape[0])
    effectiveness = np.zeros(merged.shape[0])
    merged["intercept"] = 1

    # equal to all locations
    for b, w in sim_coefs.features.baseline.items():
        baseline += merged[b] * w

    for e, w in sim_coefs.features.effectiveness.items():
        effectiveness += merged[b] * w

    # now location specific
    state_cols = ["heat_qi", "excess_heat", "alerts_2wks", "intercept"]
    for c in state_cols:
        if c in sim_coefs.confounders.baseline.keys():
            for b, w in sim_coefs.confounders.baseline[c].items():
                v = confounders[b].loc[merged.fips.values].values
                baseline += w * merged[c].values * v

        if c in sim_coefs.confounders.effectiveness.keys():
            for e, w in sim_coefs.confounders.effectiveness[c].items():
                v = confounders[e].loc[merged.fips.values].values
                effectiveness += w * merged[c].values * v

    # unnormalized hosp_rate
    baseline = np.exp(np.clip(baseline, -10, 10))
    effectiveness = expit(np.clip(effectiveness, -10, 10))
    alert = merged.alert.values
    rate = baseline * (1 - alert * effectiveness)

    # simulate eligible population
    pop = confounders.total_pop.loc[merged.fips.values].values
    eligible_pop = np.random.uniform(0.001, 0.005) * pop

    # total expected rate
    mu = rate * eligible_pop

    # simulate hospitalizations
    mu[np.isnan(mu)] = 0.01  # should not happen but to avoid errors
    h = np.random.poisson(mu)

    # hosps data frame with total pop, fips, date
    hosps = merged[["fips", "date"]].copy()
    hosps["hospitalizations"] = h
    hosps["eligible_pop"] = eligible_pop

    return hosps
