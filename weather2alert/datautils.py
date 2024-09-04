import pandas as pd
from sklearn.preprocessing import StandardScaler
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
]  # ND, SD, NE and KS together only add 10 counties to the cold western group

SOUTHERN_STATES = [
    "TX",
    "OK",
    "AR",
    "LA",
    "MS",
    "AL",
    "GA",
    "FL",
    "TN",
    "KY",
    "SC",
    "NC",
    "VA",
    "WV",
    "VA",
    "MD",
    "DE",
    "NM",
    "AZ",
    "CA",  # including CA in South based on the specific counties in our final 30
]

FIPS2STATE = {
    "01": "AL",
    "02": "AK",
    "03": "AZ",
    "04": "AZ",
    "05": "AR",
    "06": "CA",
    "08": "CO",
    "09": "CT",
    "10": "DE",
    "11": "DC",
    "12": "FL",
    "13": "GA",
    "15": "HI",
    "16": "ID",
    "17": "IL",
    "18": "IN",
    "19": "IA",
    "20": "KS",
    "21": "KY",
    "22": "LA",
    "23": "ME",
    "24": "MD",
    "25": "MA",
    "26": "MI",
    "27": "MN",
    "28": "MS",
    "29": "MO",
    "30": "MT",
    "31": "NE",
    "32": "NV",
    "33": "NH",
    "34": "NJ",
    "35": "NM",
    "36": "NY",
    "37": "NC",
    "38": "ND",
    "39": "OH",
    "40": "OK",
    "41": "OR",
    "42": "PA",
    "44": "RI",
    "45": "SC",
    "46": "SD",
    "47": "TN",
    "48": "TX",
    "49": "UT",
    "50": "VT",
    "51": "VA",
    "53": "WA",
    "54": "WV",
    "55": "WI",
    "56": "WY",
    "72": "PR",
    "60": "AS",
    "66": "GU",
    "69": "MP",
    "78": "VI",
}


def get_similar_counties(fips: str, confounders: pd.DataFrame):
    """Returns a dictionary that assigns similar counties to each county based
    on climate zones
    """

    confounders = confounders.copy().set_index("fips")
    confounders["state"] = confounders.index.str[:2].map(FIPS2STATE)
    confounders["western"] = confounders["state"].isin(WESTERN_STATES)

    # now replace Cold with Cold-West or Cold-East in column ba_zone
    confounders["ba_zone"] = confounders.apply(
        lambda x: (
            "Cold-West"
            if x["western"]
            else "Cold-East" if x["ba_zone"] == "Cold" else x["ba_zone"]
        ),
        axis=1,
    )

    # get all similar
    county_zone = confounders.loc[fips].ba_zone
    similar_counties = confounders[confounders.ba_zone == county_zone].index.tolist()

    return similar_counties


def process_features(
    exogenous_states: pd.DataFrame | None = None,
    endogenous_states_actions: pd.DataFrame | None = None,
    confounders: pd.DataFrame | None = None,
    # bspline_basis: pd.DataFrame | None = None,
):
    """This function mirrors the __init__ function of the HeatAlertDataModule class used
    in the reward-training/train.py script. It is used to load the data."""

    # read all raw data
    merged = pd.merge(
        exogenous_states,
        endogenous_states_actions,
        on=["fips", "date"],
        how="inner",
    )
    confounders = confounders.copy()
    confounders["intercept"] = 1.0

    # TODO: clean data during preprocessing
    comb = pd.merge(merged, hosps, on=["fips", "date"], how="left")
    rows_with_nans = comb.isnull().any(axis=1)
    fipsdates = comb.fips + comb.date
    valid_fipsdates = fipsdates[~rows_with_nans].unique()
    valid_fips = set(comb[~rows_with_nans].fips.unique())
    # -----

    # remove bad fips from everywhere
    # merged = merged[merged.fips.isin(valid_fips)]
    merged = merged[(merged.fips + merged.date).isin(valid_fipsdates)]
    confounders = confounders[confounders.fips.isin(valid_fips)]

    # match the index os hosps with merged
    hosps = pd.merge(merged, hosps, on=["fips", "date"], how="left")
    hosps = hosps[["fips", "date", "hospitalizations", "eligible_pop"]]

    # create location indicator integer id
    fips_list = confounders.fips.values
    fips2ix = {f: i for i, f in enumerate(fips_list)}
    self.fips_list = fips_list
    sind = merged.fips.map(fips2ix).values
    year = merged.date.str.slice(0, 4).astype(int).values
    offset = hosps.eligible_pop.values
    Y = hosps.hospitalizations.values
    alert = merged.alert.values

    # spatial metadata
    self.spatial_features_names = [
        "broadband_usage",
        "log_med_hh_income",
        "democrat",
        "log_pop_density",
        "iecc_climate_zone",
        # "pm25"
        "intercept",
    ]
    W = confounders[self.spatial_features_names]

    wscaler = StandardScaler()
    wscaler_cols = self.spatial_features_names[:-1]  # don't scale the intercept
    W[wscaler_cols] = wscaler.fit_transform(W[wscaler_cols])

    self.spatial_features = torch.FloatTensor(W.values)
    self.spatial_features_idx = fips_list

    # save outcome, action and location features, metadata
    location_indicator = torch.LongTensor(sind)
    offset = torch.FloatTensor(offset)
    hospitalizations = torch.FloatTensor(Y)
    alert = torch.FloatTensor(alert)
    year = torch.LongTensor(year)
    # budget = torch.LongTensor(budget.budget.values)
    # hi_mean = torch.FloatTensor(X.HI_mean.values)  # for RL

    # compute budget as the total sum per summer-year
    merged["year"] = year
    budget = merged.groupby(["fips", "year"])["alert"].sum().reset_index()
    budget = budget.rename(columns={"alert": "budget"})
    budget = merged.merge(budget, on=["fips", "year"], how="left")
    budget = torch.LongTensor(budget.budget.values)

    # prepare covariates
    heat_qi = torch.FloatTensor(merged.heat_qi.values)
    heat_qi_above_25 = torch.FloatTensor(merged.heat_qi_above_25.values)
    heat_qi_above_75 = torch.FloatTensor(merged.heat_qi_above_75.values)
    excess_heat = torch.FloatTensor(merged.excess_heat.values)
    alert_lag1 = torch.FloatTensor(merged.alert_lag1.values)
    alerts_2wks = torch.FloatTensor(merged.alerts_2wks.values)
    weekend = torch.FloatTensor(merged.weekend.values)

    # get all cols that start with bsplines_dos in one tensor
    bsplines_dos = torch.FloatTensor(merged.filter(regex="bspline_dos", axis=1).values)
    n_basis = bsplines_dos.shape[1]

    # alert effectiveness features
    effectiveness_features = {
        "heat_qi": heat_qi,
        "excess_heat": excess_heat,
        "alert_lag1": alert_lag1,
        "alerts_2wks": alerts_2wks,
        "weekend": weekend,
        **{f"bsplines_dos_{i}": bsplines_dos[:, i] for i in range(n_basis)},
    }

    # baseline rate features
    # for now just use a simple 3-step piecewise linear function
    baseline_features = {
        "heat_qi_base": heat_qi,
        "heat_qi_above_25": heat_qi_above_25,
        "heat_qi_above_75": heat_qi_above_75,
        "excess_heat": excess_heat,
        "alert_lag1": alert_lag1,
        "alerts_2wks": alerts_2wks,
        "weekend": weekend,
        **{f"bsplines_dos_{i}": bsplines_dos[:, i] for i in range(n_basis)},
    }

    effectiveness_features = pd.DataFrame(effectiveness_features, index=merged.index)
    baseline_features = pd.DataFrame(baseline_features, index=merged.index)
    merged = pd.concat([merged, effectiveness_features, baseline_features], axis=1)

    return merged
