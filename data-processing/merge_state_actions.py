from collections import defaultdict
import numpy as np
import pandas as pd
import hydra
import logging
import glob
from tqdm import tqdm
import patsy
import holidays
import datetime as dt


LOGGER = logging.getLogger(__name__)


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg):
    """This script merges data from the heatmetrics and heatalerts and computes
    the state space variables. The merged data is saved as a parquet file."""

    # Load heatmetrics
    hm = pd.read_parquet(f"{cfg.data_dir}/processed/all/heatmetrics.parquet")
    hm = hm.sort_values(["fips", "date"])

    # Load and post process heat alerts data
    alerts_files = glob.glob(f"{cfg.data_dir}/processed/alerts/*.parquet")
    alerts = []
    for f in alerts_files:
        alerts.append(pd.read_parquet(f))
    alerts = pd.concat(alerts)
    alerts = alerts.drop_duplicates().reset_index()

    # map weather zones to fips in alerts from the crosswalk provided by the NWS
    # file pre-downloaded from https://www.weather.gov/source/gis/Shapefiles/County/bp05mr24.dbx
    # see info at https://www.weather.gov/gis/ZoneCounty
    crosswalk_path = f"{cfg.data_dir}/raw/zone-county-crosswalk.dbx"
    zones = pd.read_csv(crosswalk_path, delimiter="|")
    # zones = zones[["state", "zone", "fips", "tz"]]
    zones["zone"] = zones.zone.astype(str).str.zfill(3)
    zones["fips"] = zones.fips.astype(str).str.zfill(5)
    zones["code"] = zones["state"] + zones["zone"]

    # map the UGC codes to fips and remove missing values
    alerts["code"] = alerts["UGC"].str.slice(0, 2) + alerts["UGC"].str.slice(3, 6)
    alerts = alerts.rename(columns={"Name": "name"})

    # the outer join will make as many matches as fips codes corresponding to the UGC codes
    matched_chunks = []

    for _, row in tqdm(alerts.iterrows(), total=alerts.shape[0]):
        # try to find the corresponding zone/due to year discrepancy, some zones are missing
        if row["code"] in zones["code"].values:
            zones_code = zones[zones["code"] == row["code"]]
        elif row["name"] in zones["name"].values:
            zones_code = zones[zones["name"] == row["name"]]
        else:
            # LOGGER.warning(f"Could not find zone for {row['code']}, {row['name']}")
            continue
        ncounties = zones_code.shape[0]
        row = row.drop(["UGC", "name", "Status"])
        chunk = pd.DataFrame([row] * ncounties).reset_index(drop=True)
        cols = ["fips", "county", "cwa", "tz", "state"]
        chunk[cols] = zones_code[cols].values
        matched_chunks.append(chunk)
    alerts = pd.concat(matched_chunks).reset_index(drop=True)

    # convert all time to time zones
    dtcols = ["Issue", "Issuance", "Expire", "Initial Expire"]
    for col in dtcols:
        alerts[col] = pd.to_datetime(alerts[col])
    C = 60 * 60 * 24  # convert to days
    delta = (alerts["Issue"] - alerts["Issuance"]).dt.total_seconds() / C
    alerts["issued_in_advance"] = delta
    dur = (alerts["Initial Expire"] - alerts["Issue"]).dt.total_seconds() / C
    alerts["duration"] = dur

    tz_map = defaultdict(lambda x: "US/Central")
    tz_map.update(**cfg.alerts.tz_map)
    for col in ["Issue", "Issuance", "Expire", "Initial Expire"]:
        alerts[col] = alerts.apply(lambda x: x[col].tz_convert(tz_map[x["tz"]]), axis=1)

    # now loop through all alerts and expand the date range
    expanded_alerts = defaultdict(list)
    for _, row in alerts.iterrows():
        issue = row["Issue"]
        expire = row["Expire"]
        fips = row["fips"]
        remaining = row["duration"]
        date = dt.datetime(issue.year, issue.month, issue.day)
        end = dt.datetime(expire.year, expire.month, expire.day)
        while date <= end:
            expanded_alerts["fips"].append(fips)
            expanded_alerts["date"].append(date)
            expanded_alerts["issued_in_advance"].append(row["issued_in_advance"])
            expanded_alerts["remaining"].append(remaining)
            expanded_alerts["issue"].append(issue)
            expanded_alerts["expire"].append(expire)
            expanded_alerts["duration"].append(row["duration"])
            expanded_alerts["phenomena"].append(row["phenomena"])
            expanded_alerts["significance"].append(row["significance"])
            expanded_alerts["state"].append(row["state"])
            date += dt.timedelta(days=1)
            remaining = max(0, remaining - 1)
    alerts = pd.DataFrame(expanded_alerts)

    # important to remove duplicates here
    alerts = alerts.drop_duplicates(["fips", "date"]).sort_values(["fips", "date"])

    valid_fips = set(alerts["fips"].unique())
    hm = hm[hm.fips.isin(valid_fips)]

    # merge using heat metrics as the base df
    df = hm.merge(alerts, on=["fips", "date"], how="left")
    df["alert"] = df.issue.notnull()

    # ---------
    # compute state variables/auxiliary features
    # ---------
    # percentile transform by ranking and normalizing, grouping by fips
    # make head index farenheit and divide my 100 for better scaling
    df["hi_max"] = 0.01 * (df.HImax_C * 9 / 5 + 32)
    df["heat_qi"] = df.groupby("fips")["hi_max"].rank(pct=True)

    # make spline components
    df["heat_qi_above_25"] = (df["heat_qi"] > 0.25).astype(int) * df["heat_qi"]
    df["heat_qi_above_75"] = (df["heat_qi"] > 0.75).astype(int) * df["heat_qi"]
    df["hi_max_above_25"] = (df["hi_max"] > 25).astype(int) * df["hi_max"]
    df["hi_max_above_75"] = (df["hi_max"] > 75).astype(int) * df["hi_max"]

    # add interaction variables of hi variables with heat_qi by product
    df["hi_max*heat_qi"] = df["heat_qi"] * df["hi_max"]
    df["hi_max_above_25*heat_qi"] = df["heat_qi_above_25"] * df["hi_max"]
    df["hi_max_above_75*heat_qi"] = df["heat_qi_above_75"] * df["hi_max"]

    # compute the 3-day and 7-day moving average of heat_qi, but make the rolling mean have no nans
    df["heat_qi_3d"] = df.groupby("fips")["heat_qi"].transform(
        lambda x: x.rolling(3, min_periods=1).mean()
    )
    df["heat_qi_7d"] = df.groupby("fips")["heat_qi"].transform(
        lambda x: x.rolling(7, min_periods=1).mean()
    )

    # # compute excess heat and and multiplicative interaction with heat_qi
    df["excess_heat_3d"] = (df["heat_qi"] - df["heat_qi_3d"]).clip(lower=0)
    df["excess_heat_7d"] = (df["heat_qi"] - df["heat_qi_7d"]).clip(lower=0)
    df["excess_heat_3d*heat_qi"] = df["excess_heat_3d"] * df["heat_qi"]
    df["excess_heat_7d*heat_qi"] = df["excess_heat_7d"] * df["heat_qi"]

    # compute day of summer as an integer from 0 to 152, by ranking date per year and fips
    df["year"] = df.date.dt.year
    df["dos"] = (df.groupby(["fips", "year"])["date"].rank() - 1).astype(int)

    # compute alerts in last 2 weeks, alert lag 1, and current streak
    df["alerts_2wks"] = df.groupby("fips")["alert"].transform(
        lambda x: x.rolling(14, min_periods=1).sum()
    )
    df["alert_lag1"] = df.groupby("fips")["alert"].shift(1).fillna(0).astype(int)

    # comptue the streak as the number of consecutive true before the next false
    # it should return to 0 after a false
    def streak(x: np.ndarray) -> np.ndarray:
        out = np.zeros(len(x), dtype=int)
        streak = 0
        for i, xi in enumerate(x):
            if xi:
                streak += 1
            else:
                streak = 0
            out[i] = streak
        return out

    df["alert_streak"] = df.groupby("fips")["alert"].transform(streak)

    # add issued in advance and signficance
    df["issued_in_advance"] = df["issued_in_advance"].fillna(0)

    # weekend indicator
    df["weekend"] = df.date.dt.weekday.isin([5, 6]).astype(int)

    # convert date to string
    dtmin, dtmax = df.date.min(), df.date.max()
    df["date"] = df.date.dt.strftime("%Y-%m-%d")

    # US holidays
    hdays = holidays.US(years=range(dtmin.year, dtmax.year + 1))
    hdays = set(x.strftime("%Y-%m-%d") for x in hdays.keys())
    df["holiday"] = df.date.isin(hdays).astype(int)

    # compute budget, which is the total sum per summer-year
    budget = df.groupby(["fips", "year"])["alert"].sum().reset_index()
    budget = budget.rename(columns={"alert": "budget"})
    df = df.merge(budget, on=["fips", "year"], how="left")

    # compute the rolling of alerts in the the entire summer
    df["rolling_alerts"] = df.groupby(["fips", "year"])["alert"].transform("cumsum")
    df["remaining_budget"] = df["budget"] - df["rolling_alerts"]

    # dos splines
    M = max(df.dos)
    bspline_dos = patsy.dmatrix(
        f"bs(dos, df=3, degree=3, lower_bound=0, upper_bound={M + 1}) - 1",
        {"dos": df.dos.values / M},
        return_type="dataframe",
    )
    bspline_col_means = bspline_dos.mean(axis=0)
    bspline_col_stds = bspline_dos.std(axis=0)
    bspline_dos = (bspline_dos - bspline_col_means) / bspline_col_stds

    for i in range(bspline_dos.shape[1]):
        df[f"bspline_dos_{i}"] = bspline_dos.iloc[:, i]

    # save bslines basis (e.g. for plots)
    bspline_basis = patsy.dmatrix(
        f"bs(dos, df=3, degree=3, lower_bound=0, upper_bound={M + 1}) - 1",
        {"dos": np.arange(0, M + 1) / M},
        return_type="dataframe",
    )
    # standardize and save
    bspline_basis = (bspline_basis - bspline_col_means) / bspline_col_stds
    bspline_basis.columns = [f"bspline_dos_{i}" for i in range(bspline_basis.shape[1])]
    bspline_basis.to_parquet(f"{cfg.data_dir}/processed/bspline_basis.parquet")

    # -------------------
    # save exogenous states, endogenous states, actions
    # -------------------

    # exogenous_states
    exogenous_state_vars = [
        "heat_qi",
        "heat_qi_above_25",
        "heat_qi_above_75",
        "hi_max",
        "hi_max_above_25",
        "hi_max_above_75",
        "hi_max*heat_qi",
        "hi_max_above_25*heat_qi",
        "hi_max_above_75*heat_qi",
        "heat_qi_3d",
        "excess_heat_3d",
        "excess_heat_3d*heat_qi",
        "heat_qi_7d",
        "excess_heat_7d",
        "excess_heat_7d*heat_qi",
        "weekend",
        "holiday",
        "dos",
        *[f"bspline_dos_{i}" for i in range(bspline_dos.shape[1])],
    ]
    exogenous_states = df[exogenous_state_vars + ["fips", "date"]]

    # save in 'all' and 65k split
    exogenous_states.to_parquet(
        f"{cfg.data_dir}/processed/all/exogenous_states.parquet"
    )

    confounders_65k = pd.read_parquet(
        f"{cfg.data_dir}/processed/65k/confounders.parquet"
    )
    exogenous_states[exogenous_states.fips.isin(confounders_65k.fips)].to_parquet(
        f"{cfg.data_dir}/processed/65k/exogenous_states.parquet"
    )

    # actions and endogenous states
    action_vars = [
        "alert",
        "alerts_2wks",
        "alert_lag1",
        "alert_streak",
        "remaining_budget",
        "issued_in_advance",
        "significance",
    ]
    action_states = df[["fips", "date"] + action_vars]

    # save in all and 65k
    action_states.to_parquet(
        f"{cfg.data_dir}/processed/all/endogenous_states_actions.parquet"
    )
    action_states[action_states.fips.isin(confounders_65k.fips)].to_parquet(
        f"{cfg.data_dir}/processed/65k/endogenous_states_actions.parquet"
    )

    # save budget
    budget.to_parquet(f"{cfg.data_dir}/processed/all/budget.parquet")
    budget[budget.fips.isin(confounders_65k.fips)].to_parquet(
        f"{cfg.data_dir}/processed/65k/budget.parquet"
    )


if __name__ == "__main__":
    main()
