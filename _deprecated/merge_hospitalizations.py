import pandas as pd
import hydra
import logging
import numpy as np
from scipy.special import expit


LOGGER = logging.getLogger(__name__)


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg):
    """Simulates or loads hospitalizations"""
    # Load confounders
    confounders = pd.read_parquet(f"{cfg.data_dir}/processed/confounders.parquet")
    confounders = confounders.set_index("fips", drop=True)

    # Load heatmetrics and alertsc
    exo = pd.read_parquet(f"{cfg.data_dir}/processed/exogenous_states.parquet")
    endo = pd.read_parquet(f"{cfg.data_dir}/processed/endogenous_states_actions.parquet")
    merged = pd.merge(exo, endo, on=["fips", "date"], how="inner")

    # split cases to obtain hostpitalizations
    if cfg.hospitalizations.data_path is not None:
        LOGGER.info("Loading hospitalizations from data_path")
        hosps = pd.read_parquet(cfg.hospitalizations.data_path)
        hosps = hosps[["fips", "date", "other_hosps", "total_count"]]
        # rename for clarity
        hosps = hosps.rename(
            columns={"other_hosps": "hospitalizations", "total_count": "eligible_pop"}
        )
        hosps = pd.merge(merged, hosps, on=["fips", "date"], how="inner")

    elif cfg.hospitalizations.synthetic_path is not None:
        LOGGER.info("Loading synthetic hospitalizations from pretrained weights")
        # TODO
        pass
    else:
        LOGGER.info("Simulating hospitalizations")
        baseline = np.zeros(merged.shape[0])
        effectiveness = np.zeros(merged.shape[0])
        merged["intercept"] = 1

        # equal to all locations
        for b, w in cfg.hospitalizations.sim_coefs.features.baseline.items():
            baseline += merged[b] * w

        for e, w in cfg.hospitalizations.sim_coefs.features.effectiveness.items():
            effectiveness += merged[e] * w

        # now location specific
        state_cols = ["heat_qi", "excess_heat", "alerts_2wks", "intercept"]
        for c in state_cols:
            if c in cfg.hospitalizations.sim_coefs.confounders.baseline.keys():
                for b, w in cfg.hospitalizations.sim_coefs.confounders.baseline[
                    c
                ].items():
                    v = confounders[b].loc[merged.fips.values].values
                    baseline += w * merged[c].values * v

            if c in cfg.hospitalizations.sim_coefs.confounders.effectiveness.keys():
                for e, w in cfg.hospitalizations.sim_coefs.confounders.effectiveness[
                    c
                ].items():
                    v = confounders[e].loc[merged.fips.values].values
                    effectiveness += w * merged[c].values * v

        # unnormalized hosp_rate
        baseline = np.exp(baseline)
        effectiveness = expit(effectiveness)
        alert = merged.alert.values
        rate = baseline * (1 - alert * effectiveness)

        # simulate eligible population
        pop = confounders.total_pop.loc[merged.fips.values].values
        eligible_pop = np.random.uniform(0.01, 0.02) * pop

        # total expected rate
        mu = rate * eligible_pop

        # simulate hospitalizations
        h = np.random.poisson(mu)

        # hosps data frame with total pop, fips, date
        hosps = merged[["fips", "date"]].copy()
        hosps["hospitalizations"] = h
        hosps["eligible_pop"] = eligible_pop

    # save
    target_file = f"{cfg.data_dir}/processed/training_data.parquet"
    hosps.to_parquet(target_file, index=False)


if __name__ == "__main__":
    main()
