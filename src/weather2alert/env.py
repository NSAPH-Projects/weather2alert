import os
from importlib.util import find_spec
from typing import Literal

import numpy as np
import pandas as pd
import torch
import yaml
from gymnasium import Env, spaces
from scipy.special import expit as sigmoid

from .datautils import get_similar_counties


class HeatAlertEnv(Env):
    """Class to simulate the environment for the online RL agent."""

    def __init__(
        self,
        weights: str = "nn_full_medicare",
        years: list | None = None,
        fips_list: list | None = None,
        similar_climate_counties: bool = False,
        budget: int | None = None,
    ):
        """Initialize the environment."""
        super().__init__()
        self.valid_years = years
        self.similar_climate_counties = similar_climate_counties
        self.budget = budget
        if years is None:
            years = list(range(2006, 2017))

        # load state and confounders data

        # check if path data/processed exists, then we are working with local data
        if os.path.exists("data/processed"):
            root = "./"
        else:
            root = find_spec("weather2alert").submodule_search_locations[0]

        processed_path = os.path.join(root, "data/processed")
        weights_path = os.path.join(root, "weights")

        exogenous_states = pd.read_parquet(processed_path + "/exogenous_states.parquet")
        endogenous_states_actions = pd.read_parquet(
            processed_path + "/endogenous_states_actions.parquet"
        )
        merged = pd.merge(
            exogenous_states, endogenous_states_actions, on=["fips", "date"]
        )
        merged["year"] = merged.date.str[:4].astype(int)

        # make sure merged is order by fips date and remove dates outside of the range
        # 152 days of the summer starting on May 1st to Sep 30th
        month = merged.date.str[5:7]
        merged = merged[(month >= "05") & (month <= "09")].copy()
        merged = merged.drop_duplicates(["fips", "date"])

        # merged.set_index(["fips", "date"], inplace=True)
        confounders = pd.read_parquet(processed_path + "/confounders.parquet")

        self.merged = merged.set_index(["fips", "year"])
        self.confounders = confounders

        # load posterior parameters and config
        posterior_samples = torch.load(
            f"{weights_path}/{weights}/posterior_samples.pt", weights_only=True
        )

        self.fips_list = fips_list
        if fips_list is None:
            self.fips_list = posterior_samples["fips_list"]

        self.baseline_coefs = {
            k: v for k, v in posterior_samples.items() if k.startswith("baseline")
        }
        self.effectiveness_coefs = {
            k: v for k, v in posterior_samples.items() if k.startswith("effectiveness")
        }
        with open(rf"{weights_path}/{weights}/config.yaml", "r") as f:
            self.config = yaml.safe_load(f)

        # get num posterior samples
        self.n_samples = posterior_samples["baseline_bias"].shape[0]

        # setup obs space
        obs_dim = len(merged.columns) + 2  # don't include date
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(2)  # alert or no alert

        # make additional features from merged to match names used during training
        # TODO: clean since should not be needed
        self.merged["heat_qi_base"] = self.merged["heat_qi"]
        for k in self.merged.columns:
            if k.startswith("bspline_"):
                self.merged[k.replace("bspline_", "bsplines_")] = self.merged[k]

        if self.valid_years is None:
            self.valid_years = list(self.merged.index.get_level_values("year").unique())

    def _get_episode(
        self,
        location: str,
        augment: bool = False,
        year: int | None = None,
    ):
        if augment:
            # get similar counties
            locations = get_similar_counties(location, self.confounders)
            locations = [x for x in locations if x in self.fips_list]
            self.location_index = self.rng.choice(range(len(locations)))
            self.location = locations[self.location_index]
        else:
            self.location = location
            self.location_index = self.fips_list.index(location)

        # split by year and index by dos, drop data
        if year is None:
            year = self.rng.choice(self.valid_years)

        year_data = self.merged.loc[(location, year)]
        year_data = (
            year_data.reset_index().drop(columns=["fips", "year"]).set_index("date")
        )
        return year_data, year

    def reset(
        self,
        location: str | None = None,
        similar_climate_counties: bool | None = None,
        seed: int | None = None,
        budget: int | None = None,
        sample_budget: bool = False,
        sample_budget_type: Literal["less_than", "centered"] = "less_than",
    ):
        # make rng
        if seed is None:
            seed = np.random.randint(0, 10000)
        self.rng = np.random.default_rng(seed)

        if similar_climate_counties is None:
            similar_climate_counties = self.similar_climate_counties

        # if location is None, pick a random location
        if location is None:
            location = self.rng.choice(self.fips_list)

        # get potential episode
        self.ep, year = self._get_episode(location, similar_climate_counties)
        self.ep_index = location + "_" + str(year)
        self.n_days = self.ep.shape[0]

        # sample coef index for episode
        self.coef_index = self.rng.integers(0, self.n_samples)

        self.attempted_alert_buffer = []
        self.actual_alert_buffer = []
        self.alert_streak = 0
        self.t = 0  # day of summer indicator

        if self.budget is None:
            self.budget = (
                self.ep["remaining_budget"].iloc[0] if budget is None else budget
            )

        if sample_budget:
            b = self.budget
            if sample_budget_type == "less_than":
                self.budget = self.rng.integers(0, b + 1)
            elif sample_budget_type == "centered":
                self.budget = self.rng.integers(0.5 * b, 1.5 * b + 1)
        self.remaining_budget = self.budget

        self.at_budget = False
        self.observation = self._get_obs()
        if not hasattr(self, "feat_names"):
            self.feat_names = self.observation.index.tolist()
        return self.observation.values, self._get_info()

    def _get_obs(self):
        row = self.ep.iloc[self.t].copy()

        # replace endogeous states with the actual agent behavior
        row["alert_lag1"] = self.actual_alert_buffer[-1] if self.t > 0 else 0
        row["alert_2wks"] = sum(self.actual_alert_buffer[-14:])
        row["alert_streak"] = self.alert_streak
        row["remaining_budget"] = self.budget - sum(self.actual_alert_buffer)

        return row

    def _get_reward(self, action):
        # location index
        li = self.location_index

        # current row
        row = self._get_obs()
        row["bias"] = 1.0

        # baseline function
        baseline_contribs = []
        for k, v in self.baseline_coefs.items():
            x = row[k.replace("baseline_", "")]
            v = v[self.coef_index, 0, li].item()
            baseline_contribs.append(x * v)
        baseline = np.exp(np.clip(sum(baseline_contribs), -10, 10))

        effectiveness_contribs = []
        for k, v in self.effectiveness_coefs.items():
            x = row[k.replace("effectiveness_", "")]
            v = v[self.coef_index, 0, li].item()
            effectiveness_contribs.append(x * v)
        effectiveness = sigmoid(sum(effectiveness_contribs)) * (row["heat_qi"] > 0.5)

        # reward is - normalized hospitalization rate / 10_000
        reward = float(-10_000 * baseline * (1 - effectiveness * action))

        if action == 1 and self.at_budget:
            reward = -1

        return reward

    def _get_info(self) -> dict:
        return {
            "episode_index": self.ep_index,
            "remaining_budget": self.remaining_budget,
            "at_budget": self.at_budget,
            "feature_names": self.feat_names,
            "location": self.location,
            "location_index": self.location_index,
        }

    def step(self, action: int):
        self.attempted_alert_buffer.append(action)

        # Enforcing the alert budget:
        self.at_budget = sum(self.actual_alert_buffer) == self.budget
        if action == 1 and self.at_budget:
            actual_action = 0
        else:
            actual_action = action

        self.actual_alert_buffer.append(actual_action)
        if actual_action == 1:
            self.remaining_budget -= 1

        # compute reward for the new state
        reward = self._get_reward(actual_action)

        # advance state
        done = self.t >= self.n_days - 1
        if not done:
            self.observation = self._get_obs()
            self.t += 1
            self.alert_streak = self.alert_streak + 1 if actual_action else 0

        return self.observation.values, reward, done, False, self._get_info()


if __name__ == "__main__":
    env = HeatAlertEnv()
    # obs, info = env.reset(location='06037', similar_climate_counties=True)
    obs, info = env.reset(location="06037", similar_climate_counties=False)

    # test step
    done = False
    ret = 0
    while not done:
        action = env.action_space.sample()
        obs, reward, done, _, info = env.step(action)
        ret += reward
    print("OK")
