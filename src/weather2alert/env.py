from typing import Literal

import numpy as np
import pandas as pd
import yaml
from gymnasium import Env, spaces
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from scipy.special import expit as sigmoid

from .datautils import get_similar_counties


class HeatAlertEnv(Env):
    """Class to simulate the environment for the online RL agent."""

    def __init__(
        self,
        weights: str = "nn_full_medicare_all",
        years: list | None = None,
        fips_list: list | None = None,
        similar_climate_counties: bool = False,
        budget: int | None = None,
        data_dir: str | None = None,  # passed to hf_hub_download
        split: str = "65k",
        random_starts: bool = False,
        sample_budget: bool = False,
        sample_budget_type: Literal["less_than", "centered"] = "less_than",
        min_duration: int = 65,
        min_effectiveness: float = 0.05,
        max_effectiveness: float = 0.95,
        min_heat_qi: float = 0.75,
        min_start: int = 7,
        top_k_fips: int | None = None,
        reward_type: Literal["hospitalizations", "saved"] = "saved",
    ):
        """Initialize the environment."""
        super().__init__()
        self.valid_years = years
        self.similar_climate_counties = similar_climate_counties
        self.budget = budget
        self.sample_budget_type = sample_budget_type
        self.sample_budget = sample_budget
        self.random_starts = random_starts
        self.min_start = min_start
        self.min_duration = min_duration
        self.min_effectiveness = min_effectiveness
        self.max_effectiveness = max_effectiveness
        self.min_heat_qi = min_heat_qi
        self.reward_type = reward_type

        if years is None:
            years = list(range(2006, 2017))

        # load data
        paths = {}
        for file in ["confounders", "exogenous_states", "endogenous_states_actions"]:
            paths[file] = hf_hub_download(
                repo_id="mauriciogtec/HeatAlertsRL-Data",
                repo_type="dataset",
                subfolder="data/" + split,
                filename=file + ".parquet",
                local_dir=data_dir,
            )

        merged = pd.merge(
            pd.read_parquet(paths["exogenous_states"]),
            pd.read_parquet(paths["endogenous_states_actions"]),
            on=["fips", "date"],
        )
        merged["year"] = merged.date.str[:4].astype(int)

        self.merged = merged.set_index(["fips", "year"]).drop(columns=["significance"])
        self.confounders = pd.read_parquet(paths["confounders"])

        # average fips by temperature
        if top_k_fips is not None and fips_list is None:
            gmeans = self.merged.groupby("fips")["hi_max"].mean()
            fips_list = gmeans.sort_values(ascending=False).index[:top_k_fips]

        # load posterior parameters and config
        for file in ["posterior_samples.safetensors", "config.yaml"]:
            paths[file] = hf_hub_download(
                repo_id="mauriciogtec/HeatAlertsRL-Models",
                repo_type="model",
                subfolder=weights,
                filename=file,
                local_dir=data_dir,
            )

        posterior_samples = {}
        with safe_open(paths["posterior_samples.safetensors"], framework="pt") as f:
            for k in f.keys():
                posterior_samples[k] = f.get_tensor(k)

        self.config = yaml.safe_load(open(paths["config.yaml"], "r"))
        self.fips_list = [str(x) for x in self.config["fips_list"]]
        if fips_list is not None:
            self.fips_list = [x for x in self.fips_list if x in fips_list]
            if len(self.fips_list) == 0:
                raise ValueError("No valid FIPS codes in fips_list")

        self.baseline_coefs = {
            k: v for k, v in posterior_samples.items() if k.startswith("baseline")
        }
        self.effectiveness_coefs = {
            k: v for k, v in posterior_samples.items() if k.startswith("effectiveness")
        }

        # get num posterior samples
        self.n_samples = posterior_samples["baseline_bias"].shape[0]

        # setup obs space
        self.obs_dim = len(merged.columns) - 3  # don't include date
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.obs_dim,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(2)  # alert or no alert

        # make additional features from merged to match names used during training
        # TODO: clean since should not be needed
        # self.merged["heat_qi_base"] = self.merged["heat_qi"]
        # for k in self.merged.columns:
        #     if k.startswith("bspline_"):
        #         self.merged[k.replace("bspline_", "bsplines_")] = self.merged[k]

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
            self.location_index = self.np_random.choice(range(len(locations)))
            self.location = locations[self.location_index]
        else:
            self.location = location
            self.location_index = self.fips_list.index(location)

        # split by year and index by dos, drop data
        if year is None:
            year = self.np_random.choice(self.valid_years)

        year_data = self.merged.loc[(location, year)]
        year_data = (
            year_data.reset_index().drop(columns=["fips", "year"]).set_index("date")
        )
        return year_data, year

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ):
        #
        super().reset(seed=seed)
        if options is None:
            options = {}

        location = options.get("locations", None)
        similar_climate_counties = options.get(
            "similar_climate_counties", self.similar_climate_counties
        )
        budget = options.get("budget", self.budget)
        sample_budget = options.get("sample_budget", self.sample_budget)
        sample_budget_type = options.get("sample_budget_type", self.sample_budget_type)

        # if location is None, pick a random location
        if location is None:
            location = self.np_random.choice(self.fips_list)

        # get potential episode
        self.ep, year = self._get_episode(location, similar_climate_counties)
        self.ep_index = location + "_" + str(year)
        self.n_days = self.ep.shape[0]

        # sample coef index for episode
        self.coef_index = self.np_random.integers(0, self.n_samples)

        self.attempted_alert_buffer = []
        self.actual_alert_buffer = []
        self.alert_streak = 0

        if self.random_starts:
            self.t = self.np_random.integers(
                self.min_start, self.n_days - self.min_duration - 14
            )
            self._step_counter = 0
        else:
            self.t = self.min_start

        self._step_counter = 0

        if budget is None:
            budget = self.ep["remaining_budget"].iloc[0] if budget is None else budget

        if sample_budget:
            b = budget
            if sample_budget_type == "less_than":
                budget = self.np_random.integers(0, b + 1)
            elif sample_budget_type == "centered":
                budget = self.np_random.integers(0.5 * b, 1.5 * b + 1)
        self.remaining_budget = budget

        self.at_budget = False
        self.observation = self._get_obs()
        if not hasattr(self, "feat_names"):
            self.feat_names = self.observation.index.tolist()
        return self.observation.values, self._get_info()

    def _get_obs(self):
        row = self.ep.iloc[self.t].copy().astype(np.float32)
        row = row.fillna(0)

        # replace endogeous states with the actual agent behavior
        row["alert_lag1"] = (
            self.actual_alert_buffer[-1] if self._step_counter > 0 else 0
        )
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
        baseline = sigmoid(sum(baseline_contribs))

        # effectiveness = sigmoid(5 + sum(effectiveness_contribs))
        if row["heat_qi"] > self.min_heat_qi:
            # effectiveness_contribs = []
            # for k, v in self.effectiveness_coefs.items():
            #     x = row[k.replace("effectiveness_", "")]
            #     v = v[self.coef_index, 0, li].item()
            #     effectiveness_contribs.append(x * v)
            # subtract for alert streak and last alerts
            effectiveness = (
                max(0, row["heat_qi"] - 0.8)
                - 0.05 * self.alert_streak
                - 0.01 * (sum(self.actual_alert_buffer[-7:]) - 1)
            )
            effectiveness = np.clip(
                self.min_effectiveness + effectiveness, 0, self.max_effectiveness
            )
        else:
            effectiveness = 0

        # reward is - normalized at the per 100 per day level
        if self.reward_type == "hospitalizations":
            reward = float(-1000 * baseline * (1 - effectiveness * action))
        elif self.reward_type == "saved":
            reward = float(1000 * baseline * effectiveness * action)

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
        self._needs_truncation = True

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
            self._step_counter += 1
            self.alert_streak = self.alert_streak + 1 if actual_action else 0

        # penalize if action is taken and at budget
        if action == 1 and self.at_budget:
            reward -= 1

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
