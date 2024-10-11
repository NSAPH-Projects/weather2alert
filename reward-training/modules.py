import numpy as np
import pandas as pd
import pyro
import pytorch_lightning as pl
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from pyro.distributions import (
    LogNormal,
    Normal,
    Poisson,
    Uniform,
    constraints,
    HalfCauchy,
)
from pyro.distributions.torch_distribution import TorchDistribution
from pyro.infer.trace_elbo import JitTrace_ELBO, Trace_ELBO
from torch.distributions.utils import broadcast_all
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class NegativeLogNormal(
    TorchDistribution
):  # helps us define constraints on certain coefficients to be only positive or negative
    arg_constraints = {"loc": constraints.real, "scale": constraints.positive}
    support = constraints.less_than(0.0)

    def __init__(self, loc, scale, validate_args=None):
        self.loc, self.scale = broadcast_all(loc, scale)
        self._base_dist = LogNormal(self.loc, self.scale)
        super().__init__(self.loc.shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(NegativeLogNormal, _instance)
        batch_shape = torch.Size(batch_shape)
        new._base_dist = self._base_dist.expand(batch_shape)
        super(NegativeLogNormal, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    @property
    def mean(self):
        return -self._base_dist.mean

    @property
    def variance(self):
        return self._base_dist.variance

    def sample(self, sample_shape=torch.Size()):
        return -self._base_dist.sample(sample_shape)

    def rsample(self, sample_shape=torch.Size()):
        return -self._base_dist.rsample(sample_shape)

    def log_prob(self, value):
        return self._base_dist.log_prob(-value)


class MLP(nn.Module):  # for learning a prior informed by the spatial variables
    """Simple MLP with the given dimensions and activation function"""

    def __init__(
        self, indim: int, outdim: int, hdim: int, num_hidden: int, act=nn.SiLU
    ):
        super().__init__()
        modules = []
        d_from = indim
        for _ in range(num_hidden):
            modules.append(nn.Sequential(nn.Linear(d_from, hdim), act()))
            d_from = hdim
        modules.append(nn.Linear(d_from, outdim))
        self.net = nn.Sequential(*modules)

    def forward(self, x):
        return self.net(x)


class HeatAlertModel(nn.Module):
    """main model definition, uses Pytorch syntax / mechanics under the hood of Pyro"""

    def __init__(
        self,
        spatial_features: torch.Tensor | None = None,
        data_size: int | None = None,
        d_baseline: int | None = None,
        d_effectiveness: int | None = None,
        baseline_constraints: dict = {},
        baseline_feature_names: list = [],
        effectiveness_constraints: dict = {},
        effectiveness_feature_names: list = [],
        hidden_dim: int = 32,
        num_hidden_layers: int = 1,
    ):
        super().__init__()

        # sample size is required to know subsampling adjustments
        self.N = data_size

        # register spatial features as buffer and size
        self.S, self.d_spatial = spatial_features.shape
        self.register_buffer("spatial_features", spatial_features)

        # save temporal feature names and dimensions
        self.d_baseline = len(baseline_feature_names)
        self.d_effectiveness = len(effectiveness_feature_names)
        self.baseline_feature_names = baseline_feature_names
        self.effectiveness_feature_names = effectiveness_feature_names

        # save constraints to define sampling
        self.baseline_constraints = baseline_constraints
        self.effectiveness_constraints = effectiveness_constraints

        # define the nn's for prior of the coefficients
        self.loc_baseline_coefs = MLP(
            self.d_spatial, d_baseline, hidden_dim, num_hidden_layers
        )
        self.loc_effectiveness_coefs = MLP(
            self.d_spatial, d_effectiveness, hidden_dim, num_hidden_layers
        )

    def forward(
        self, *inputs: torch.Tensor, condition: bool = True, return_outcomes=False
    ):
        # rebuild tensor from inputs
        hosps = inputs[0]
        loc_ind = inputs[1].long()
        offset = inputs[2]
        alert = inputs[3]
        baseline_features = inputs[4]
        eff_features = inputs[5]
        index = inputs[6]

        # register prior mean modules
        pyro.module("loc_baseline_coefs", self.loc_baseline_coefs)
        pyro.module("loc_effectiveness_coefs", self.loc_effectiveness_coefs)

        # sample coefficients with constraints to ensure correct sign
        baseline_samples = {}
        effectiveness_samples = {}

        spatial_features = self.spatial_features

        # sample coefficients
        baseline_loc = self.loc_baseline_coefs(spatial_features)
        eff_loc = self.loc_effectiveness_coefs(spatial_features)

        for i, name in enumerate(self.baseline_feature_names):
            scale = pyro.sample(f"baseline_scale_{name}", HalfCauchy(1))
            dist = self.get_dist(
                name, self.baseline_constraints, baseline_loc[:, i], scale + 1e-3
            )
            baseline_samples[name] = pyro.sample("baseline_" + name, dist)

        for i, name in enumerate(self.effectiveness_feature_names):
            scale = pyro.sample(f"effectiveness_scale_{name}", HalfCauchy(1))
            dist = self.get_dist(
                name, self.effectiveness_constraints, eff_loc[:, i], scale + 1e-3
            )
            effectiveness_samples[name] = pyro.sample("effectiveness_" + name, dist)

        # we need to match the time varying features in x with the correct coefficient sample
        baseline_contribs = []
        for i, name in enumerate(self.baseline_feature_names):
            coef = baseline_samples[name][loc_ind]
            baseline_contribs.append(coef * baseline_features[:, i])

        # compute baseline hospitalizations
        baseline_bias = pyro.sample(
            "baseline_bias", Uniform(-10, 10).expand([self.S]).to_event(1)
        )
        baseline = sum(baseline_contribs) + baseline_bias[loc_ind]

        # -----------------------------------------------------------------#
        # clamp & replace nans for stability, after training shouldnot be active
        baseline = torch.exp(baseline.clamp(-10, 10))
        baseline = torch.where(
            torch.isnan(baseline), torch.zeros_like(baseline), baseline
        )
        # =-----------------------------------------------------------------=

        effectiveness_contribs = []
        for i, name in enumerate(self.effectiveness_feature_names):
            coef = effectiveness_samples[name][loc_ind]
            effectiveness_contribs.append(coef * eff_features[:, i])

        eff_bias = pyro.sample(
            # "effectiveness_bias", Uniform(-8, -5).expand([self.S]).to_event(1)
            "effectiveness_bias",
            Uniform(-10, 10).expand([self.S]).to_event(1),
        )
        effectiveness = torch.sigmoid(sum(effectiveness_contribs) + eff_bias[loc_ind])

        # -----------------------------------------------------------------#
        # clamp & replace nans for stability, after training shouldnot be active
        effectiveness = effectiveness.clamp(1e-6, 1 - 1e-6)
        effectiveness = torch.where(
            torch.isnan(effectiveness), torch.zeros_like(effectiveness), effectiveness
        )
        # =-----------------------------------------------------------------=

        # sample the outcome
        outcome_mean = offset * baseline * (1 - alert * effectiveness)
        outcome_mean = outcome_mean.clamp(1e-3, 1e3)
        y = hosps if condition else None
        # subsample = index

        # # TODO: this is a temp bug fix, there are nans during data loading
        # nans = torch.isnan(offset)
        # y = y[~nans] if condition else None
        # outcome_mean = outcome_mean[~nans]
        # index = index[~nans]
        # effectiveness = effectiveness[~nans]
        # baseline = baseline[~nans]
        # # -----

        with pyro.plate("data", self.N, subsample=index):
            obs = pyro.sample("hospitalizations", Poisson(outcome_mean + 1e-3), obs=y)

        if not return_outcomes:
            return obs
        else:
            return torch.stack([effectiveness, baseline, outcome_mean], dim=1)

    @staticmethod
    def get_dist(name, constraints, loc, scale):
        if name not in constraints:
            return Normal(loc, 1).to_event(1)
        elif constraints[name] == "positive":
            return LogNormal(loc, scale).to_event(1)
        elif constraints[name] == "negative":
            return NegativeLogNormal(loc, scale).to_event(1)
        else:
            raise ValueError(f"unknown constraint {constraints[name]}")


class HeatAlertDataModule(pl.LightningDataModule):
    """Reads the preprocess data and prepares it for training using tensordicts"""

    def __init__(
        self,
        exogenous_states: pd.DataFrame | None = None,
        endogenous_states_actions: pd.DataFrame | None = None,
        confounders: pd.DataFrame | None = None,
        hosps: pd.DataFrame | None = None,
        bspline_basis: pd.DataFrame | None = None,
        # budget: pd.DataFrame | None = None,
        batch_size: int | None = None,
        num_workers: int = 8,
    ):
        super().__init__()
        self.workers = num_workers

        # read all raw data
        merged = pd.merge(
            exogenous_states,
            endogenous_states_actions,
            on=["fips", "date"],
            how="inner",
        )
        merged = merged.drop(columns=["significance"])
        confounders = confounders.copy()
        confounders["intercept"] = 1.0

        # TODO: clean data during preprocessing
        comb = pd.merge(merged, hosps, on=["fips", "date"], how="left")

        rows_with_nans = comb.isnull().any(axis=1)
        fipsdates = comb.fips + comb.date
        valid_fipsdates = fipsdates[~rows_with_nans].unique()
        valid_fips = set(comb[~rows_with_nans].fips.unique())
        # -----

        # nans = merged[merged.heat_qi.isnull()]
        # nans2 = hosps[hosps.hospitalizations.isnull() | hosps.eligible_pop.isnull()]
        # bad_fips = set(nans.fips.unique()) | set(nans2.fips.unique())
        # hosps_fips = set(hosps.fips.unique())
        # confounders_fips = set(confounders.fips.unique())
        # valid_fips = (valid_fips & confounders_fips)

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

        if batch_size is None:
            n = merged.shape[0]
            m = confounders.shape[0]
            self.batch_size = n // m  # ~ as batches as locations

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

        # compute budget as the total sum per summer-year
        merged["year"] = year

        budget = merged.groupby(["fips", "year"])["alert"].sum().reset_index()
        budget = budget.rename(columns={"alert": "budget"})
        budget = merged.merge(budget, on=["fips", "year"], how="left")
        budget = torch.LongTensor(budget.budget.values)

        # prepare state/action features
        keys = [k for k in merged.columns if k not in ["date", "fips", "year"]]
        tensors = {k: torch.FloatTensor(merged[k].values) for k in keys}
        # heat_qi = torch.FloatTensor(merged.heat_qi.values)
        # heat_qi_above_25 = torch.FloatTensor(merged.heat_qi_above_25.values)
        # heat_qi_above_75 = torch.FloatTensor(merged.heat_qi_above_75.values)
        # excess_heat = torch.FloatTensor(merged.excess_heat.values)
        # alert_lag1 = torch.FloatTensor(merged.alert_lag1.values)
        # alerts_2wks = torch.FloatTensor(merged.alerts_2wks.values)
        # weekend = torch.FloatTensor(merged.weekend.values)

        # get all cols that start with bspline_dos in one tensor
        # bspline_dos = torch.FloatTensor(
        #     merged.filter(regex="bspline_dos", axis=1).values
        # )
        # n_basis = bspline_dos.shape[1]

        # save dos spline basis for plots
        # # save day of summer splines as tensor
        self.bspline_basis = torch.FloatTensor(bspline_basis.values)

        # alert effectiveness features
        # effectiveness_features = {
        #     "heat_qi": heat_qi,
        #     "excess_heat": excess_heat,
        #     "alert_lag1": alert_lag1,
        #     "alerts_2wks": alerts_2wks,
        #     "weekend": weekend,
        #     **{k: tensors[k] for k in tensors if k.startswith("bspline")},
        #     **{f"bspline_dos_{i}": bspline_dos[:, i] for i in range(n_basis)},
        # }
        effectiveness_features = tensors
        self.effectiveness_feature_names = list(effectiveness_features.keys())

        # baseline rate features
        # for now just use a simple 3-step piecewise linear function
        # baseline_features = {
        #     "heat_qi_base": heat_qi,
        #     "heat_qi_above_25": heat_qi_above_25,
        #     "heat_qi_above_75": heat_qi_above_75,
        #     "excess_heat": excess_heat,
        #     "alert_lag1": alert_lag1,
        #     "alerts_2wks": alerts_2wks,
        #     "weekend": weekend,
        #     **{f"bspline_dos_{i}": bspline_dos[:, i] for i in range(n_basis)},
        # }
        baseline_features = tensors
        self.baseline_feature_names = list(baseline_features.keys())

        baseline_features_tensor = torch.stack(
            [baseline_features[k] for k in self.baseline_feature_names], dim=1
        )
        effectiveness_features_tensor = torch.stack(
            [effectiveness_features[k] for k in self.effectiveness_feature_names], dim=1
        )

        self.dataset = torch.utils.data.TensorDataset(
            hospitalizations,
            location_indicator,
            offset,
            alert,
            baseline_features_tensor,
            effectiveness_features_tensor,
            torch.arange(n),
            year,
            budget,
        )

        # get dimensions
        self.data_size = n
        self.d_baseline = len(self.baseline_feature_names)
        self.d_effectiveness = len(self.effectiveness_feature_names)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.workers,
            pin_memory=True,
            persistent_workers=True if self.workers > 0 else False,
        )


class HeatAlertLightning(pl.LightningModule):
    """Pytorch Lightning helps to efficiently train Pytorch models"""

    def __init__(
        self,
        model: nn.Module,
        guide: nn.Module,
        bspline_basis: torch.Tensor | None = None,
        jit: bool = False,
        num_particles: int = 1,
        lr: float = 1e-3,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model", "guide"])
        elbo = JitTrace_ELBO if jit else Trace_ELBO
        self.loss_fn = elbo(num_particles=num_particles).differentiable_loss
        self.lr = lr
        self.register_module("model", model)
        self.register_module("guide", guide)

        # spline basis for day of summer, used for plots
        if bspline_basis is not None:
            self.register_buffer("dos_spline_basis", bspline_basis)

    def training_step(self, batch, batch_idx):
        loss = self.loss_fn(self.model, self.guide, *batch)

        # regularization, evaluate the model
        output = self.model(*batch, return_outcomes=True)
        eff, baseline, outcome_mean = output[:, 0], output[:, 1], output[:, 2]

        # # add a penalizations to prevent overfitting, or shrink when lack of data
        # loss += 0.001 * 0.5 * (outcome_mean ** 2).mean()
        # loss += 0.001 * 0.5 * (eff ** 2).mean()

        if (
            batch_idx == 0
        ):  # do some plots at the beginning of each epoch -- see these plots using tensorboard
            with torch.no_grad():
                # one sample from the posterior
                predictive = pyro.infer.Predictive(
                    self.model,
                    guide=self.guide,
                    num_samples=1,
                    return_sites=["_RETURN"],
                )
                outputs = predictive(*batch, return_outcomes=True)["_RETURN"][0]
                eff, baseline, outcome_mean = (
                    outputs[:, 0],
                    outputs[:, 1],
                    outputs[:, 2],
                )

                # make histograms of each, comare outcome mean to real outcome
                fig, ax = plt.subplots(1, 4, figsize=(10, 3))
                ax[0].hist(eff)
                ax[0].set_title("effectiveness")
                ax[1].hist(baseline)
                ax[1].set_title("baseline")
                ax[2].hist(outcome_mean)
                ax[2].set_title("outcome mean")
                ax[3].scatter(outcome_mean, batch[0], alpha=0.1)
                ax[3].set_title("outcome mean vs real outcome")
                ax[3].set_xlabel("outcome mean")
                ax[3].set_ylabel("real outcome")
                self.logger.experiment.add_figure(
                    "distribs", fig, global_step=self.global_step
                )
                mse = ((outcome_mean - batch[0]) ** 2).mean().item()
                poisson_loss = Poisson(outcome_mean).log_prob(batch[0]).mean().item()
                self.log("mse", mse, on_epoch=True)
                self.log("poisson_loss", poisson_loss, on_epoch=True)

                # obtain quantiles
                sample = self.guide(*batch)
                keys0 = [k for k in sample.keys() if k.startswith("effectiveness_") and "_scale" not in k]
                keys1 = [k for k in sample.keys() if k.startswith("baseline_") and "_scale" not in k]
                medians_0 = np.array(
                    [torch.quantile(sample[k], 0.5).item() for k in keys0]
                )
                medians_1 = np.array(
                    [torch.quantile(sample[k], 0.5).item() for k in keys1]
                )
                q25_0 = np.array(
                    [torch.quantile(sample[k], 0.25).item() for k in keys0]
                )
                q25_1 = np.array(
                    [torch.quantile(sample[k], 0.25).item() for k in keys1]
                )
                q75_0 = np.array(
                    [torch.quantile(sample[k], 0.75).item() for k in keys0]
                )
                q75_1 = np.array(
                    [torch.quantile(sample[k], 0.75).item() for k in keys1]
                )
                l0, u0 = medians_0 - q25_0, q75_0 - medians_0
                l1, u1 = medians_1 - q25_1, q75_1 - medians_1

                # make coefficient distribution plots for coefficients, error bars are iqr
                fig, ax = plt.subplots(1, 2, figsize=(10, 4))
                ax[0].errorbar(x=keys0, y=medians_0, yerr=[l0, u0], fmt="o")
                plt.setp(ax[0].get_xticklabels(), rotation=90)
                ax[0].set_title("effectiveness coeff distribution")
                ax[0].set_ylabel("coeff value")
                ax[1].errorbar(x=keys1, y=medians_1, yerr=[l1, u1], fmt="o")
                plt.setp(ax[1].get_xticklabels(), rotation=90)
                ax[1].set_title("baseline coeff distribution")
                ax[1].set_ylabel("coeff value")
                plt.subplots_adjust(bottom=0.6)
                self.logger.experiment.add_figure(
                    "coeffs", fig, global_step=self.global_step
                )

                # now a plot of the effect of day of summer
                n_basis = self.dos_spline_basis.shape[1]
                basis = self.dos_spline_basis
                eff_coefs = [
                    sample[f"effectiveness_bspline_dos_{i}"] for i in range(n_basis)
                ]
                baseline_coefs = [
                    sample[f"baseline_bspline_dos_{i}"] for i in range(n_basis)
                ]
                eff_contribs = [
                    basis[:, i] * eff_coefs[i][:, None] for i in range(n_basis)
                ]  # list of len(n_basis) each of size (S, T)
                baseline_contribs = [
                    basis[:, i] * baseline_coefs[i][:, None] for i in range(n_basis)
                ]
                dos_beta_eff = sum(baseline_contribs)
                dos_gamma_eff = sum(eff_contribs)
                fig, ax = plt.subplots(1, 2, figsize=(8, 4))
                ax[0].plot(dos_beta_eff.T, color="k", alpha=0.05, lw=0.5)
                ax[0].plot(dos_beta_eff.mean(0), color="k", lw=2)
                ax[0].set_xlabel("Day of summer")
                ax[0].set_title("Baseline rate")
                ax[1].plot(dos_gamma_eff.T, color="k", alpha=0.05, lw=0.5)
                ax[1].plot(dos_gamma_eff.mean(0), color="k", lw=2)
                ax[1].set_xlabel("Day of summer")
                ax[1].set_title("Heat alert effectiveness")
                self.logger.experiment.add_figure(
                    "dos_effect", fig, global_step=self.global_step
                )

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
