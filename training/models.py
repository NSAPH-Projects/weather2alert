import numpy as np
import pandas as pd
import pyro
import pytorch_lightning as pl
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from pyro.distributions import LogNormal, Normal, Poisson, Uniform, constraints
from pyro.distributions.torch_distribution import TorchDistribution
from pyro.infer.trace_elbo import JitTrace_ELBO, Trace_ELBO
from torch.distributions.utils import broadcast_all


class NegativeLogNormal(TorchDistribution): # helps us define constraints on certain coefficients to be only positive or negative
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


class MLP(nn.Module): # for learning a prior informed by the spatial variables
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


class HeatAlertModel(nn.Module): # main model definition, uses Pytorch syntax / mechanics under the hood of Pyro
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
        county_summer_mean = inputs[2]
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
            dist = self.get_dist(name, self.baseline_constraints, baseline_loc[:, i])
            baseline_samples[name] = pyro.sample("baseline_" + name, dist)

        for i, name in enumerate(self.effectiveness_feature_names):
            dist = self.get_dist(name, self.effectiveness_constraints, eff_loc[:, i])
            effectiveness_samples[name] = pyro.sample("effectiveness_" + name, dist)

        # we need to match the time varying features in x with the correct coefficient sample
        baseline_contribs = []
        for i, name in enumerate(self.baseline_feature_names):
            coef = baseline_samples[name][loc_ind]
            baseline_contribs.append(coef * baseline_features[:, i])

        # compute baseline hospitalizations
        baseline_bias = pyro.sample(
            "baseline_bias", Uniform(-0.5, 0.5).expand([self.S]).to_event(1)
        )
        baseline = torch.exp(sum(baseline_contribs) + baseline_bias[loc_ind])
        baseline = baseline.clamp(max=1e6)

        effectiveness_contribs = []
        for i, name in enumerate(self.effectiveness_feature_names):
            coef = effectiveness_samples[name][loc_ind]
            effectiveness_contribs.append(coef * eff_features[:, i])

        eff_bias = pyro.sample(
            # "effectiveness_bias", Uniform(-8, -5).expand([self.S]).to_event(1)
            "effectiveness_bias", Uniform(-10, 2).expand([self.S]).to_event(1)
        )
        effectiveness = torch.sigmoid(sum(effectiveness_contribs) + eff_bias[loc_ind])
        effectiveness = effectiveness.clamp(1e-6, 1 - 1e-6)

        # sample the outcome
        outcome_mean = county_summer_mean * baseline * (1 - alert * effectiveness)

        y = hosps if condition else None
        with pyro.plate("data", self.N, subsample=index):
            obs = pyro.sample("hospitalizations", Poisson(outcome_mean + 1e-3), obs=y)

        if not return_outcomes:
            return obs
        else:
            return torch.stack([effectiveness, baseline, outcome_mean], dim=1)

    @staticmethod
    def get_dist(name, constraints, loc):
        if name not in constraints:
            return Normal(loc, 1).to_event(1)
        elif constraints[name] == "positive":
            return LogNormal(loc, 1).to_event(1)
        elif constraints[name] == "negative":
            return NegativeLogNormal(loc, 1).to_event(1)
        else:
            raise ValueError(f"unknown constraint {constraints[name]}")


class HeatAlertDataModule(pl.LightningDataModule): # used both by the bayesian model and later by the RL models
    """Reads the preprocess data and prepares it for training using tensordicts"""

    def __init__(
        self,
        dir: str,
        batch_size: int | None = None,
        num_workers: int = 8,
        load_outcome: bool = True,
        sampled_Y: bool = False,
        constrain: str = "all",
        for_gym: bool = False,
    ):
        super().__init__()
        self.dir = dir
        self.workers = num_workers
        dir = self.dir

        # read all raw data and transform into tensors
        X = pd.read_parquet(f"{dir}/states.parquet").drop(columns="intercept")
        A = pd.read_parquet(f"{dir}/actions.parquet")
        if load_outcome:
            offset = pd.read_parquet(f"{dir}/offset.parquet")
            if not sampled_Y:
                Y = pd.read_parquet(f"{dir}/outcomes.parquet")
            else:
                Y = pd.read_parquet(f"{dir}/sampled_outcomes.parquet") # when validating the model using sampled coefficients as "truth"
        else:
            Y = pd.DataFrame({"outcome": np.full(A.shape[0], np.nan)}, index=A.index)
            offset = pd.DataFrame({"offset": np.full(A.shape[0], 1)}, index=A.index)
        W = pd.read_parquet(f"{dir}/spatial_feats.parquet")
        sind = pd.read_parquet(f"{dir}/location_indicator.parquet")
        dos = pd.read_parquet(f"{dir}/Btdos.parquet")
        year = pd.read_parquet(f"{dir}/year.parquet")
        budget = pd.read_parquet(f"{dir}/budget.parquet")
        state = pd.read_parquet(f"{dir}/state.parquet")

        if batch_size is None:
            self.batch_size = X.shape[0] // W.shape[0]  # ~ as batches as locations

        # spatial metadata
        self.spatial_features = torch.FloatTensor(W.values)
        self.spatial_features_names = W.columns
        self.spatial_features_idx = W.index

        # save day of summer splines as tensor
        self.dos_spline_basis = torch.FloatTensor(dos.values)

        # save outcome, action and location features, metadata
        location_indicator = torch.FloatTensor(sind.values[:, 0])
        county_summer_mean = torch.FloatTensor(offset.values[:, 0])
        hospitalizations = torch.FloatTensor(Y.values[:, 0])
        alert = torch.FloatTensor(A.values[:, 0])
        year = torch.LongTensor(year.values[:, 0])
        budget = torch.LongTensor(budget.values[:, 0])
        hi_mean = torch.FloatTensor(X.HI_mean.values)  # for RL

        # prepare covariates
        heat_qi = torch.FloatTensor(X.quant_HI_county.values)
        heat_qi_3d = torch.FloatTensor(X.quant_HI_3d_county.values)
        excess_heat = (heat_qi - heat_qi_3d).clamp(min=0)
        alert_lag1 = torch.LongTensor(X.alert_lag1.values)
        prev_a = X.alerts_2wks.values
        self.prev_alert_mean = prev_a.mean()
        self.prev_alert_std = prev_a.std()
        previous_alerts = (prev_a - self.prev_alert_mean) / (2 * self.prev_alert_std)
        previous_alerts = torch.FloatTensor(previous_alerts)
        weekend = torch.LongTensor(X.weekend.values)
        n_dos_basis = self.dos_spline_basis.shape[1]
        dos = [torch.FloatTensor(X[f"dos_{i}"].values) for i in range(n_dos_basis)]

        # alert effectiveness features
        effectiveness_features = {
            "heat_qi": heat_qi,
            "excess_heat": excess_heat,
            "alert_lag1": alert_lag1,
            "previous_alerts": previous_alerts,
            "weekend": weekend,
            **{f"dos_{i}": v for i, v in enumerate(dos)},
        }
        self.effectiveness_feature_names = list(effectiveness_features.keys())

        # baseline rate features
        # for now just use a simple 3-step piecewise linear function
        heat_qi_base = heat_qi
        heat_qi1_above_25 = (heat_qi - 0.25) * (heat_qi > 0.25)
        heat_qi2_above_75 = (heat_qi - 0.75) * (heat_qi > 0.75)
        baseline_features = {
            "heat_qi_base": heat_qi_base,
            "heat_qi1_above_25": heat_qi1_above_25,
            "heat_qi2_above_75": heat_qi2_above_75,
            "excess_heat": excess_heat,
            "alert_lag1": alert_lag1,
            "previous_alerts": previous_alerts,
            "weekend": weekend,
            **{f"dos_{i}": v for i, v in enumerate(dos)},
        }
        self.baseline_feature_names = list(baseline_features.keys())

        ## note: constraints are passed to the heat alert model
        if constrain == "all":
            self.effectiveness_constraints = dict(
                heat_qi="positive",  # more heat more effective
                excess_heat="positive",  # more excess here more effective
                alert_lag1="negative",  # alert yesterday less effective
                previous_alerts="negative",  # more alerts less effective
            )
            self.baseline_constraints = dict(
                heat_qi1_above_25="positive",  # heat could have any slope at first
                heat_qi2_above_75="positive",  #    but should be increasingly worse
                excess_heat="positive",  # more excess heat more hospitalizations
                alert_lag1="negative",  # alert yesterday less hospitalizations
                previous_alerts="negative",  # more trailing alerts less hospitalizations
            )
        elif constrain == "none":
            self.effectiveness_constraints = dict()
            self.baseline_constraints = dict()
        elif constrain == "HI":
            self.effectiveness_constraints = dict(
                heat_qi="positive",  # more heat more effective
                excess_heat="positive",  # more excess here more effective
            )
            self.baseline_constraints = dict(
                heat_qi1_above_25="positive",  # heat could have any slope at first
                heat_qi2_above_75="positive",  #    but should be increasingly worst
                excess_heat="positive",  # more excess heat more hospitalizations
            )
        elif constrain == "alerts":
            self.effectiveness_constraints = dict(
                alert_lag1="negative",  # alert yesterday less effective
                previous_alerts="negative",  # more alerts less effective
            )
            self.baseline_constraints = dict(
                alert_lag1="negative",  # alert yesterday less hospitalizations
                previous_alerts="negative",  # more trailing alerts less hospitalizations
            )
        elif constrain == "mixed": # model used in the main analysis of the paper!
            self.effectiveness_constraints = dict(
                heat_qi="positive",  # more heat more effective
                excess_heat="positive",  # more excess here more effective
            )
            self.baseline_constraints = dict(
                alert_lag1="negative",  # alert yesterday less hospitalizations
                previous_alerts="negative",  # more trailing alerts less hospitalizations
            )

        baseline_features_tensor = torch.stack(
            [baseline_features[k] for k in self.baseline_feature_names], dim=1
        )
        effectiveness_features_tensor = torch.stack(
            [effectiveness_features[k] for k in self.effectiveness_feature_names], dim=1
        )
        
        self.dataset = torch.utils.data.TensorDataset(
                hospitalizations,
                location_indicator,
                county_summer_mean,
                alert,
                baseline_features_tensor,
                effectiveness_features_tensor,
                torch.arange(X.shape[0]),
                year,
                budget,
            )
        if for_gym:
            self.gym_dataset = [
                hospitalizations,
                location_indicator,
                county_summer_mean,
                alert,
                baseline_features_tensor,
                effectiveness_features_tensor,
                torch.arange(X.shape[0]),
                year,
                budget,
                state,
                hi_mean,  # for RL
            ]

        # get dimensions
        self.data_size = X.shape[0]
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


class HeatAlertLightning(pl.LightningModule): # Pytorch Lightning helps to efficiently train Pytorch models 
    def __init__(
        self,
        model: nn.Module,
        guide: nn.Module,
        dos_spline_basis: torch.Tensor | None = None,
        jit: bool = False,
        num_particles: int = 1,
        lr: float = 1e-3,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model", "guide"])
        self.model = model
        self.guide = guide
        elbo = JitTrace_ELBO if jit else Trace_ELBO
        self.loss_fn = elbo(num_particles=num_particles).differentiable_loss
        self.lr = lr

        # spline basis for day of summer, used for plots
        if dos_spline_basis is not None:
            self.register_buffer("dos_spline_basis", dos_spline_basis)

    def training_step(self, batch, batch_idx):
        loss = self.loss_fn(self.model, self.guide, *batch)

        if batch_idx == 0:  # do some plots at the beginning of each epoch -- see these plots using tensorboard
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

                # obtain quantiles
                sample = self.guide(*batch)
                keys0 = [k for k in sample.keys() if k.startswith("effectiveness_")]
                keys1 = [k for k in sample.keys() if k.startswith("baseline_")]
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
                fig, ax = plt.subplots(1, 2, figsize=(8, 4))
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
                eff_coefs = [sample[f"effectiveness_dos_{i}"] for i in range(n_basis)]
                baseline_coefs = [sample[f"baseline_dos_{i}"] for i in range(n_basis)]
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
