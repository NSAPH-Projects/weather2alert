
from argparse import ArgumentParser
import numpy as np
from matplotlib import pyplot as plt
import json
import csv
import pandas as pd
import re

import pyro
from pyro.distributions import Poisson
import pytorch_lightning as pl
import torch

# from heat_alerts.bayesian_model.pyro_heat_alert import (HeatAlertDataModule, HeatAlertLightning,
#                              HeatAlertModel)
from pyro_heat_alert import (HeatAlertDataModule, HeatAlertLightning,
                             HeatAlertModel)


def main(params):
    # params = {"type": "initial", "model_name": "FF_C-M_wide-EB-prior", "n_samples": 1, "SC": "F", "county": 36005, "constrain": "mixed"}
    # params = {"type": "validation", "model_name": "FF_C-M_sample", "n_samples": 1000, "SC": "F", "county": 36005, "constrain": "mixed"}
    params=vars(params)
    ## Read in data:
    n_days = 153
    years = set(range(2006, 2017))
    n_years = len(years)
    dm = HeatAlertDataModule(
            dir="data/processed", # dir="data/processed",
            batch_size=n_days*n_years,
            num_workers=4,
            for_gym=True,
            constrain=params["constrain"]
        )
    data = dm.gym_dataset
    hosps = data[0]
    loc_ind = data[1].long()
    county_summer_mean = data[2]
    alert = data[3]
    baseline_features = data[4]
    eff_features = data[5]
    index = data[6]

    ## Set up the rewards model, previously trained using pyro:
    model = HeatAlertModel(
            spatial_features=dm.spatial_features,
            data_size=dm.data_size,
            d_baseline=dm.d_baseline,
            d_effectiveness=dm.d_effectiveness,
            baseline_constraints=dm.baseline_constraints,
            baseline_feature_names=dm.baseline_feature_names,
            effectiveness_constraints=dm.effectiveness_constraints,
            effectiveness_feature_names=dm.effectiveness_feature_names,
            hidden_dim= 32, #cfg.model.hidden_dim,
            num_hidden_layers= 1, #cfg.model.num_hidden_layers,
        )
    model.load_state_dict(torch.load("ckpts/" + params["model_name"] + "_model.pt"))
    guide = pyro.infer.autoguide.AutoLowRankMultivariateNormal(model)
    guide(*dm.dataset.tensors)
    guide.load_state_dict(torch.load("ckpts/" + params["model_name"] + "_guide.pt"))

    inputs = [
        hosps, 
        loc_ind, 
        county_summer_mean, 
        alert,
        baseline_features, 
        eff_features, 
        index
    ]

    if params["type"] == "initial": # Saving one sample to test model identification:
      sample = guide(*inputs)
  
      w = csv.writer(open("data/processed/Coef_sample.csv", "w"))
      for key, val in sample.items():
          w.writerow([key, val])

      ## Calculate fake Y based on the sampled coefficients:
      baseline_contribs = []
      for i, name in enumerate(dm.baseline_feature_names):
          coef = sample["baseline_" + name][loc_ind]
          baseline_contribs.append(coef * baseline_features[:, i])
  
      baseline_bias = sample['baseline_bias']
      baseline = torch.exp(sum(baseline_contribs) + baseline_bias[loc_ind])
      baseline = baseline.clamp(max=1e6)
  
      effectiveness_contribs = []
      for i, name in enumerate(dm.effectiveness_feature_names):
          coef = sample["effectiveness_" + name][loc_ind]
          effectiveness_contribs.append(coef * eff_features[:, i])
  
      eff_bias = sample['effectiveness_bias']
      effectiveness = torch.sigmoid(sum(effectiveness_contribs) + eff_bias[loc_ind])
      effectiveness = effectiveness.clamp(1e-6, 1 - 1e-6)
  
      outcome_mean = county_summer_mean * baseline * (1 - alert * effectiveness)
      with pyro.plate("data", len(hosps), subsample=index):
          obs = pyro.sample("hospitalizations", Poisson(outcome_mean + 1e-3), obs=hosps)

      ## Save for training the bayesian model with the 
      df = pd.DataFrame(obs.numpy())
      df.to_parquet("data/processed/sampled_outcomes.parquet")
      
    else: # Saving posterior samples from the validation model
      with torch.no_grad():
          samples = [guide(*inputs) for _ in range(params["n_samples"])]
  
      w = csv.writer(open("data/processed/Validation_coefs.csv", "w"))
      for i in range(params["n_samples"]):
          for key, val in samples[i].items():
              w.writerow([key, val.numpy().tolist()]) 
      


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--type", type=str, default="initial", help="initial samples or validation model samples?")
    parser.add_argument("--model_name", type=str, default="Full_8-7", help="model name")
    parser.add_argument("--n_samples", type=int, default=1000, help="number of samples to take")
    parser.add_argument("--SC", type=str, default="F", help="Make plot for single county?")
    parser.add_argument("--county", type=int, default=36005, help="county to make plots for")
    parser.add_argument("--constrain", type=str, default="all", help="constrained vars?")
    args = parser.parse_args()
    main(args)
