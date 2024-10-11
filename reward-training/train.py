import logging
import os

import hydra
import pandas as pd
import pyro
import pytorch_lightning as pl
import torch
import yaml
from modules import HeatAlertDataModule, HeatAlertLightning, HeatAlertModel
from omegaconf import DictConfig, OmegaConf
from safetensors.torch import save_file

LOGGER = logging.getLogger(__name__)


def load_data(dir, conf="65k"):
    D = {}  # data dictionary

    # Load data
    path = f"{dir}/processed/{conf}/exogenous_states.parquet"
    D["exogenous_states"] = pd.read_parquet(path)

    path = f"{dir}/processed/{conf}/endogenous_states_actions.parquet"
    D["endogenous_states_actions"] = pd.read_parquet(path)

    path = f"{dir}/processed/{conf}/confounders.parquet"
    D["confounders"] = pd.read_parquet(path)

    path = f"{dir}/processed/bspline_basis.parquet"
    D["bspline_basis"] = pd.read_parquet(path)

    path = f"{dir}/processed/{conf}/budget.parquet"
    D["budget"] = pd.read_parquet(path)

    return D


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # Load data
    LOGGER.info("Loading preprocessed data")
    data_dict = load_data(cfg.data_dir)

    # Load hospitalizations, it uses real, sim, or syntethic data
    LOGGER.info("Loading hospitalizations")
    hosps = hydra.utils.instantiate(
        cfg.hospitalizations,
        confounders=data_dict["confounders"],
        exogenous_states=data_dict["exogenous_states"],
        endogenous_states_actions=data_dict["endogenous_states_actions"],
    )

    # Perform a last filter to remove nans from exo, endo if hosps has nans
    LOGGER.info("Filtering nans")

    # Create data module
    LOGGER.info("Creating data module")
    dm = HeatAlertDataModule(
        confounders=data_dict["confounders"],
        exogenous_states=data_dict["exogenous_states"],
        endogenous_states_actions=data_dict["endogenous_states_actions"],
        hosps=hosps,
        bspline_basis=data_dict["bspline_basis"],
        # budget=data_dict["budget"],
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers,
    )

    # Load model
    LOGGER.info("Loading model")
    model = HeatAlertModel(
        spatial_features=dm.spatial_features,
        data_size=dm.data_size,
        d_baseline=dm.d_baseline,
        d_effectiveness=dm.d_effectiveness,
        baseline_constraints=cfg.constraints.baseline,
        baseline_feature_names=dm.baseline_feature_names,
        effectiveness_constraints=cfg.constraints.effectiveness,
        effectiveness_feature_names=dm.effectiveness_feature_names,
        hidden_dim=cfg.arch.hidden_dim,
        num_hidden_layers=cfg.arch.num_hidden_layers,
    )

    # use low-rank normal guide and initialize by calling it once
    guide = pyro.infer.autoguide.AutoLowRankMultivariateNormal(model, rank=20)
    guide(*dm.dataset.tensors)  # always needed to initialize guide

    # create lightning module for training
    module = HeatAlertLightning(
        model=model,
        guide=guide,
        num_particles=cfg.training.num_particles,
        lr=cfg.training.lr,
        jit=cfg.training.jit,
        bspline_basis=dm.bspline_basis,
    )

    # Train model
    logger = pl.loggers.TensorBoardLogger(
        "logs/", name=cfg.name
    )  # to see output, from terminal run "tensorboard --logdir logs/[cfg.arch.name]"
    trainer = pl.Trainer(
        max_epochs=cfg.training.epochs,
        accelerator=cfg.training.accelerator,
        enable_checkpointing=False,
        logger=logger,
        gradient_clip_val=cfg.training.gradient_clip_val,
        max_steps=cfg.training.max_steps,
    )
    logging.info("Training model")
    trainer.fit(module, dm)

    # save posterior coefficients by using the predictive
    predictive = pyro.infer.Predictive(model, guide=guide, num_samples=cfg.num_samples)
    preds = predictive(*dm.dataset.tensors)

    # save posterior samples
    td = dict()
    for b in dm.baseline_feature_names + ["bias"]:
        td[f"baseline_{b}"] = preds[f"baseline_{b}"]

    for e in dm.effectiveness_feature_names + ["bias"]:
        td[f"effectiveness_{e}"] = preds[f"effectiveness_{e}"]

    savedir = f"../weights/{cfg.name}"
    os.makedirs(savedir, exist_ok=True)
    save_file(td, f"{savedir}/posterior_samples.safetensors")

    # save the config in the folder for completeness
    # make sure to resolve the config with hydra/omegaconf
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    cfg_dict["fips_list"] = dm.fips_list.tolist()
    with open(f"{savedir}/config.yaml", "w") as f:
        yaml.dump(cfg_dict, f)


if __name__ == "__main__":
    main()
