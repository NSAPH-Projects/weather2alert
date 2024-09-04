import logging
import os
import pandas as pd
import hydra
import pyro
import pytorch_lightning as pl
import torch
# import tensordict
from omegaconf import DictConfig, OmegaConf

from modules import (
    HeatAlertDataModule,
    HeatAlertLightning,
    HeatAlertModel,
)


LOGGER = logging.getLogger(__name__)


def load_data(dir):
    # Load data
    path = f"{dir}/processed/exogenous_states.parquet"
    exogenous_states = pd.read_parquet(path)

    path = f"{dir}/processed/endogenous_states_actions.parquet"
    endogenous_states_actions = pd.read_parquet(path)

    path = f"{dir}/processed/confounders.parquet"
    confounders = pd.read_parquet(path)

    path = f"{dir}/processed/bspline_basis.parquet"
    bspline_basis = pd.read_parquet(path)

    return exogenous_states, endogenous_states_actions, confounders, bspline_basis


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # Load data
    LOGGER.info("Loading preprocessed data")
    exo, endo, confounders, bspline_basis = load_data(cfg.data_dir)

    # Load hospitalizations, it uses real, sim, or syntethic data
    LOGGER.info("Loading hospitalizations")
    hosps = hydra.utils.instantiate(
        cfg.hospitalizations,
        confounders=confounders,
        exogenous_states=exo,
        endogenous_states_actions=endo,
    )

    # Perform a last filter to remove nans from exo, endo if hosps has nans
    LOGGER.info("Filtering nans")

    # Create data module
    LOGGER.info("Creating data module")
    dm = HeatAlertDataModule(
        confounders=confounders,
        exogenous_states=exo,
        endogenous_states_actions=endo,
        hosps=hosps,
        bspline_basis=bspline_basis,
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
    guide = pyro.infer.autoguide.AutoLowRankMultivariateNormal(model)
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

    td["fips_list"] = list(dm.fips_list)
    savedir = f"../weights/{cfg.name}"
    os.makedirs(savedir, exist_ok=True)
    torch.save(td, f"{savedir}/posterior_samples.pt")
    # preds = tensordict.TensorDict(td, batch_size=cfg.num_samples)
    # torch.save(td, f"{savedir}/posterior_samples.pt")

    # save the config in the folder for completeness
    # make sure to resolve the config with hydra/omegaconf
    with open(f"{savedir}/config.yaml", "w") as f:
        f.write(OmegaConf.to_yaml(cfg, resolve=True))


if __name__ == "__main__":
    main()
