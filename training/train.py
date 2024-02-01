import logging
import os
import numpy as np

import hydra
import pyro
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf

from heat_alerts.bayesian_model import (
    HeatAlertDataModule,
    HeatAlertLightning,
    HeatAlertModel,
)

# hydra.initialize(config_path="conf/bayesian_model", version_base=None)
# cfg = hydra.compose(config_name="config")
# cfg.model.name = "FullFast_8-16"
# cfg.model.name = "FF_sample"
# cfg.training.num_particles = 1 # for full_fast
# cfg.training.batch_size = None

@hydra.main(config_path="conf/bayesian_model", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # Load data
    logging.info("Loading data")
    dm = HeatAlertDataModule(
        dir=cfg.datadir,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers,
        sampled_Y=cfg.sample_Y,
        constrain=cfg.constrain,
    )

    # Load model
    logging.info("Creating model")
    model = HeatAlertModel(
        spatial_features=dm.spatial_features,
        data_size=dm.data_size,
        d_baseline=dm.d_baseline,
        d_effectiveness=dm.d_effectiveness,
        baseline_constraints=dm.baseline_constraints,
        baseline_feature_names=dm.baseline_feature_names,
        effectiveness_constraints=dm.effectiveness_constraints,
        effectiveness_feature_names=dm.effectiveness_feature_names,
        hidden_dim=cfg.model.hidden_dim,
        num_hidden_layers=cfg.model.num_hidden_layers,
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
        dos_spline_basis=dm.dos_spline_basis,
    )

    # Train model
    logger = pl.loggers.TensorBoardLogger(
        "logs/", name=cfg.model.name
    )  # to see output, from terminal run "tensorboard --logdir logs/[cfg.model.name]"
    trainer = pl.Trainer(
        max_epochs=cfg.training.epochs,
        accelerator="auto",
        enable_checkpointing=False,
        logger=logger,
        gradient_clip_val=cfg.training.gradient_clip_val,
        max_steps=cfg.training.max_steps,
    )
    logging.info("Training model")
    trainer.fit(module, dm)

    # test saving the model using pytorch lightning
    logging.info("Saving ckpts")
    ckpt_lightning = f"ckpts/{cfg.model.name}_lightning.ckpt"
    ckpt_guide = f"ckpts/{cfg.model.name}_guide.pt"
    ckpt_model = f"ckpts/{cfg.model.name}_model.pt"

    trainer.save_checkpoint(ckpt_lightning)
    torch.save(model.state_dict(), ckpt_model)
    torch.save(guide.state_dict(), ckpt_guide)

    # save config for easy reproducibility
    with open(f"ckpts/{cfg.model.name}_cfg.yaml", "w") as f:
        OmegaConf.save(cfg, f)

    ## save average predictions for comparison to Y:
    # model.load_state_dict(torch.load(ckpt_model, map_location=torch.device("cpu")))
    # guide.load_state_dict(torch.load(ckpt_guide, map_location=torch.device("cpu")))
    sample = guide(*dm.dataset.tensors)
    sites = list(sample.keys()) + ["_RETURN"]
    predictive = pyro.infer.Predictive(
        model, guide=guide, num_samples=1000, return_sites=sites
    )
    preds = predictive(*dm.dataset.tensors, return_outcomes=True)["_RETURN"]
    Preds = torch.mean(preds, dim=0).numpy()
    os.makedirs("heat_alerts/bayesian_model/results", exist_ok=True)
    np.savetxt(f"heat_alerts/bayesian_model/results/Bayesian_{cfg.model.name}.csv", Preds, delimiter=",")


if __name__ == "__main__":
    main()
