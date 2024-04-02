## Rewards model training

This folder contains the code to train the underlying rewards model for the `weather2alert` RL environment.

To train the model install the conda environment `weather2alert-training` defined in `training/requirement.yaml`:

```bash
cd reward-training
conda env create -f requirement.yaml
```

from the folder, train the model with the foll

```bash
conda activate weather2alert-training 
python train.py model=<model_config> training=<training_config>
```

See available configurations in the folder `training/conf`. The code uses the [Hydra](https://hydra.cc/) library to manage configurations.

You can monitor the training process with tensorboard:

```bash
tensorboard --logdir=logs
```