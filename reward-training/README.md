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
python train.py model=<model_cfg> training=<training_cfg> hospitalizations=<hosps_cfg>
```
See the defaults and available configurations in the folder `training/conf`. The code uses the [Hydra](https://hydra.cc/) library to manage configurations.

You can monitor the training process with tensorboard:

```bash
tensorboard --logdir=logs
```

Upon completion, the model will be saved in the folder `weights` in the root folder. Note that the `weather2alert` packages comes with pre-trained weights obtained with these configurations.