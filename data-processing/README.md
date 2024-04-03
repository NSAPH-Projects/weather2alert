## Data preprocessing

This pipeline is fully automated and takes care of:
- Creating the dataset of location confounders
- Downloading and processing the heat alerts data from the Iowa Environmental Mesonet API
- Downloading heat quantile metrics
- Processing and merging the data

We use the [snakemake](https://snakemake.readthedocs.io/en/stable/) workflow management system to automate the pipeline. The pipeline is defined in the `Snakefile` file. The pipeline is executed by running the command `snakemake` in the root directory of the repository.

```bash
pip install snakemake
snakemake --use-conda --core 8
```

Running the pipelines will results in populating the `data/process` directory.