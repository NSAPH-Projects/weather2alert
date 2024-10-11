import os
import pandas as pd
import wget
import hydra
import logging


LOGGER = logging.getLogger(__name__)


def transform_rds_to_parquet(rds_path, parquet_path):
    """Transform downloaded RDS file to parquet format.
    Note that rpy2 and pyreadr both fail.
    Rpy2 fails in conda environment, and pyreadr detecs invalid features."""

    # Convert RDS to feather via R cmd line
    cmd = f'Rscript -e \'library(arrow); arrow::write_parquet(readRDS("{rds_path}"), "{parquet_path}")\''
    os.system(cmd)

    # verify that the file was created
    assert os.path.exists(parquet_path), f"Failed to create {parquet_path}.parquet"


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg):
    # Download data if not already present
    download_path = (
        f"{cfg.data_dir}/raw/heatmetrics.rds"  # data is in R's native format
    )
    url = cfg.heatmetrics.url

    raw_parquet_path = download_path.replace(".rds", ".parquet")
    if not os.path.exists(raw_parquet_path):
        if not os.path.exists(download_path):
            wget.download(url, download_path)
        transform_rds_to_parquet(download_path, raw_parquet_path)
        os.remove(download_path)
    else:
        LOGGER.info(f"Skipping download and transform, file exists.")

    # Read data frame stored as R native format
    df = pd.read_parquet(download_path.replace(".rds", ".parquet"))
    df["Date"] = pd.to_datetime(df["Date"])

    # Filter
    min_month, max_month = cfg.heatmetrics.min_month, cfg.heatmetrics.max_month
    min_year, max_year = cfg.heatmetrics.min_year, cfg.heatmetrics.max_year
    df = df[df["Date"].dt.month.between(min_month, max_month)]
    df = df[df["Date"].dt.year.between(min_year, max_year)]
    df = df[["StCoFIPS", "Date"] + cfg.heatmetrics.cols]
    df.rename(columns={"StCoFIPS": "fips", "Date": "date"}, inplace=True)

    # Sort by fips and date
    df = df.sort_values(["fips", "date"])

    # Save the version for all fips
    confounders_all = pd.read_parquet(
        f"{cfg.data_dir}/processed/all/confounders.parquet"
    )
    processed_path = f"{cfg.data_dir}/processed/all/heatmetrics.parquet"
    df_all = df[df.fips.isin(confounders_all.fips)]
    df_all.to_parquet(processed_path)
    LOGGER.info(f"Data written to {processed_path} with head\n: {df_all.head()}")

    # Save the version for fips in 65k split
    confounders_65k = pd.read_parquet(
        f"{cfg.data_dir}/processed/65k/confounders.parquet"
    )
    processed_path = f"{cfg.data_dir}/processed/65k/heatmetrics.parquet"
    df_65k = df[df.fips.isin(confounders_65k.fips)]
    df_65k.to_parquet(processed_path)
    LOGGER.info(f"Data written to {processed_path} with head\n: {df_65k.head()}")


if __name__ == "__main__":
    main()
