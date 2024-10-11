import os
import numpy as np
import pandas as pd
import geopandas as gpd
import wget
import zipfile
from census import Census
import hydra
import logging

LOGGER = logging.getLogger(__name__)


def download_and_uncompress(url, tgt_dir):
    """Download and uncompress a zip file."""
    tgt_file = os.path.join(tgt_dir, os.path.basename(url))
    if not os.path.exists(tgt_file):
        os.makedirs(tgt_dir, exist_ok=True)
        wget.download(url, tgt_file)
        with zipfile.ZipFile(tgt_file, "r") as zip_ref:
            zip_ref.extractall(tgt_dir)
        os.remove(tgt_file)


def process_shapefile(tgt_dir, url):
    """Process shapefile to prepare land data."""

    # get shp name, e.g. cb_2013_us_county_500k.zip" -> "cb_2013_us_county_500k"
    shp_name = url.split("/")[-1].split(".")[0] + ".shp"

    tgt_shpfile = os.path.join(tgt_dir, shp_name)

    # Download and uncompress the shapefile if it doesn't exist
    if not os.path.exists(tgt_shpfile):
        download_and_uncompress(url, tgt_dir)

    # Load the shapefile into a GeoDataFrame
    counties = gpd.read_file(tgt_shpfile)
    counties["lon"] = counties.to_crs("epsg:4326").centroid.x
    counties["lat"] = counties.to_crs("epsg:4326").centroid.y
    counties.rename(columns={"ALAND": "area", "GEOID": "fips"}, inplace=True)
    counties["area"] = counties["area"] * 3.86102e-7  # Convert to square miles
    counties = counties[["fips", "area", "lon", "lat"]]

    return counties


def fetch_census_data(api_key):
    """Fetch population and income data from the Census API."""
    c = Census(api_key, year=2013)

    # Fetch data for all counties across the US
    pop_data = c.acs5.state_county("B01001_001E", Census.ALL, Census.ALL)
    income_data = c.acs5.state_county("B19013_001E", Census.ALL, Census.ALL)

    pop_df = (
        pd.DataFrame(pop_data)
        .assign(fips=lambda df: df.state + df.county)
        .rename(columns={"B01001_001E": "total_pop"})
        .loc[:, ["fips", "total_pop"]]
    )
    income_df = (
        pd.DataFrame(income_data)
        .assign(fips=lambda df: df.state + df.county)
        .rename(columns={"B19013_001E": "med_hh_income"})
        .loc[:, ["fips", "med_hh_income"]]
        .assign(log_med_hh_income=lambda df: np.log(df.med_hh_income))
    )

    # merge
    census_df = pop_df.merge(income_df, on="fips", how="left")

    return census_df


def download_and_process_broadband(tgt_file):
    """Download and process broadband data."""
    broadband_url = (
        "https://raw.githubusercontent.com/microsoft/"
        "USBroadbandUsagePercentages/master/dataset/broadband_data_2020October.csv"
    )
    if not os.path.exists(tgt_file):
        wget.download(broadband_url, tgt_file)

    # Load the data into a DataFrame
    broadband_df = pd.read_csv(tgt_file)

    # Clean up the column names by stripping whitespace and converting to uppercase
    broadband_df = broadband_df
    broadband_df.rename(str.strip, axis="columns", inplace=True)
    broadband_df.rename(columns={"BROADBAND USAGE": "BROADBAND_USAGE"}, inplace=True)

    # convert BROADBAND_USAGE to numeric, but fill with nans
    broadband_df["BROADBAND_USAGE"] = pd.to_numeric(
        broadband_df["BROADBAND_USAGE"], errors="coerce"
    )

    # fill the nans with the mean
    m = np.nanmean(broadband_df["BROADBAND_USAGE"])
    broadband_df["BROADBAND_USAGE"] = broadband_df["BROADBAND_USAGE"].fillna(m)

    # Clean index
    broadband_df["fips"] = broadband_df["COUNTY ID"].astype(str).str.zfill(5)

    return broadband_df[["fips", "BROADBAND_USAGE"]]


def download_and_prepare_climate_zones(tgt_file):
    """Download and prepare climate zones data."""
    climate_zones_url = (
        "https://gist.githubusercontent.com/philngo/d3e251040569dba67942/"
        "raw/0c98f906f452b9c80d42aec3c8c3e1aafab9add8/climate_zones.csv"
    )
    if not os.path.exists(tgt_file):
        wget.download(climate_zones_url, tgt_file)
    zones = pd.read_csv(tgt_file)
    zones["State FIPS"] = zones["State FIPS"].astype(str).str.zfill(2)
    zones["County FIPS"] = zones["County FIPS"].astype(str).str.zfill(3)
    zones["fips"] = zones["State FIPS"] + zones["County FIPS"]
    return zones[["fips", "IECC Climate Zone", "BA Climate Zone"]].rename(
        columns={"IECC.Climate Zone": "IECC_zone", "BA Climate Zone": "BA_zone"}
    )


def process_election_data(file_path):
    # Read the election data
    data = pd.read_csv(file_path)

    # Filter for years of interest and ensure county_fips is not NaN
    years_of_interest = [2004, 2008, 2012, 2016]
    filtered_data = data.loc[
        data["year"].isin(years_of_interest) & data["county_fips"].notna()
    ]

    # Calculate vote rates
    filtered_data["Rates"] = (
        filtered_data["candidatevotes"] / filtered_data["totalvotes"]
    )

    # Filter for Democrat and Republican positions
    dem_data = filtered_data.loc[(filtered_data["party"] == "DEMOCRAT")]
    rep_data = filtered_data.loc[(filtered_data["party"] == "REPUBLICAN")]

    # Prepare the Democrat and Republican DataFrames
    dem_df = dem_data[["year", "county_fips", "Rates"]].rename(
        columns={"Rates": "democrat"}
    )
    rep_df = rep_data[["year", "county_fips", "Rates"]].rename(
        columns={"Rates": "republican"}
    )
    election_df = pd.merge(dem_df, rep_df, on=["year", "county_fips"], how="outer")

    # fill missing values with total nanmean
    demmean = np.nanmean(election_df["democrat"])
    repmean = np.nanmean(election_df["republican"])
    election_df["democrat"] = election_df["democrat"].fillna(demmean)
    election_df["republican"] = election_df["republican"].fillna(repmean)

    # county for compat
    election_df["fips"] = (
        election_df["county_fips"].astype(int).astype(str).str.zfill(5)
    )

    # Save the cleaned and merged DataFrame
    election_df = election_df[["fips", "year", "democrat", "republican"]]

    # Take the average per year, TODO: allow yearly
    election_df = election_df.groupby("fips").mean().reset_index()
    election_df = election_df.drop(columns=["year"])

    return election_df


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg):
    tgt_dir_shapefile = f"{cfg.data_dir}/raw/shapefile"
    counties = process_shapefile(tgt_dir_shapefile, cfg.confounders.county_shapefile)

    census_df = fetch_census_data(cfg.census_api_key)

    broadband_file = f"{cfg.data_dir}/raw/broadband_data.csv"
    broadband_df = download_and_process_broadband(broadband_file)

    climate_zones_file = f"{cfg.data_dir}/raw/DoE_climate_zones.csv"
    climate_zones_df = download_and_prepare_climate_zones(climate_zones_file)

    # process the election data, originally downloaded from
    #  https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/VOQCHQ&version=12.0
    #  (license public domain CC0 1.0)
    election_data_file = f"{cfg.data_dir}/raw/countypres_2000-2020.csv"
    election_df = process_election_data(election_data_file)

    # Merging all datasets
    merged_df = (
        counties.merge(census_df, on="fips", how="left")
        .merge(broadband_df, on="fips", how="left")
        .merge(climate_zones_df, on="fips", how="left")
        .merge(election_df, on="fips", how="left")
        .assign(pop_density=lambda df: df.total_pop / df.area)
        .assign(log_pop_density=lambda df: np.log(df.pop_density))
    )

    # TODO: allow for different confounders per year

    # Save the final combined DataFrame
    # normalize all names to lower case and replace spaces with underscores
    merged_df.columns = [col.lower().replace(" ", "_") for col in merged_df.columns]

    # Keep only those places with population > 65000 and complete data cases
    print(merged_df.shape)

    # Save two versions, one with all counties, and one with filter above 65k
    merged_df = merged_df.dropna()

    os.makedirs(f"{cfg.data_dir}/processed/65k", exist_ok=True)
    os.makedirs(f"{cfg.data_dir}/processed/all", exist_ok=True)

    merged_df.to_parquet(
        f"{cfg.data_dir}/processed/all/confounders.parquet", index=False
    )
    merged_df.loc[merged_df.total_pop > 65000].to_parquet(
        f"{cfg.data_dir}/processed/65k/confounders.parquet", index=False
    )


if __name__ == "__main__":
    main()
