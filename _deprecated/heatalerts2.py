import itertools
import time
import hydra
import os
import logging
import pandas as pd
import geopandas as gpd
import tempfile

from tqdm import tqdm
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options


LOGGER = logging.getLogger(__name__)


def wait_for_csv_download(directory, timeout, check_interval=5):
    """
    Waits for a .csv file to appear in the specified directory within a given timeout.

    :param directory: The directory to check for a .csv file.
    :param timeout: The maximum time to wait in seconds.
    :param check_interval: The interval between checks in seconds.
    """
    elapsed_time = 0
    while elapsed_time <= timeout:
        files = [file for file in os.listdir(directory) if file.endswith(".csv")]
        if len(files) == 1:
            print("CSV file found.")
            return files[0]
        time.sleep(check_interval)
        elapsed_time += check_interval
    raise TimeoutError("The CSV file did not appear within the specified timeout.")


def process_county_name(name, state):
    """Process county name to remove state and keywords."""
    # apply some known exceptions
    name = name.replace("St. ", "Saint ")
    name = name.replace("De ", "De")
    name = name.replace("La ", "La")
    name = name.replace("Du ", "Du")
    name = name.replace("/", " ")

    #

    if state == "AZ":
        name = name.replace("Phoenix", "Maricopa")
        name = name.replace("Sonoran Desert Natl Monument", "Maricopa")
        name = name.replace("Queen Creek", "Maricopa")
        name = name.replace("Northwest Valley", "Maricopa")
        name = name.replace("Southeast Valley", "Maricopa")
        name = name.replace("Southwest Valley", "Maricopa")
        name = name.replace("Buckeye", "Maricopa")

    # remove state
    location_keywords = ["County", "Parish", "Borough", "City and Borough", "Municipality"]
    words = name.split(" ")
    remaining = []
    for w in words:
        if not any([w.startswith(k) for k in location_keywords]):
            remaining.append(w)

    return " ".join(remaining)


def process_state_year(state, year, event, significance, cfg):
    """Download the raw csv per state/year from IOWA State University Mesonet."""
    # If targetfile exists and not overwrite skip
    id = f"state/{state}/{year}/{event}/{significance}"
    target_file = (
        f"{cfg.data_dir}/raw/alerts/{event}_{significance}_{state}_{year}.parquet"
    )
    if not cfg.alerts.overwrite and os.path.exists(target_file):
        LOGGER.info(f"File {target_file} exists, skipping.")
        return

    # Make tmp dir for download, use pytho tmp function
    tmpdir = tempfile.mkdtemp()

    # Set up Chrome options for headless mode and automatic downloads
    chrome_options = Options()
    if cfg.alerts.headless:
        chrome_options.add_argument("--headless")
    chrome_options.add_experimental_option(
        "prefs",
        {
            "download.default_directory": tmpdir,
            "download.prompt_for_download": False,
            "download.directory_upgrade": True,
            "safebrowsing.enabled": True,
        },
    )

    driver = webdriver.Chrome(options=chrome_options)

    try:
        # Navigate to the website
        url = f"https://mesonet.agron.iastate.edu/vtec/search.php#list/{id}"
        driver.get(url)
        time.sleep(2)  # ensure 2 seconds between downloads

        # Wait for the button to be clickable
        wait = WebDriverWait(driver, 10)
        cond = 'button[data-opt="csv"][data-table="3"]'
        button = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, cond)))
        button.click()

        file = wait_for_csv_download(
            tmpdir, cfg.alerts.download_wait, cfg.alerts.download_check_interval
        )

        # read csv and save to target_file as parquet
        # check first if file had any data, if empty skip and throw info to logger
        with open(f"{tmpdir}/{file}", "r") as f:
            first_line = f.readline()

        os.makedirs(f"{cfg.data_dir}/processed/alerts", exist_ok=True)

        if first_line == "\n":
            LOGGER.info(f"File {file} is empty")
            cols = [
                "phenomena",
                "significance",
                "eventid",
                "hvtec_nwsli",
                "area",
                "issue",
                "product_issue",
                "expire",
                "init_expire",
                "uri",
                "wfo",
                "fips",
                "county",
                "state",
                "year",
            ]
            df = pd.DataFrame(columns=cols)
            os.makedirs(os.path.dirname(target_file), exist_ok=True)
            df.to_parquet(target_file, index=False)

        else:
            df = pd.read_csv(f"{tmpdir}/{file}")

            # read shapefile to obtain fips
            shp_name = cfg.confounders.county_shapefile.split("/")[-1].split(".")[0]
            shp_path = f"{cfg.data_dir}/raw/shapefile/{shp_name}.shp"

            gdf = gpd.read_file(shp_path)

            state_fips = gdf.STATEFP.astype(str).str.zfill(2)
            county_fips = gdf.COUNTYFP.astype(str).str.zfill(3)
            fips = state_fips + county_fips
            gdf["Name"] = gdf["NAME"].apply(
                lambda x: process_county_name(x, state)
            )

            # read also state2fips
            f = f"{cfg.data_dir}/raw/state2fips.csv"
            state2fips = pd.read_csv(f, index_col="stusps")
            state2fips = state2fips.st.astype(str).str.zfill(2).to_dict()

            # for each row, convert locations entry to list of counties
            chunks = []

            # iterate over rows
            for _, row in df.iterrows():
                # split locations by comma
                locs = row["locations"].split(",")

                # iterate over locations
                for loc in locs:
                    # extract county name and state, the format is "county [state]"
                    state_name = loc.split("[")[1].replace("]", "")
                    state_code = state2fips[state_name]

                    # obtain county matching brackets outer content with regex
                    county_name = process_county_name(
                        loc.split("[")[0].strip(), cfg.alerts.location_keywords, state
                    )

                    # fetch counties from state and make name2fips dict
                    gdf_state = gdf[gdf.STATEFP == state_code]
                    name2fips = dict(zip(gdf_state["Name"], gdf_state["GEOID"]))

                    # get fips for each location, we need a bit of error handling
                    if county_name not in name2fips:
                        LOGGER.warning(f"County {county_name} not found.")

                        most_matches = 0
                        matched_name = None
                        for name in name2fips.keys():
                            shp_name_words = set(name.split(" "))
                            county_name_words = set(county_name.split(" "))
                            matches = len(shp_name_words & county_name_words)
                            if matches > most_matches:
                                most_matches = matches
                                matched_name = name

                        if matched_name is not None:
                            fips = name2fips[matched_name]
                            LOGGER.warning(
                                f"Using {matched_name} instead of {county_name}."
                            )
                        else:
                            LOGGER.warning(
                                f"County {county_name} not found in shapefile."
                            )
                            fips = None
                    else:
                        fips = name2fips[county_name]

                    # copy row but replace locations with fips
                    new_row = row.copy()
                    new_row["fips"] = fips
                    new_row["county"] = county_name
                    new_row["state"] = state_name
                    new_row["year"] = year

                    new_row = new_row.drop("locations")

                    chunks.append(new_row)

            # concatenate chunks
            df = pd.DataFrame(chunks)
            df.to_parquet(target_file, index=False)

    except Exception as e:
        LOGGER.error(f"An error occurred: {e}")

    finally:
        # Close the browser
        driver.quit()


@hydra.main(config_path="../../conf", config_name="data_processing", version_base=None)
def main(cfg):
    # read list of all states
    states = pd.read_csv(f"{cfg.data_dir}/raw/state2fips.csv").stusps.to_list()
    events = cfg.alerts.events
    significances = cfg.alerts.significances
    years = range(cfg.alerts.years[0], cfg.alerts.years[1] + 1)

    # states = [states[0]]  # for debug
    years = [years[-1]]  

    iterator = itertools.product(states, years, events, significances)
    n = len(states) * len(years) * len(events) * len(significances)
    for state, year, event, significance in tqdm(iterator, total=n):
        LOGGER.info(f"Processing {state}, {year}, {event}, {significance}") 
        process_state_year(state, year, event, significance, cfg)


if __name__ == "__main__":
    main()
