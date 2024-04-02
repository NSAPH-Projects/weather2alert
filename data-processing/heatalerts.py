from collections import defaultdict
import itertools
import time
import hydra
import os
import io
import logging
import pandas as pd
import datetime as dt
import tempfile

from tqdm import tqdm
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options


LOGGER = logging.getLogger(__name__)


def get_driver(cfg, dir):
    # Set up Chrome options for headless mode and automatic downloads
    chrome_options = Options()
    if cfg.alerts.headless:
        chrome_options.add_argument("--headless")
    chrome_options.add_experimental_option(
        "prefs",
        {
            "download.default_directory": dir,
            "download.prompt_for_download": False,
            "download.directory_upgrade": True,
            "safebrowsing.enabled": True,
        },
    )
    return webdriver.Chrome(options=chrome_options)


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
            return files[0]
        time.sleep(check_interval)
        elapsed_time += check_interval
    return -1  # error code


def process_request(state, year, event, significance, cfg):
    """Download the raw csv per state/year from IOWA State University Mesonet.
    https://mesonet.agron.iastate.edu/vtec/search.php"""
    # If targetfile exists and not overwrite skip
    id = f"state/{state}/{year}/{event}/{significance}"
    # target_file = (
    #     f"{cfg.data_dir}/raw/alerts/{event}_{significance}_{state}_{year}.parquet"
    # )
    # if not cfg.alerts.overwrite and os.path.exists(target_file):
    #     LOGGER.info(f"File {target_file} exists, skipping.")
    #     return

    # Make tmp dir for download, use pytho tmp function
    tmpdir = tempfile.mkdtemp()

    try:
        # Navigate to the website
        url = f"https://mesonet.agron.iastate.edu/vtec/search.php#list/{id}"

        retries = 0  # count how many times to try to download data file
        while True:
            try:
                driver = get_driver(cfg, tmpdir)
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
                if file != -1:
                    driver.quit()
                    break
            except:
                LOGGER.warning(f"Failed to download {id}. Retrying.")

            retries += 1
            if retries >= cfg.alerts.download_retries:
                raise TimeoutError("The CSV file did not appear within the specified timeout.")

        # read csv and save to target_file as parquet
        # check first if file had any data, if empty skip and throw info to logger
        with open(f"{tmpdir}/{file}", "r") as f:
            first_line = f.readline()

        os.makedirs(f"{cfg.data_dir}/processed/alerts", exist_ok=True)

        if first_line == "\n":
            return

        else:
            df = pd.read_csv(f"{tmpdir}/{file}")

            # for each row, convert locations entry to list of counties
            chunks = []

            # iterate over rows, the navigate to the url to obtain county info
            for _, row in df.iterrows():
                # go the uri
                extra_info_url = f"https://mesonet.agron.iastate.edu{row['uri']}"

                # navigate to the website
                driver = get_driver(cfg, tmpdir)
                driver.get(extra_info_url)
                # time.sleep(2)
                wait = WebDriverWait(driver, 10)

                # select 100 entries from the dropdown <select> with name='ugctable_length'
                cond = 'select[name="ugctable_length"]'
                select = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, cond)))
                select.click()
                time.sleep(0.1)
                cond = 'option[value="100"]'
                option = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, cond)))
                option.click()
                time.sleep(0.1)

                # Find the class dataTables_info and extract the number of entries
                cond = "div.dataTables_info"
                info = wait.until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, cond))
                )
                n = int(info.text.split(" ")[-2])

                num_pages = n // 100 + 1

                for page in range(num_pages):
                    # read and parse the table with id='ugctable' into pandas
                    cond = 'table[id="ugctable"]'
                    table = wait.until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, cond))
                    )
                    table_html = table.get_attribute("outerHTML")
                    info = pd.read_html(io.StringIO(table_html))[0]

                    # click on next page if not last
                    if page < num_pages - 1:
                        cond = 'a[title="Next"]'
                        button = wait.until(
                            EC.element_to_be_clickable((By.CSS_SELECTOR, cond))
                        )
                        button.click()
                        time.sleep(0.1)

                    # append row to columns of extra_info
                    for c in ["phenomena", "significance", "eventid", "uri", "wfo"]:
                        info[c] = row[c]

                    # append to chunks
                    chunks.append(info)
                    driver.quit()

            df = pd.concat(chunks)

            return df

    except Exception as e:
        LOGGER.error(f"An error occurred: {e}")


def empty_df():
    df = pd.DataFrame(
        columns=[
            "fips",
            "date",
            "issued_in_advance",
            "remaining",
            "issue",
            "expire",
            "duration",
            "phenomena",
            "significance",
            "state",
        ],
    )
    dtypes = {
        "fips": "object",
        "date": "datetime64[us]",
        "issued_in_advance": "float64",
        "remaining": "float64",
        "issue": "datetime64[us]",
        "expire": "datetime64[us]",
        "duration": "float64",
        "phenomena": "object",
        "significance": "object",
        "state": "object",
    }
    for col, dtype in dtypes.items():
        df[col] = df[col].astype(dtype)

    return df


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg):
    # read list of all states
    state = cfg.alerts.state
    events = cfg.alerts.events
    significances = cfg.alerts.significances
    years = range(cfg.alerts.years[0], cfg.alerts.years[1] + 1)

    iterator = itertools.product(years, events, significances)
    n = len(years) * len(events) * len(significances)

    chunks = []
    for year, event, significance in tqdm(iterator, total=n):
        LOGGER.info(f"Processing {state}, {year}, {event}, {significance}")
        # if year > 2006:
        #     break # <- for debugging

        res = process_request(state, year, event, significance, cfg)
        if res is not None:
            chunks.append(res)
        else:
            LOGGER.info(f"Empty download for {state}, {year}, {event}, {significance}")

    os.makedirs(f"{cfg.data_dir}/processed/alerts", exist_ok=True)
    target_file = f"{cfg.data_dir}/processed/alerts/{state}.parquet"

    if len(chunks) > 0:
        alerts = pd.concat(chunks)
        alerts.to_parquet(target_file, index=False)
    else:
        LOGGER.warning("No data was collected.")
        # save empty parquet with compatible columns
        empty_df().to_parquet(target_file, index=False)


if __name__ == "__main__":
    main()
