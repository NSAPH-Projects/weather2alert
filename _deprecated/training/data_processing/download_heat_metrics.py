import os
import requests
import pandas as pd
import pyreadr

def download_file(url, target_path):
    """Download a file from a URL to a given target path."""
    response = requests.get(url)
    response.raise_for_status()  # Raise an error for bad status codes
    with open(target_path, 'wb') as file:
        file.write(response.content)

def filter_and_transform_data(df):
    """Filter and transform the dataframe according to specified criteria."""
    # Convert 'Date' to datetime and filter by month and year
    df['Date'] = pd.to_datetime(df['Date'])
    df = df[df['Date'].dt.month.between(5, 9)]
    df = df[df['Date'].dt.year.between(2006, 2019)]
    # Select specific columns
    df = df[['StCoFIPS', 'Date', 'HImin_C', 'HImax_C', 'HImean_C']]
    return df

def main():
    # Ensure data directory exists
    data_dir = "./data/heatmetrics"
    os.makedirs(data_dir, exist_ok=True)

    # Download data if not already present
    url = "https://ndownloader.figstatic.com/files/35070550"
    target_file = f"{data_dir}/heatmetrics.rds"
    if not os.path.exists(target_file):
        download_file(url, target_file)
    
    # Read RDS file
    result = pyreadr.read_r(target_file)
    # pyreadr.read_r returns a dictionary where keys are the variable names from R,
    # and the values are pandas dataframes. Assuming the RDS file contains one dataframe:
    df = result[None]  # use None for unnamed objects
    
    # Filter and transform data
    df = filter_and_transform_data(df)
    
    # Write to parquet
    processed_path = "data/processed/heatmetrics.parquet"
    df.to_parquet(processed_path)

if __name__ == "__main__":
    main()
