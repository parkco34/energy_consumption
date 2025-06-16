#!/usr/bin/env python
"""
Oracle Project
"""
import time
start = time.time()
import os
#from data_utils.data_cleaning import DataCleaning
from nasa_power_api import NASAPowerAPI as api # Import NASA API Class
import pandas as pd

# List of common date-related keywords for columns
DATE_KEYWORDS = ["month", "year", "date", "day", "time"]

def read_energy_data(filename):
    """
    Reads dataset (csv file)
    ----------------------------------------
    INPUT:
        filename: (str)

    OUTPUT:
        df:(pd.DataFrame)
    """
    # Ensure filename exists in current directory
    if os.path.exists(filename):
        return pd.read_csv(filename)

    else:
        print(f"{filename} Doesn't exist in given path")
        return None

def datetime_conversion(dataframe, sort_by_date=True):
    """
    Determines the date-like columns and if it's and int or a str,
    appropriately convert to datetime.
    --------------------------------------------------
    INPUT:
        dataframe: (pd.DataFrame) Original dataframe
        sort_by_date: (bool; default=True) Whether to sort the dataframe by
        datre or leave it as it is.

    OUTPUT:
        new_df: (pd.DataFrame) Datetime converted dataframe
    """
    # Create a copy of dataframe
    df = dataframe.copy()

    # Ensure that the index isn't already datetime and any date-like columns
    # aren't already datetime !
    # !-> If this is the case, return the original dataframe so as to preserve
    # memory!
    if isinstance(df.index, pd.DatetimeIndex):
        # Remove date-like columns since the dataframe is already setup with a
        # datetime index
        # pdf.drop(labels=None, *, axis=0, index=None, columns=None, level=None, inplace=False, errors='raise')
        df.drop(cols, axis=1, inplace=True)

        return df

    # Isolate the relevant column names, if any
    year_col = [col.lower() for col in df.columns]
    month_col =  [col.lower() for col in df.columns if col.lower() == "month"]

    try: # ? -> Refactor!
        # Separate year and month at least
        if year_col and month_col:
            # If there's no 'date' column, make it
            if "date" not in df.columns:
                # Check for a 'day' column
                if any("day" in col.lower() for col in df.columns):
                    day_col = [col.lower() for col in df.columns if col.lower()
                              == "day"][0]
                    # Convert to datetime
                    df["date"] = pd.to_datetime(
                        df[year_col].astype(str) + "-" + 
                        df[month_col].astype(str).str.zfill(2) + "-" +
                        df[day_col].astype(str).str.zfill(2),
                        format="%Y-%m-%d",
                        errors="coerce"
                    )

                else:
                    # No 'day' column
                    df["date"] = pd.to_datetime(
                        df[year_col].astype(str) + "-" +
                        # MOnth will need to often prepend a '0' for two-digit
                        # month
                        df[month_col].astype(str).str.zfill(2) + "-" +
                        "08", # day
                        format="%Y-%m-%d",
                        errors="coerce" # Invalid items set as NaT
                    )

            elif year_col and (not month_col):
                df["date"] = pd.to_datetime(
                    df[year_col].astype(str) + "-" +
                    "06" + "-" + # Arbitrary month
                    "08", # Arbitrary day
                    format="%Y-%m-%d",
                    errors="coerce"
                )


    except Exception as e:
        print(f"Issue with converting to datetime! {e}")

def get_weather_data(parameters, coordinates, year_range):
    """
    Calls to NASA POWER API for weather data
    ------------------------------------------------
    INPUT:
        parameters: (list) Parameters of the NASA Power API
        coordinates: (tuple) Latitute/Longitude
        year_range: (tuple) start_year/end_year

    OUTPUT:
        weather_data: (pd.DataFrame) Weather dataframe via specific parameters
    """
    try:
        # Implement NASA Power API call
        nasa = api(parameters, coordinates, year_range)
        # Weather data
        weather_data = nasa.get_weather_data()
        return weather_data
   
    except Exception as e:
        print(f"Something went wrong: {e}")
        return None




energy_df = read_energy_data("data/raw/Utility_Energy_Registry_Monthly_County_Energy_Use__Beginning_2021_20241208.csv")
weather_df = get_weather_data(["T2M,T2M_MAX,T2M_MIN,PRECTOTCORR,RH2M,"
              "ALLSKY_SFC_SW_DWN,CLOUD_AMT,WS10M,GWETROOT,QV2M,T2MWET"], (42, 44, -78, -76), (2001, 2024))
conversion = datetime_conversion(energy_df)
#breakpoint()
print(f"Execution time: {time.time() - start}")
