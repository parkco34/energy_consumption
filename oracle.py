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

    # List of common date-related keywords for columns
    DATE_KEYWORDS = ["month", "year", "date", "day"]

    # If separate month and year, since those are what this script is
    # intereted in ...
    has_month = any(col for col in df.columns if col.lower() == "month")
    has_year = any(col for col in df.columns if col.lower() == "year")

    # Ensure that the index isn't already datetime and any date-like columns
    # aren't already datetime !
    # !-> If this is the case, return the original dataframe so as to preserve
    # memory!
    if isinstance(df.index, pd.DatetimeIndex):
        # get list of columns to remove from dataframe
        remove_cols = [col for col in df.columns if col in DATE_KEYWORDS]

        # If there no other date-like columns, nevermind
        if not remove_cols:
            return dataframe

        else:
        # If there are, remove them inplace, returning the copy
        # pdf.drop(labels=None, *, axis=0, index=None, columns=None, level=None, inplace=False, errors='raise')
            df.drop(remove_cols, axis=1, inplace=True)

            return df
    
    # Separate date-like columns
    if has_month and has_year:
        # Get column name without knowing where it is via list comprehension,
        # only including the first instance
        month_col = [col.lower() for col in df.columns if col.lower() ==
                     "month"][0]
        year_col = [col.lower() for col in df.columns if col.lower() ==
                    "year"][0]

        # Determine if there's a 'day' column
        if not "day" in df.columns:
            # add an arbtitrary day to the date for fromatting purposes
            # .str -> Accessor used to apply string methods to each element in
            # the Series
            df["date"] = pd.to_datetime(
                df[year_col].astype(str) + "-" +
                df[month_col].astype(str).str.zfill(2) +  "-"
                "08",
                format="%Y-%m-%d",
                errors="coerce"
            )

        else:
            # 'day' column exists; get column name via list comprehension
            day_col = [col.lower for col in df.columns if col.lower() ==
                       "day"][0]

            df["date"] = pd.to_datetime(
                df[year_col].astype(str) + "-" +
                df[month_col].astype(str).str.zfill(2) +  "-" +
                df[day_col].astype(str).str.zfill(2),
                format="%Y-%m-%d",
                errors="coerce"
            )

    # Date-like columns are not separate
    elif has_month or has_year:
        pass


    # Determine date-like columns and datatypes and whether they're separate or
    # together (year, month, or YYmm, etc.)

    # Convert date-like columns accordingly to datetime

    # Make the datetime column the index and remove the date-like columns

    # Return new dataframe like a ( ͡° ͜ʖ ͡°  ) boss 
    pass


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




energy_df = read_energy_data("data/raw/Utility_Energy_Registry_Monthly_County_Energy_Use__Beginning_2021_20241208.csv")
weather_df = get_weather_data(["T2M,T2M_MAX,T2M_MIN,PRECTOTCORR,RH2M,"
              "ALLSKY_SFC_SW_DWN,CLOUD_AMT,WS10M,GWETROOT,QV2M,T2MWET"], (42, 44, -78, -76), (2001, 2024))
conversion = datetime_conversion(energy_df)
#breakpoint()
print(f"Execution time: {time.time() - start}")
