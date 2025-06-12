#!/usr/bin/env python
"""
Oracle Project
"""
import time
start = time.time()
import os
from data_utils.data_cleaning import DataCleaning
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
    Looks for a 'date' column and converts it to a datetime.
    If there's separate month, year columns, it properly combines them into one
    datetime column, removing the original columns.
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
    date_keywords = ["month", "year", "date", "day"]

    # Initiate emptry to track converted columns 
    converted_cols = []

    # CHeck if separate year and/or month rolumns
    month_col = any("month" for col in df.columns)
    year_col = any("year" for col in df.columns)

    # If separate year and month columns, try ti create datetime columns
            # get correct column names via list comprehension, extracting first element from the list

            # Create new 'date' column if none exists

                # if there's a 'day' column, use it!
                
                # If no day column, use the 8th day of the month

                # Append converted columns to the list

    # Process each column that might contain date information

        # Skip columns we've already processed

        # CHeck if any date keywords appear int he column name

            # Get a sample of non-null values;  determine the format

            # Make user aware that ther sare no non-null values for the column

            # Check for different data formats, like if hte value has all date components

            # Date components

                # Format "20250608"

                # Try pandas automatic parsing as a last resort

    # Ensure 'date' column is the first column of dataframe

    # sort the dataframe by 'date', ascending

    # Reset index

    return new_df


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
breakpoint()
print(f"Execution time: {time.time() - start}")
