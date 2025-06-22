#!/usr/bin/env python
"""
Oracle Project
"""
import time
import os
#from data_utils.data_cleaning import DataCleaning
from nasa_power_api import NASAPowerAPI as api # Import NASA API Class
import pandas as pd
from data_utils.data_cleaning import DataCleaning

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

def datetime_conversion(dataframe):
    """
    Converts the date-like columns to datetime into one 'date' column, if it
    doesn't already exist, using arbitrary values for 'month' and/or 'day' if
    not already one of the dataframe columns
    --------------------------------------------------
    INPUT:
        dataframe: (pd.DataFrame) Original dataframe
        sort_by_date: (bool; default=True) Whether to sort the dataframe by
        datre or leave it as it is.

    OUTPUT:
        df: (pd.DataFrame) Datetime converted dataframe
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
        # If there's no 'date' column, make it
        if "date" not in df.columns:
            # Separate year and month at least
            if year_col and month_col:
                # Check for a 'day' column
                if any("day" in col.lower() for col in df.columns):
                    day_col = [col.lower() for col in df.columns if col.lower()
                              == "day"][0]
                    # Convert to datetime
                    df["date"] = pd.to_datetime(
                        df[year_col[0]].astype(str) + "-" + 
                        df[month_col[0]].astype(str).str.zfill(2) + "-" +
                        df[day_col].astype(str).str.zfill(2),
                        format="%Y-%m-%d",
                        errors="coerce"
                    )

                else:
                    # No 'day' column
                    df["date"] = pd.to_datetime(
                        df[year_col[0]].astype(str) + "-" +
                        # MOnth will need to often prepend a '0' for two-digit
                        # month
                        df[month_col[0]].astype(str).str.zfill(2) + "-" +
                        "08", # day
                        format="%Y-%m-%d",
                        errors="coerce" # Invalid items set as NaT
                    )

            elif year_col and (not month_col):
                # 'Day' columns already exists
                if any("day" in col.lower() for col in df.columns): # ?redundant
                    day_col = [col.lower() for col in df.columns if col.lower()
                              == "day"][0]

                    df["date"] = pd.to_datetime(
                        df[year_col[0]].astype(str) + "-" +
                        "06" + "-" + # Arbitrary month
                        # prepend '0' to a two-digit day
                        df[day_col].astype(str).str.zfill(2), 
                        format="%Y-%m-%d",
                        errors="coerce"
                    )

                else:
                    # Include 'day' column
                    df["date"] = pd.to_datetime(
                        df[year_col[0]].astype(str) + "-" +
                        "06" + "-" +
                        "08",
                        format="%Y-%m-%d",
                        errors="coerce"
                    )

    except Exception as e:
        print(f"Issue with converting to datetime! {e}")

    return df

def make_datetime_index(dataframe):
    """
    Makes the 'date' column into the datetime index for the dataframe,
    removing the other date-like columns.
    First, you must sort the datetime values in ascending (or whatevs) using,
        ;pd.sort_values(*, axis=0, ascending=True, inplace=False, kind='quicksort', 
        na_position='last', ignore_index=False, key=None)
        
    This is done by finding the datetime column and converting it to the index
    using, 
        'pd.set_index(keys, *, drop=True, append=False, inplace=False, verify_integrity=False)'
    then dropping the remaining date-like columns using 
        'pd.drop(labels=None, *, axis=0, index=None, columns=None, level=None, inplace=False, errors='raise')'

    Finally, we sort the index in ascending order (date going from latest to
    oldest):
        'DataFrame.sort_index(*, axis=0, level=None, ascending=True,
        inplace=False, kind='quicksort', na_position='last',
        sort_remaining=True, ignore_index=False, key=None)'

    Now we have our new dataframe without any date-like columns and a datetime
    index, in ascending order.
    ------------------------------------------------------------------------
    INPUT:
        dataframe: (pd.DataFrame) Dataframe with 'date' column as datetime

    OUTPUT: 
        df: (pd.DataFrame) New dataframe with the index as the datetime index,
        with no other date-like columns unless neccessary
    """
    # Copy dataframe
    df = dataframe.copy()

    # Relevant date-like keywords
    remove_cols = [col.lower() for col in df.columns if col in DATE_KEYWORDS]

    # Convert relevant columns to datetime columns
    converted_df = datetime_conversion(df)
    # Sort dates, ascending
    new_df = converted_df.set_index("date", drop="date")
    # Drop daete-like columns; no longer needed
    new_df.drop(remove_cols, axis=1, inplace=True)
    
    # Sort the dataframe's index in descending order
    new_df.sort_index(axis=0, ascending=False, inplace=True)
   
    return new_df
    # 
def _get_weather_data(parameters, coordinates, year_range):
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

def combine_dataframes(energy_df, weather_df):
    """
    Combines the two dataframes using
        'pd..concat(objs, *, axis=0, join='outer', ignore_index=False, 
        keys=None, levels=None, names=None, verify_integrity=False, 
        sort=False, copy=None)'
        ---------------------------------------------------------------
        INPUT:
            energy_df: (pd.DataFrame) Energy dataframe in this case
            weather_df: (pd.DataFrame) Weather dataframe in this case

        OUTPUT:
            df: (pd.DataFrame) Concatenated dataframe
    """
    # Ensure both dataframes use identical DatetimeIndex
    energy = energy_df.copy()
    weather = weather_df.copy()

    # Outer join on index
    df = energy.join(weather, how="outer")

    # Sort newest-oldestr to match existing design
    df.sort_index(ascending=False, inplace=True)

    return df

def main():
    # Get dataframes
    energy_df = read_energy_data("data/raw/Utility_Energy_Registry_Monthly_County_Energy_Use__Beginning_2021_20241208.csv")
    weather_df = _get_weather_data(["T2M,T2M_MAX,T2M_MIN,PRECTOTCORR,RH2M,"
                  "ALLSKY_SFC_SW_DWN,CLOUD_AMT,WS10M,GWETROOT,QV2M,T2MWET"], (42, 44, -78, -76), (2001, 2024))

    # Convert to datetime index
    proper_energy_df = make_datetime_index(energy_df)

    breakpoint()
    # Merge dataframes int o one main dataframe
    dframe = combine_dataframes(proper_energy_df, weather_df)

    # Clean the data plz
    clean_obj = DataCleaning(dframe)
    col_summary = clean_obj.column_summary()

if __name__ == "__main__":
    start = time.time()
    main()
    print(f"Execution time: {time.time() - start}")
