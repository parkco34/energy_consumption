#!/usr/bin/env python
"""
Oracle Project
"""
import traceback
import re
import sys
import time
import os
#from data_utils.data_cleaning import DataCleaning
#from nasa_power_api import NASAPowerAPI as api # Import NASA API Class
import pandas as pd
from data_utils.data_cleaning import DataCleaning
import sys

# List of common date-related keywords for columns
_DATE_KEYWORDS = ["month", "year", "date", "day", "time"]

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

def make_datetime_col(dataframe):
    """
    Converts the date-like columns to datetime into one 'date' column, if it
    doesn't already exist, using arbitrary values for 'month' and/or 'day' if
    not already one of the dataframe columns.
    ? -> Need to deal with the cases where there's a 'date' column already.
    --------------------------------------------------------------------
    INPUT:
        dataframe: (pd.DataFrame) Original dataframe

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
        print(f"\n\n{df.index} is DateTime index already!\n\n")
        # Remove date-like columns since the dataframe is already setup with a
        # datetime index
        # pdf.drop(labels=None, *, axis=0, index=None, columns=None, level=None, inplace=False, errors='raise')
        df.drop(cols, axis=1, inplace=True)

        return df

    # If index in date format via strings, convert to datetime indices
    elif isinstance(df.index[0], str): # ? More conditions needed, obviously
        # Check if format is same as "YYYY-MM-DD"
        # 1900's or 2000's, valid two-digit month
        pattern = re.compile(r"^(19|20)\d{2}(0[1-9]|1[0-2])$")
        if df.index.to_series().apply(pattern.fullmatch).all():
            # Remove all invalid dates (13-months) via mask
            good_mask = df.index.str.match(pattern)
            # New dataframe without invalid months
            df = df[good_mask]

            # Create 'date' column with appropriate formatting
            dates = df.index.to_series().apply(lambda x: x[:4] + "-"+ x[4:7] + "-" + "01")
            # Add to the dataframe
            df["date"] = dates
            
            # Convert to datetime
            df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")

        # Replace indices with datetime indices and delete the 'date' column
        df.index = df["date"]

        return df

    # Isolate the relevant column names, if any
    year_col = [col.lower() for col in df.columns if col.lower() == "year"]
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

            elif (not year_col) and (not month_col):
                print(f"\n\nNo year_col and no month_col\n\n")
                if any("day" in col.lower() for col in df.columns):
                    day_col = [col.lower() for col in df.columns if col.lower()
                              == "day"][0]
                            
                else:
                    print(f"\n\nNo year/month column: \nColumns: {df.columns}\n\n")

        else:
            print(f"'date' in the dataframe's columns: {df.columns}")

    except KeyError as e:
        print(f"KeyError!\nIssue with converting to datetime! {e}")

    return df

def make_datetime_index(dataframe):
    """
    Makes the 'date' column into the datetime index for the dataframe,
    removing the other date-like columns.
    First, you must sort the datetime values in ascending (or whatevs) using,
        ;pd.sort_values(*, axis=0, ascending=True, inplace=False, kind='quicksort', 
        a_position='last', ignore_index=False, key=None)
        
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
    remove_cols = [col.lower() for col in df.columns if col in _DATE_KEYWORDS]

    # Add 'date' as datetime column in dataframe
    df_with_date = make_datetime_col(df)
    
    # Sort dates, ascending
    converted_df = df_with_date.set_index("date", drop="date")
    # Drop daete-like columns; no longer needed
    converted_df.drop(remove_cols, axis=1, inplace=True)
    
    # Sort the dataframe's index in descending order
    converted_df.sort_index(axis=0, ascending=False, inplace=True)
   
    return converted_df
    # 
def _get_weather_data(date_range, coordinates, parameters):
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
    # Import NasaAPI module
    from nasa_api import fetch_weather

    try:
        # Implement NASA Power API call
        weather_data = fetch_weather(date_range, coordinates, parameters)
        # Weather data
        return weather_data
   
    except Exception as e:
        print("========================================")
        print(f"\n\nSomething went wrong: {e}\n\n")
        print("========================================")

        # OUtput traceback information including filename and linenumber
        exc_type, exc_value, exc_traceback = sys.exc_info()
        tb_list = traceback.extract_tb(exc_traceback)
        # Last element of traceback list, for the error location
        error_location = tb_list[-1]

        # Filename, line number output
        filename = error_location.filename
        proper_filename = filename.rpartition('/')[-1]
        linenumber = error_location.lineno
        print(f"Filename: '{proper_filename}'")
        print(f"Error occurred @ line number: {linenumber}")

        # If this doesn't work, piss off
        exit()

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
    def align_to_month(dataframe):
        """
        Inner function to convert datetime index to month-ending periods and
        back to cannonical date.
        ------------------------------------------------
        INPUT:
            dataframe: (pd.DataFrame) DataFrame to change.

        OUTPUT:
            df: (pd.DataFrame) Sorted/modified dataframe
        """
        # Copy dataframes
        df = dataframe.copy
        # Convert to PeriodIndex at monthly frequency using
        # pd.DatetimeIndex.to_period function
        dframe = df.index.to_period("M")
        # Choose first dat of month
        dfframe.index = dframe.index.to_timestamp("M")

        return dframe.sort_index()

    # Energy/Weather dataframes
    energy_aligned = align_to_month(energy_df)
    weather_aligned = align_to_month(weather_df)

    # Inner join keeps only months present


def main():
    # Get dataframes
    energy_df = read_energy_data("data/raw/Utility_Energy_Registry_Monthly_County_Energy_Use__Beginning_2021_20241208.csv")
    weather_df = _get_weather_data(
        (2001, 2024), 
        (42, -77), 
        ["T2M","T2M_MAX","T2M_MIN","PRECTOTCORR","RH2M", 
            "ALLSKY_SFC_SW_DWN","CLOUD_AMT","WS10M","GWETROOT","QV2M","T2MWET"]
    )
    
    # Convert to datetime index
    proper_energy_df = make_datetime_index(energy_df)
    proper_weather_df = make_datetime_index(weather_df)

    # Merge dataframes int o one main dataframe
    dframe = combine_dataframes(proper_energy_df, proper_weather_df)
    breakpoint()

    # Clean the data plz
#    clean_obj = DataCleaning(dframe)
#    col_summary = clean_obj.column_summary()
#    print(f"Column Summary: \n{col_summary}")


if __name__ == "__main__":
    main()
