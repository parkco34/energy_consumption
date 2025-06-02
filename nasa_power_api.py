#!/usr/bin/env python
"""
======================
NASA POWER API
======================
Includes long-term climatologically averaged estimates of meteorological quantities adn surface solar energy fluxes:
    -> Climate - Long term stastical patterns, about 30 years+
    -> Weather - short-term atmospheric conditions
Averaging process smooths out short-term weather variability to reveal underlying climate patterns.

Estimates - Computed values, where the data comes from numerical models.

Meterological Quantities - Standard atmospheric variables like temperature, pressure, humidity, wind speed and direction, precipitation, and cloud cover. These are the fundamental state variables that describe atmospheric conditions.

Surface Solar Energy Fluxes - Ratge of solar energy transfer at Earth's surface, measured in (W/m^2) Watts per sqaure meter.
------------------------------------------------
The data is global and contiguous in time.

------------------------------------------------
T2M → Temperature at 2 meters (°C)
RH2M → Relative Humidity at 2 meters (%)
WS10M → Wind Speed at 10 meters (m/s)
ALLSKY_SFC_SW_DWN → Solar radiation (W/m²)
"""
import pandas as pd
import requests
import re
import time
import asyncio
import aiohttp

class NASAPowerAPI:
    def __init__(self, parameters, min_lat, max_lat, min_lon, max_lon):
        # Latitude/Longitude
        self.min_lat, self.max_lat, self.min_lon, self.max_lon = min_lat, max_lat, min_lon, max_lon

        # List of parameters
        self.parameters = parameters[0].split(",")

        # Base URL used for accessing API, looping through each relevant
        # parameter, where I need to replace 'param' with the actual
        # parameter(s) to be used for the REGION... Storing the shit in a
        self.base_urls = {}
        for param in self.parameters:
            self.base_urls[param] = \
            f"https://power.larc.nasa.gov/api/temporal/monthly/regional?start=2001&end=2024&latitude-min={self.min_lat}&latitude-max={self.max_lat}&longitude-min={self.min_lon}&longitude-max={self.max_lon}&community=re&parameters={param}&format=json&user=parkdaddy&header=true&time-standard=lst"

    def get_weather_data(self, start_year, end_year):
        """
        Fetch historical weather & solar data from NASA POWER API.
        Paramaters:
            T2M,T2M_MAX,T2M_MIN,PRECTOTCORR,RH2M,ALLSKY_SFC_SW_DWN,CLOUD_AMT,WS10M,GWETROOT,QV2M,T2MWET
        -------------------------------------------------------------------------------------------------
        INPUT:
            start_year (str): Start year (YYYY).
            end_year (str): End year (YYYY).

        OUTPUT:
            df: (pd.DataFrame) Dataframe with datetime index
        """
        # Store json data in pandas Series in List
        all_series = []

        # Measure how long it takes for loop to complete
        start = time.time()

        # Iterate thru parameters via url and request that shit
        for param, url in self.base_urls.items():
            res = requests.get(url)
            # Raise error for bad status
            res.raise_for_status()
            
            # Get json data form
            data = res.json()

            try:
                # Get parameter data
                param_data = data["features"][0]["properties"]["parameter"]

            except KeyError:
                print(f"Data missing for parameter: {param}")
                # Go to next item
                continue

            # Iterate thru names and values of parameter data
            for name, val in param_data.items():
                # Filters bad date values (for example: 202413)
                valid_entries = {k: v for k, v in val.items() if re.fullmatch(r"\d{6}", k) and 1 <= int(k[-2:]) <= 12}

                if not valid_entries:
                    # In case there's no valid entries
                    continue

                # Convert to Series and parse dates
                series = pd.Series(valid_entries, name=name)

                try:
                    series.index = pd.to_datetime(series.index, format="%Y%m")

                except Exception as e:
                    print(f"Date parsing error for parameter: {param}\nError: {e}")

                all_series.append(series)

        end = time.time()

        print(f"Execution time: {end - start}")

        # Combine all series into a daraframe
        if not all_series:
            raise ValueError("No valid dates from NASA Power API")

        # Concatenate dataframes
        df = pd.concat(all_series, axis=1)

        # Filter by date range
        df = df[(df.index.year >= start_year) & (df.index.year <= end_year)]

        return df

    async def fetch_parameter(self, session, param, url):
        """
        Uses asynchronous method for faster execution.
        ------------------------------------------------
        INPUT:
            session: ()

        OUTPUT:
        """
        try:
            async with session.get(url) as res:
                # await keyword for yeilding control back to event loop,
                # allowing other tasks to run
                data = await res.json()
                param_data = data["features"][0]["parameter"]
                series_list = []

                # Iterate thru name-value pairs got relevant data
                for name, val in param_data.items():
                    # Only allow valid entries (not 202413)
                    # USing regex for 6 digits and month is less than 13
                    valid_entries = {k: v for k, v in val.items() if re.fullmatch(r"\d{6}", k) and 1 <= int(k[-2:]) <= 12}

                    if not valid_entries:
                        continue

                    # Create pandas Series
                    series = pd.Series(valid_entries, name=name)

                    # Convert index to datetime index
                    try:
                        series.index = pd.to_datetime(series.index, format="%Y%m")
                        series_list.append(series)

                    except Exception as e:
                        print(f"Date parsing error for: {param}: {e}")

                return series_list

        except Exception as e:
            print(f"Error fetching parameter: {param}: {e}")
            # Return empty list
            return []

    async def fetch_weather_data(self, start_year, end_year):
        """
        Fetch historical weather & solar data from NASA POWER API.
        Paramaters:
            T2M,T2M_MAX,T2M_MIN,PRECTOTCORR,RH2M,ALLSKY_SFC_SW_DWN,CLOUD_AMT,WS10M,GWETROOT,QV2M,T2MWET
        -------------------------------------------------------------------------------------------------
        INPUT:
            start_year (str): Start year (YYYY).
            end_year (str): End year (YYYY).

        OUTPUT:
            df: (pd.DataFrame) Dataframe with datetime index
        """
        # Initialize empty task list
        task = []
        
        # Create asynchronous HTTP session w/ ClientSession and assign session
        # object, where it'll automatically close
        async with aiohttp.ClienSession() as session:
            for param, url in self.base_urls.items():
                task.append(self.fetch_parameter(session, url, param))

            # Launch API requests at same time and aggregate results into a
            # list
            results = await asyncio.gather(*tasks)

        # Flatten list of lists
        all_series = [series for sublist in results for series in sublist]
       
        if not all_series:
            raise ValueError("No valid data retrieved from NASA Power API")

        # Concatenate lists and convert to a DataFrame
        df = pd.concat(all_series, axis=1)

        # Ensure datetime index is within the appropriate range
        df = df[(df.index.year >= start_year) & (df.index.year < end_year)]
    
        return df

    def get_weather_data(self, start_year, end_year):
        """
        Synchronously wraps the async method and measure execution time.
        ---------------------------------------------------------------
        INPUT:
            start_year: (str) Earliest year in range
            end_year: (str) Latest year in range

        OUTPUT:
            df: (pd.DataFrame) Dataframe with the results of the request
        """
        start_time = time.time()
        # Execute coroutine adn return result and closes the executor
        df = asyncio.run(self.fetch_weather_data(start_year, end_year))
        end_time = time.time()
        print(f"Execution time: {end_time - start_time}")
        return df



parameters = ["T2M,T2M_MAX,T2M_MIN,PRECTOTCORR,RH2M,ALLSKY_SFC_SW_DWN,CLOUD_AMT,WS10M,GWETROOT,QV2M,T2MWET"]
# For REGION
min_lat, max_lat, min_lon, max_lon = 42, 44, -78, -76
nasa = NASAPowerAPI(parameters, min_lat, max_lat, min_lon, max_lon)
weather = nasa.get_weather_data(2001, 2024)







