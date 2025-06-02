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
                # Filter for clean date values
                for date in val.keys():
                    # If greater than 12, discard
                    if int(date[-2:]) > 12:
                        continue

                    # Convert to Series and parse dates
                    series = pd.Series(val, name=name)
                    breakpoint()
                    series.index = pd.to_datetime(series.index, format="%Y%m")



parameters = ["T2M,T2M_MAX,T2M_MIN,PRECTOTCORR,RH2M,ALLSKY_SFC_SW_DWN,CLOUD_AMT,WS10M,GWETROOT,QV2M,T2MWET"]
# For REGION
min_lat, max_lat, min_lon, max_lon = 42, 44, -78, -76
nasa = NASAPowerAPI(parameters, min_lat, max_lat, min_lon, max_lon)
weather = nasa.get_weather_data(2001, 2024)







