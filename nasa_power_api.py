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

class NASAPowerAPI:
    def __init__(self, parameters, min_lat, max_lat, min_lon, max_lon):
        # Latitude/Longitude
        self.min_lat, self.max_lat, self.min_lon, self.max_lon = min_lat, max_lat, min_lon, max_lon

        # List of parameters
        self.parameters = parameters[0].split(",")

        # Base URL used for accessing API, looping through each relevant
        # parameter, where I need to replace 'param' with the actual
        # parameter(s) to be used for the REGION... Storing the shit in a
        # dictionary
        self.base_urls = {}
        for idx, param in enumerate(self.parameters):
            self.base_urls[idx] = f"https://power.larc.nasa.gov/api/temporal/monthly/regional?start=2001&end=2024&latitude-min=42&latitude-max=44&longitude-min=-78&longitude-max=-76&community=re&parameters={param}&format=json&user=parkdaddy&header=true&time-standard=lst"

    def get_weather_data(self, url, start_year, end_year):
        """
        Fetch historical weather & solar data from NASA POWER API.
        Paramaters:
            T2M,T2M_MAX,T2M_MIN,PRECTOTCORR,RH2M,ALLSKY_SFC_SW_DWN,CLOUD_AMT,WS10M,GWETROOT,QV2M,T2MWET
        -------------------------------------------------------------------------------------------------
        INPUT:
            ur: (str) URL for given parameter
            start_year (str): Start year (YYYY).
            end_year (str): End year (YYYY).

        OUTPUT:
            dict: JSON response containing weather & solar data.
        """
        # Loop through the dictionary, getting each parameter
        # I need coordinates, temp, and index by dates and columns should be
        # the parameters
        req = requests.get(item)
        text = req.json()
        feat = text["features"][0]
        # Get values from parameter
        param_dict = feat["propoerties"]["parameter"]
        columns = []
        # Iterate thru dictionary items to create pandas Series
        for name, vals in param_dict.items():
            series = pd.Series(val, name=name)
            # Create datetime index
            series.index = pd.to_datetime(series.index, format="%Y%m")
            columns.append(series)
            breakpoint()



parameters = ["T2M,T2M_MAX,T2M_MIN,PRECTOTCORR,RH2M,ALLSKY_SFC_SW_DWN,CLOUD_AMT,WS10M,GWETROOT,QV2M,T2MWET"]
# For REGION
min_lat, max_lat, min_lon, max_lon = 42, 44, -78, -76
nasa = NASAPowerAPI(parameters, min_lat, max_lat, min_lon, max_lon)
#lat, lon = 43.1566, -77.6088  # Rochester, NY
weather = nasa.get_weather_data(2001, 2024)







