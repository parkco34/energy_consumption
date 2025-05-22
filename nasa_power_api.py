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
import requests

class NASAPowerAPI:
    def __init__(self):
        # Base URL used for accessing API
        self.base_url = "https://power.larc.nasa.gov/api/temporal/daily/point"

    def get_weather_data(self, lat, lon, start_year, end_year):
        """
        Fetch historical weather & solar data from NASA POWER API.
        
        ---------------------------------------------------------------
        INPUT:
            lat (float): Latitude.
            lon (float): Longitude.
            start_year (str): Start year (YYYY).
            end_year (str): End year (YYYY).

        OUTPUT:
            dict: JSON response containing weather & solar data.
        """
        pass
