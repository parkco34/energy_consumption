#!/usr/bin/env python
"""
======================
NASA POWER API ? --> Needs to be refactored, if not redone completely
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
    """
    Class to call upon the  NASA POWER API to get weather data via specific
    parameters
    """
    def __init__(self,
                 parameters,
                 coordinates,
                 year_range
                ):
        # Unpack tuple of coordinates
        self.min_lat, self.max_lat, self.min_lon, self.max_lon =  coordinates
        # Unpack year range
        self.start_year, self.end_year = year_range
        """
        INPUT:
            parameters: (list)
            coordinates; (tuple)
            year_range: (tuple)

        OUTPUT:
            None
        """
        # parameter list
        self.parameters = parameters[0].split(",")

        # Build one URL per parameter
        self.base_urls = {
            p: (f"https://power.larc.nasa.gov/api/temporal/monthly/regional"
                f"?start=2001&end=2024&latitude-min={self.min_lat}"
                f"&latitude-max={self.max_lat}&longitude-min={self.min_lon}"
                f"&longitude-max={self.max_lon}&community=RE"
                f"&parameters={p}&format=json&user=parkdaddy&header=true"
                f"&time-standard=lst")
            for p in self.parameters
        }

    # Using asyncio library to fetch_parameters and fetching weather data
    async def fetch_parameter(self, session, param, url):
        """
        REturns a list of Series for one parameter
        ------------------------------------------
        INPUT:
            session: (aiohttp.client.ClientSession) Client Session to get url.
            param: (str) parameter ?
            url: (str) 

        OUTPUT:
            series_list: (list) List of pd.Series or an empty list if requests
            are not effective
        """
        try:
                # res: class aiohttp.client_reqrep.ClientResponse: Client Response  
            async with session.get(url) as res:
                # method ?
                res.raise_for_status()
                # Gets json data as dictionary
                data = await res.json()

                # correct JSON path
                param_data = data["features"][0]["properties"]["parameter"]

                series_list = []
                for name, val in param_data.items():
                    # dictionary w/ keys as dates and values as the
                    # parameter values
                    valid = {k: v for k, v in val.items() if
                             re.fullmatch(r"\d{6}", k) and 
                             1 <= int(k[-2:]) <= 12
                            }

                    # Ensure valid is not empty
                    if not valid:
                        continue

                    # Convert to pandas Series
                    series = pd.Series(valid, name=name)

                    # COnvert index to datetime index
                    series.index = pd.to_datetime(series.index, format="%Y%m")
                    # Append series to list
                    series_list.append(series)

                return series_list

        except Exception as e:
            print(f"Error fetching {param}: {e}")
            return []

    async def fetch_weather_data(self):
        """
        Asynchronoulsy collect all parameters into one DataFrame
        """
        async with aiohttp.ClientSession() as session:
            tasks = [
                self.fetch_parameter(session, param, url)
                for param, url in self.base_urls.items()
            ]
            results = await asyncio.gather(*tasks)

        all_series = [s for sub in results for s in sub]

        if not all_series:
            raise ValueError("NO valid data retrieved from NASA Power API")

        # Concatenate into one DataFrame
        df = pd.concat(all_series, axis=1)
        # Range relevant to the start and end years
        df = df[(df.index.year >= self.start_year) & (df.index.year <= self.end_year)]

        # Sort in descending order
        df = df.sort_index(ascending=False)

        return df

    def get_weather_data(self):
        """
        Fetch historical weather & solar data from NASA POWER API.Add commentMore actions
        Paramaters:
            T2M,T2M_MAX,T2M_MIN,PRECTOTCORR,RH2M,ALLSKY_SFC_SW_DWN,CLOUD_AMT,WS10M,GWETROOT,QV2M,T2MWET
        ----------------------------------------------------
        INPUT:
            start_year: (str) Beginning year
            end_year: (str) Ending year

        OUTPUT:
            df: (pd.DataFrame) Weather dataframe with correspoding parameters
        """
        df = asyncio.run(self.fetch_weather_data())

        return df



#parameters = ["T2M,T2M_MAX,T2M_MIN,PRECTOTCORR,RH2M,"
#              "ALLSKY_SFC_SW_DWN,CLOUD_AMT,WS10M,GWETROOT,QV2M,T2MWET"]
#nasa = NASAPowerAPI(parameters, 42, 44, -78, -76)
#weather = nasa.get_weather_data(2001, 2024)
#print(weather.head())
