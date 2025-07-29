#!/usr/bin/env python
import asyncio
import aiohttp
import pandas as pd

# Ony import these when using nasa_api import *
__all__ = ["featch_weather", "NasaAPI"]


class NasaAPI:
    # Primary URL for API call
    _BASE = (
        "https://power.larc.nasa.gov/api/temporal/monthly/point"
        "?start={start}&end={end}&latitude={lat}&longitude={long}"
        "&community=re&parameters={param}&format=json"
        "&user=Parkdaddy&header=true&time-standard=lst"
    )

    def __init__(self, date_range, coordinates, parameters):
        self.start_year, self.end_year = date_range
        self.lat, self.long = coordinates
        self.parameters = parameters

    async def fetch_url(self, session, param):
        """
        Asynchronous function to fetch urls associated with API
        """
        url = self._BASE.format(
            start=self.start_year, 
            end=self.end_year,
            lat=self.lat,
            long=self.long,
            param=param)

        async with session.get(url, timeout=20) as response:
            # ?
            response.raise_for_status()
            return await response.json()

    async def _gather_all(self):
        """
        INPUT:
            None

        OUTPUT:
            datframe: (pd.DataFrame) Dataframe of data
        """
        async with aiohttp.ClientSession() as session:
            tasks = [self.fetch_url(session, p) for p in self.parameters]
            responses = await asyncio.gather(*tasks)

        # Make parameter dictionary via dictionary comprehension
        param_dict = {self.parameters[i]:
                      responses[i]["properties"]["parameter"] for i in range(len(self.parameters))}

        series_dict = {}
        for param, res in zip(self.parameters, responses):
            try:
                data = res["properties"]["parameter"][param]

            except KeyError:
                raise KeyError(f"API response for '{param}' missing expected keys: {list(res.keys())}") from None

            series_dict[param] = pd.Series(data, name=param)
            
        # Align indices and return as DataFrame
        return pd.concat(series_dict, axis=1, sort=True)

    async def get_data(self):
        """
        Async returns weather dataframe
        """
        return await self._gather_all()

    def fetch(self):
        """
        Call from regular code, when imported...
        Produces a fresh event loop with 'asyncio.run'
        """
        return asyncio.run(self.get_data())

   
def fetch_weather(date_range,
                  coordinates,
                  parameters):
    """
    Synchronous one-liner ideal for caller who just want the DataFrme.
    """
    return NasaAPI(date_range, coordinates, parameters).fetch()



if __name__ == "__main__":
    import time

    parameters = ["T2M",
                  "T2M_MAX",
                  "T2M_MIN",
                  "PRECTOTCORR",
                  "RH2M",
                  "ALLSKY_SFC_SW_DWN",
                  "CLOUD_AMT",
                  "WS10M",
                  "GWETROOT",
                  "QV2M",
                  "T2MWET"]

    t0 = time.perf_counter()
    # Get dataframe of weather data
    df_demo = fetch_weather((2001, 2024), (42.5, -77), parameters)
    delt_t = time.perf_counter() - t0
    print(f"{df_demo.head()}")

    if delt_t > 8.0:
        print(f"\n\nWay too long!!\n{delt_t:.1f} seconds\n\n")
