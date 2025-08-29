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
    ? -> Swap coordinates with FIP codes !?
    ? -> How does this function work???
    ? -> Iterate over multiple coordinates!

    Synchronous one-liner ideal for caller who just want the DataFrme.
    ------------------------------------------------------------------
    INPUT:
        date_range: (tuple) Start and end dates
        coordinates: (pd.DataFrame) Tabular data of coordinates
        parameters: (list of strings) Strings ?

    OUTPUT:
        NasaAPI
    """
    for idx in range(len(coordinates[["lat", "lon"]])):
        # Assign latitude and longitude
        lat, lon = (coordinates.loc[idx][["lat", "lon"]]["lat"],
            coordinates.loc[idx][["lat", "lon"]]["lon"])
        
        return NasaAPI(date_range, (lat, lon), parameters).fetch()



if __name__ == "__main__":
    import time
    # ?
    def read_fips_county_data(filename):
        """
        Reads external data for weather FIPS codes to be mapped to appropriate
        coorindates.
        --------------------------------------------------------------------
        INPUT:
            filename: (str) Filename (absolute path)

        OUTPUT:
            df: (pd.DataFrame) Weather dataframe with FIPS and coordinates
        """
        # Read file
        df = pd.read_csv(filename, sep="\t", on_bad_lines="skip")
        # Correct dataframe columns
        df.columns = df.columns.str.strip()
        return df

    def load_fips_coords(external_datafile, state=None):
        """
        Loads the fips/county data and filters (optionally) based on the state
        needed.
        ----------------------------------------------------------------------
        INPUT:
            external_datafile:(str) Absolute/relatice path ? to external data file.
            state: (str) Acronymn for fir state wanted.

        OUTPUT:
            state_df: (pd.DataFrame) Dataframe with state's fips codes, latitude
            and longitudinal coordinates for later merging with NASA POWER API
            weather data.
        """
        # Load external data
        df = read_fips_county_data(external_datafile)

        # Rename columns of dataframe (e.g. 'GEOID' to 'FIPS')
        df.rename(columns={"GEOID": "fips", "INTPTLAT": "lat", "INTPTLONG": "lon"},
                 inplace=True)
       
        # Filter depending on state chosen
        if state:
            state_data = df[df["USPS"] == state]
            # REset inde
            state_data.reset_index(inplace=True)
            # Only keeping proper columns
            state_df =  state_data[["fips", "lat", "lon"]]

        return state_df


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
    
    # Load FIPS coordinates from external file
    fips_df = load_fips_coords("./data/external/2022_Gaz_116CDs_national.txt", "NY")
    # Get dataframe of weather data
    df_demo = fetch_weather((2001, 2024), fips_df, parameters)
    delt_t = time.perf_counter() - t0
    print(f"{df_demo.head()}")

    if delt_t > 8.0:
        print(f"\n\nWay too long!!\n{delt_t:.1f} seconds\n\n")
