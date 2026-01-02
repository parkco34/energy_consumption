#!/usr/bin/env python
import asyncio
import aiohttp
import pandas as pd

# ony import these when using nasa_api import *
__all__ = ["featch_weather", "nasaapi"]


class NasaAPI:
    # primary url for api call
    _base = (
        "https://power.larc.nasa.gov/api/temporal/monthly/point"
        "?start={start}&end={end}&latitude={lat}&longitude={long}"
        "&community=re&parameters={param}&format=json"
        "&user=parkdaddy&header=true&time-standard=lst"
    )

    def __init__(self, date_range, coordinates, parameters):
        self.start_year, self.end_year = date_range
        self.lat, self.long = coordinates
        self.parameters = parameters

    async def fetch_url(self, session, param):
        """
        asynchronous function to fetch urls associated with api
        """
        url = self._base.format(
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
        input:
           None 

        output:
            datframe: (pd.dataframe) dataframe of data
        """
        async with aiohttp.ClientSession() as session:
            tasks = [self.fetch_url(session, p) for p in self.parameters]
            responses = await asyncio.gather(*tasks)

        # make parameter dictionary via dictionary comprehension
        param_dict = {self.parameters[i]:
                      responses[i]["properties"]["parameter"] for i in range(len(self.parameters))}

        series_dict = {}
        for param, res in zip(self.parameters, responses):
            try:
                data = res["properties"]["parameter"][param]

            except keyerror:
                raise keyerror(f"api response for '{param}' missing expected keys: {list(res.keys())}")

            series_dict[param] = pd.series(data, name=param)
            
        # align indices and return as dataframe
        return pd.concat(series_dict, axis=1, sort=True)

    async def get_data(self):
        """
        async returns weather dataframe
        """
        return await self._gather_all()

    def fetch(self):
        """
        call from regular code, when imported...
        produces a fresh event loop with 'asyncio.run'
        """
        try:
            return asyncio.run(self.get_data())

        except AttributeError as e:
            print(f"\n\nOOPS!\n{e}\n\n")

        finally:
            return None

def fetch_weather(
    date_range,
    coordinates,
    parameters,
    id_col="fips"
):
    """
    fetches nasa power monthly weather.
    """
    # case 1
    if isinstance(coordinates, tuple) and len(coordinates) == 2:
        lat, lon = coordinates
        return NasaAPI(date_range, (lat, lon), parameters).fetch()

    # case 2: many points (dataframe)
    if isinstance(coordinates, pd.DataFrame):
        required = {id_col, "lat", "lon"}

        missing = required - set(coordinates.columns)

        # account for missing coordinates and shit
        if missing:
            raise valueerror(f"coordinates dataframe missing columns: {missing}."
            f"have: {list(coordinates.columns)}")

        # gather frames ?
        frames = []
        for _, row in coordinates.iterrows():
            fips = str(row[id_col]).zfill(5)
            lat = float(row["lat"])
            lon = float(row["lon"])

            df = NasaAPI(date_range, (lat, lon), parameters).fetch()

            #df index like, 'yyyymm' from nasa: keep it, but attach fips
            df = df.copy()
            df[id_col] = fips
            frames.append(df)

        return pd.concat(frames, axis=0)
        
    raise typeerror("coordinates must be (lat, lon) tuple or a dataframe with columns [fips, lat, lon]")


if __name__ == "__main__":
    import time
    # ?
    def read_fips_county_data(filename):
        """
        reads external data for weather fips codes to be mapped to appropriate
        coorindates.
        --------------------------------------------------------------------
        input:
            filename: (str) filename (absolute path)

        output:
            df: (pd.dataframe) weather dataframe with fips and coordinates
        """
        # read file
        df = pd.read_csv(filename, sep="\t", on_bad_lines="skip")
        # correct dataframe columns
        df.columns = df.columns.str.strip()
        return df

    def load_fips_coords(external_datafile, state=None):
        """
        loads the fips/county data and filters (optionally) based on the state
        needed.
        ----------------------------------------------------------------------
        input:
            external_datafile:(str) absolute/relatice path ? to external data file.
            state: (str) acronymn for fir state wanted.

        output:
            state_df: (pd.dataframe) dataframe with state's fips codes, latitude
            and longitudinal coordinates for later merging with nasa power api
            weather data.
        """
        # load external data
        df = read_fips_county_data(external_datafile)

        # rename columns of dataframe (e.g. 'geoid' to 'fips')
        df.rename(columns={"GEOID": "fips", "INTPTLAT": "lat", "INTPTLONG": "lon"},
                 inplace=True)
       
        # filter depending on state chosen
        if state:
            state_data = df[df["USPS"] == state]
            # reset inde
            state_data.reset_index(inplace=True)
            # only keeping proper columns
            breakpoint()
            state_df =  state_data[["fips", "lat", "lon"]]

        return state_df


    parameters = ["t2m",
                  "t2m_max",
                  "t2m_min",
                  "prectotcorr",
                  "rh2m",
                  "allsky_sfc_sw_dwn",
                  "cloud_amt",
                  "ws10m",
                  "gwetroot",
                  "qv2m",
                  "t2mwet"]

    t0 = time.perf_counter()
    
    # load fips coordinates from external file
    fips_df = load_fips_coords("data/external/2022_Gaz_counties_national.txt", "NY")
    # get dataframe of weather data
    df_demo = fetch_weather((2001, 2024), fips_df, parameters)
    delt_t = time.perf_counter() - t0
    print(f"{df_demo.head()}")

    if delt_t > 8.0:
        print(f"\n\nway too long!!\n{delt_t:.1f} seconds\n\n")
