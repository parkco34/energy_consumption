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
        params = {
            "parameters": "T2M,RH2M,WS10M,ALLSKY_SFC_SW_DWN",  # Temp, humidity, wind, solar
            # Dates need month and day concatenated at the end
            "start": f"{start_year}0101",
            "end": f"{end_year}1231",
            "latitude": lat,
            "longitude": lon,
            "community": "RE",  # Renewable Energy
            "format": "JSON"
        }

        # Sends a get request: requests.get(url, params=None, *kwargs)
        # params: Dictionary , list or tuples or bytes to send in query string
        response = requests.get(self.base_url, params=params)
        
        if response.status_code == 200:
            return response.json()

        else:
            print(f"Error: {response.status_code}, {response.text}")
            return None

def max_depth_iddfs(data):
    """
    ChatGPT ?
    ======
    Find maximum depth of nested dictionaries using Iterative Deepening DFS.
    ----------------------------------------------------------------------------
    Depth limit 0: Searches root level - finds weather dictionary ✓
    Depth limit 1: Searches first level - finds dictionaries like "geometry", "header" ✓
    Depth limit 2: Searches second level - finds dictionaries like "header"["api"], "parameters"["T2M"] ✓
    Depth limit 3: Searches third level - finds leaf dictionaries with "units", "longname" ✓
    Depth limit 4: Searches fourth level - finds only strings/numbers, no more dictionaries ✗
    -----------------------------------------------------------------------------
    INPUT:
        data: (dict) The root dictionary to analyze
   
    OUTPUT:
        max_depth_found: (int) Maximum depth (root = 0)
    """
    def depth_limited_search(obj, current_depth, depth_limit):
        """
        Perform DFS up to a specified depth limit.
        
        Args:
            obj: Current object being examined
            current_depth: How deep we currently are
            depth_limit: Maximum depth to explore in this iteration
        
        Returns:
            bool: True if we found a dictionary at exactly depth_limit
        """
        # Base case: if we've reached our depth limit
        if current_depth == depth_limit:
            # Check if this object is a dictionary (indicating deeper structure exists)
            return isinstance(obj, dict) and len(obj) > 0
        
        # If current object is a dictionary and we haven't hit the limit, explore deeper
        if isinstance(obj, dict):

            for value in obj.values():
                # Recursively search deeper, incrementing our current depth
                if depth_limited_search(value, current_depth + 1, depth_limit):
                    return True
        
        # No dictionary found at the target depth in this branch
        return False
    
    # Start iterative deepening process
    max_depth_found = 0
    current_depth_limit = 0
    
    # Keep searching with increasing depth limits
    while True:
        # Try to find a dictionary at the current depth limit
        found_dict_at_limit = depth_limited_search(data, 0, current_depth_limit)
        
        if found_dict_at_limit:
            # We found structure at this depth, so update our maximum
            max_depth_found = current_depth_limit
            # Try the next depth level
            current_depth_limit += 1
        else:
            # No dictionaries found at this depth, we've reached the maximum
            break
    
    return max_depth_found


nasa = NASAPowerAPI()
lat, lon = 43.1566, -77.6088
weather = nasa.get_weather_data(lat, lon, 2021, 2024)
thing = max_depth_iddfs(weather)
breakpoint()


# Example Usage
#if __name__ == "__main__":
#    lat, lon = 43.1566, -77.6088  # Rochester, NY
#    start_year = "2021"
#    end_year = "2024"
#
#    nasa_api = NASAPowerAPI()
#    weather_data = nasa_api.get_weather_data(lat, lon, start_year, end_year)
#
#    if weather_data:
#        print(weather_data)  # Print retrieved weather & solar data
#
#    breakpoint()
