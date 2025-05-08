#!/usr/bin/env python
"""
T2M → Temperature at 2 meters (°C)
RH2M → Relative Humidity at 2 meters (%)
WS10M → Wind Speed at 10 meters (m/s)
ALLSKY_SFC_SW_DWN → Solar radiation (W/m²)
"""
import requests

class NASAPowerAPI:
    def __init__(self):
        self.base_url = "https://power.larc.nasa.gov/api/temporal/daily/point"

    def get_weather_data(self, lat, lon, start_year, end_year):
        """
        Fetch historical weather & solar data from NASA POWER API.

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
            "start": f"{start_year}0101",
            "end": f"{end_year}1231",
            "latitude": lat,
            "longitude": lon,
            "community": "RE",  # Renewable Energy
            "format": "JSON"
        }

        response = requests.get(self.base_url, params=params)

        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error: {response.status_code}, {response.text}")
            return None

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
