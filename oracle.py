#!/usr/bin/env python
"""
Oracle Project
"""
import os
from data_utils.data_cleaning import DataCleaning
from nasa_power_api import NASAPowerAPI as api # Import NASA API Class
import pandas as pd
# Tk Provides access to the Tcl interpreter.
# Each widget attached to same instance of Tk has same value for its tk attribute
import tkinter as tk
# Tk reads/interprets profile files into the interpreter, where the path for the profile is the HOME environment variable, else: os.curdir.
from tkinter import filedialog, Tk, Label, Button, PhotoImage

# Safe for later --------------------------------------------------------
#def hunt_file(width, height, ze_path):
#    """
#    GUI for user to select file via 'File Explorer'
#    ? -> For now, just for macos ?
#    """
#
#    def browse_files():
#        """
#        This function reacts when user presses 'Browse Files'
#        """
#        # filedialog for file/directory selection
#        # askopenfilename - Creates an OPEN dialog and returns selected
#        # filename(s) that correspond to existing file(s).
#        browse_files.filename = filedialog.askopenfilename(
#            # *options
#            initialdir=os.getcwd(), # Current Directory
#            title="Select File",
#            filetypes = ((("all files", "8"),
#                         ("Excel Files", "*.xlsx"),
#                         ("CSV Files", "*.csv"),
#                         ("Text Files", "*.txt")
#                        )))
#
#        # Include the path in the GUI label
#        explorer_label.config(text=f"File Opened: {browser_files.filename}")
#
#        # ! => Immediately duplicates the chosen file into Oracle directory,
#        # ensuring program always reads from controlled workspace, where only
#        # the copy will be used later
#        shutil.copy(browse_files.filename, to_path)
#
#        return browse_files.filename
#
#    # Tkinter window generation
#    window = Tk()
#    window.title("File Explorer")

# Refactoring code --------------------------------------------------

def read_data(filename):
    """
    Reads dataset (csv file)
    ----------------------------------------
    INPUT:
        filename: (str)

    OUTPUT:
        df:(pd.DataFrame)
    """
    # Ensure filename exists in current directory
    if os.path.exists(filename):
        return pd.read_csv(filename)

    else:
        print(f"{filename} Doesn't exist in given path")
        return None

def get_weather_data(coordinates, year_range):
    """
    Calls to NASA POWER API for weather data
    ------------------------------------------------
    INPUT:
        coordinates: (tuple) Latitute/Longitude
        year_range: (tuple) start_year/end_year

    OUTPUT:
        data: (dict) Dictionary of raw weather data
    """
    # Unpack coordinate tuple into latitude/longitude
    lat, lon = coordinates
    # Unpack year_range into start/end dates, if valid
    try:
        # unpacking
        start, end = year_range
   
    except Exception as e:
        print(f"Something went wrong: {e}")



df = read_data("data/raw/Utility_Energy_Registry_Monthly_County_Energy_Use__Beginning_2021_20241208.csv")

# Example usage of GUI
#hunt_file(800, 600, "data/raw/Utility_Energy_Registry_Monthly_County_Energy_Use__Beginning_2021_20241208.csv")
