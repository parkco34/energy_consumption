#!/usr/bin/env python
"""
Oracle Project
"""
from data_utils.data_cleaning import DataCleaning
import pandas as pd
i# Tk Provides access to the Tcl interpreter.
# Each widget attached to same instance of Tk has same value for its tk attribute.mport os
import tkinter as tk
# Tk reads/interprets profile files into the interpreter, where the path for the profile is the HOME environment variable, else: os.curdir.

from tkinter import filedialog, Tk, Label, Button, PhotoImage


def get_file(to_path, width, height):
    """
    Creates a GUI for the file selection, copies the selected file to a target
    directory, and returns the new file path.
    """

    def browse_files():
        """
        Opens file dialog window with specified file type filters, 
        attaches the selected filename to itself as an attribute,
        updates the GUI label with the selected filename,
        copies file to destination path, 
        and returns the selected filename.
        """
        # Creating file/directory selection windows
        browse_files.filename = \
        filedialog.askopenfilename(intialdir=os.getcwd(),
            title="Select a Motha Fuckin File",
            # Add functionality for various file types
            filetypes=(("CSV files", "*.csv"))
        )


    # 
    window = Tk()
    window.title("MothaFuckin File Explorer")
    # String in the form widthXheight
    window.geometry(str(width) + "X" + str(height))




    


    

