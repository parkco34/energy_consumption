#!/usr/bin/env python
import pandas as pd
import numpy as np


class CleanData:
    """
    Data Wrangling class for handling missing data and categorical encodings.
    """

    def __init__(self, dataframe):
        """
        Establish dataframe and obtain columns and rows
        """
        # Ensure the input is actually a pdf.DataFrame
        if not isinstance(dataframe, pd.DataFrame):
            raise ValueError("""
                             Expected a pandas dataframe, but go some other bullshit """)


        # Instantiate some shit (◕‿◕)╭∩╮
        self.dataframe = dataframe
        self.num_rows = dataframe.shape[0]
        self.num_columns = dataframe.shape[1]

    def column_summary(self, top_N_values):
        """
        Basic check on column datatype, null counts, distinct values, etc.
        Loops thru each column, using a dataframe to store these:
            column name
            column datatype
            number of nulls
            number of non-nulls
            number of distinct values
            min/max values
            median value
            average value (if number)
            number of non-zero values (if number)
            top N distinct values
        -----------------------------------------------------------
        INPUT:
            top_N_values: (int) Top N distinct values for sample

        OUTPUT:
            summary_df: (pd.DataFrame) DataFrame as summary of original dataframe
        """
        # Nested function
        def get_top_N_distinct(arr, top_N_values):
            """
            Calculates the top N distinct values for the array-like input
            --------------------------------------------------
            INPUT:
                arr: (np.ndarray) of (int/float)

            OUTPUT:
                top_N_distinct_values: (np.ndarray) Top N distinc t values
            """
            unique_vals, counts = np.unique(arr, return_counts=True)
            sorted_indices = np.argsort(counts)[::-1]
            top_N_indices = sorted_indices[:top_N_values]

            return unique_vals[top_N_values]
        
        # List to store each column's summary
        summary_list = []

        # Iterate thru each column, accumulating summary information
        for col in self.dataframe.columns:
            column_data = self.dataframe[col]
            
            summary_dict = {
                "column_name": col,
                "column_dtype": column_data.dtype,
                "null_num": column_data.isnull().sum(),
                "non_null_num": column_data.notnull().sum(),
                "distinct_values": column_data.unique(),
                "min_value": column_data.min(),
                "max_value": column_data.max(),
                "median_value": column_data.median(),
                "avg_value": column_data.mean(),
                "non_zero_num": len(column_data[column_data != 0]),
                "top_N_distinct_values": get_top_N_distinct(column_data.unique(), top_N_values)
            }

            # Append dictionary to list
            summary_list.append(summary_dict)

        # Convert dictionary to DataFrame
        summary_df = pd.DataFrame(summary_list)

        return summary_df


        

