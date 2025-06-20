#!/usr/bin/env python
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


class DataCleaning(object):
    """
    Data Wrangling Class for handling missing data and categorical encodings.
    """
    # Class attributes go here ...

    def __init__(self, dataframe):
        """
        Takes dataframe and conducts input-validation, establishing the
        rows/columns and dataframe itself as instance attributes
        """
        self.dataframe = dataframe
        self.rows = self.dataframe.shape[0]
        self.columns = self.dataframe.shape[1]

    def column_summary(self, N=10, pretty_output=True):
        """
        Outputs summary of columns including information such as:
            column_name
            column_dtype
            num_of_nulls
            num_of_non_nulls
            num_ofdistinct_vals
            min/max vals
            average value (if number)
            num_of_non_zero_vals
            top_N_distinct_vals
        ------------------------------------------------------
        INPUT:
            N: (int; default=10) Number of distinct values in dataframe
            pretty_output: (bool; default=True) If so, output the dataframe summary in a 'pretty' format

        OUTPUT:
            summary_df: (pd.DataFrame) Dataframe as summary of riingal
            dataframe
        """
        # ------------------------------------------------------------------------------------------------------
        # Higher order function for a prettier output of dataframe summary, if
        # preferred
        def pretty_column_summary(
            summary_df,
            *,
            total_rows,
            max_null=.2,
            style=True
        ):
            """
            Outputs a pretty version of the dataframe summary for data wrangling, 
            """
            # Create copy of daraframe
            df = summary_df.copy()

            # Derived metrics
            df["null_pct"] = (df.null_num / total_rows)
            df["distinct_pct"] = (df.num_distinct_vals / (total_rows - df.null_num))
            # Alert user when null % is greater than the maximum number of null
            # values OR when % of disttinct values is less than some tolerance
            # (0.1 here) using
            # where(condition, [x, y, ]/), where it returns elements chosen from x or y depending on CONDITION
            df["alert"] = np.where((df.null_pct > max_null) |
                                   (df.distinct_pct < .1), "⚠", "")
            
            # Ordering by 'alerts' first, then dtype
            df = (df.sort_values(["alert", "column_dtype", "null_pct"], ascending=[False, True, False]))

            # Check if 'style' is active or not, returning the current
            # summary dataframe
            if not style:
                return df

            # Stylistic
            return (df.style.background_gradient(
                subset=["null_pct"], cmap="Reds").bar(subset=['num_distinct_vals'],
                                                      align="mid",
                                                      color="#5fba7d").format({"null_pct": "{:.1%}", "distinct_pct": "{:.1%}"}))
        # ------------------------------------------------------------------------------------------------------
        # Initialize summary dataframe
        summary_rows = []

        # Iterating thru dataframe columns, getting appropriate metrics for data wrangling
        for col in self.dataframe.columns:
            column_name = col
            column_dtype = self.dataframe[col].dtype
            null_num = self.dataframe[col].isnull().sum()
            non_null_num = self.dataframe[col].notnull().sum()
            # Parentheses avoids the UnboundLocalError: 'refernced before assignmenet'
            (min_val, max_val, median_val, avg_val, non_zero_num,
            top_N_distinct) = None, None, None, None, None, None

            # Ensure numerical values in column using
            # 'pd..api.types.is_numeric_dtype(arr_or_dtype)'
            if pd.api.types.is_numeric_dtype(self.dataframe[col]):
                min_val = self.dataframe[col].min()
                max_val = self.dataframe[col].max()
                median_val = self.dataframe[col].median()
                avg_val = self.dataframe[col].mean()
                non_zero_num = (self.dataframe[col] != 0).sum()

            # Calculate TOP N unique, non-na values (counts)
            distinct_vals = self.dataframe[col].dropna().value_counts()
            # Number of distinct values
            num_distinct_vals = len(distinct_vals)
            # Top N unique values using .head()
            top_N_unique = distinct_vals.head(N).index.tolist()

            # Append dictionary to summary rows
            # List of dictionaries represnting the properties of dataframe for data wrangling
            summary_rows.append({
                "column_name": column_name,
                "column_dtype": column_dtype,
                "null_num": null_num,
                "non_null_num": non_null_num,
                "min_val": min_val,
                "max_val": max_val,
                "median_val": median_val,
                "avg_val": avg_val,
                "distinct_vals": distinct_vals,
                "num_distinct_vals": num_distinct_vals,
                "non_zero_num": non_zero_num,
                "top_N_unique": top_N_unique
            })

        # Convert Dictionary ro a  Summary dataframe
        summary_df = pd.DataFrame(summary_rows)

        # Pretty output
        if pretty_output:
            pretty_column_summary(summary_df, summary_shape[0],
                                  max_null=int(df.null_num.max()))

    def replace_negative_values(self, column):
        """
        Replaces negative values if negative values do not make sense for a particular feature.
            - This should be for only columns in which negative values make no
            ( ͡° ͜ʖ ͡°  ) sense.
        ------------------------------------------------------------
        INPUT:
            column: (str) Column which should not have negative values

        OUTPUT:
            None
        """
        pass

    def drop_column_missing_data(self, threshold=0.5):
        """

        """
        pass
        # 


