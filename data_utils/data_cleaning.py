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
        =========================================================
        (MCAR) Missing Completely @ Random:
            No relationship between the missing values and any other values in the datset.
            The probability data missing is the same for all observations.

        (MAR) Mssing @ Random:
            Likelihood of a value missing in the dataset could possibly due to other variables in dataset.
            Missing values do not occur @ random but the pattern could be explained by other observations.

        (MNAR) Missing Not @ Random:
            Missing valus not random and cannoot be explained by the observed data.
            Missing values could possibly be related to unobserved data.
        =========================================================
        """
        self.dataframe = dataframe
        self.rows = self.dataframe.shape[0]
        self.columns = self.dataframe.shape[1]

    def column_summary(self, N=10):
        """
        ? -> Fix this so it outputs something more readable, involving
        perentages so the user can make sense of it quickly!!!

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

        OUTPUT:
            summary_df: (pd.DataFrame) Dataframe as summary of riingal
            dataframe
        """
        # ------------------------------------------------------------------------------------------------------
        # Initialize summary dataframe
        summary_rows = []
        total_rows = len(self.dataframe)
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
                # Summing the True values
                non_zero_num = (self.dataframe[col] != 0).sum()

            # Calculate TOP N unique, non-na values (counts)
            distinct_vals = self.dataframe[col].dropna().value_counts()
            # Number of distinct values
            num_distinct_vals = len(distinct_vals)
            # Top N unique values using .head()
            top_N_unique = distinct_vals.head(N).index.tolist()

            # % of NULL values
            null_pct = round(null_num / self.rows * 100, 2)

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
                "top_N_unique": top_N_unique,
                "null_pct": null_pct
            })

        # Convert Dictionary ro a  Summary dataframe
        summary_df = pd.DataFrame(summary_rows)

        return summary_df

    def drop_missing_columns(self, threshold=0.5):
        """
        Drops columns where proportion of missing data exceeds threhsold.
        -------------------------------------------------
        INPUT:
            threshold: (float) Proportion of missing data required to drop.

        OUTPUT:
            self: Method chaining
        """
        # Find column to drop
        drop_columns = self.dataframe.columns[self.dataframe.isnull().mean() > threshold].tolist()

        # OUtput to user
        if drop_columns:
            print(f"Dropping columns: {drop_columns}")

        else:
            print("No columns to drop - BELOW THRESHOLD")

        return self # Method chaining

    def drop_rows_missing_data(self):
        """
        Drop rows with any missing data, where there's at least one NULL value
        - This isn't best practice, since it might delete valueable
        information.
        - pd.api.types.is_numeric_dtype is a function in the
        Pandas library that is used to check if a given array
        or dtype is of a numeric type.
        ======================================================
        Removes rows that have *any* NaN after all imputations.
        ======================================================
        ----------------------------------------
        INPUT:
            None

        OUTPUT:
            self: method chaining
        """
        # Remove  relevant rows
        self.dataframe.dropna(axis=0, how="any", inplace=True)

        return self

    def imputing_vals_mean(self, column):
        """
        Imputes missing values in a numerical colum with the arithmetic mean.

        Mathematical definition:
        For a vector x = [x₁, ..., xₙ] with missing values, let:
        - I = {i | xᵢ is not missing}
        - μ = (1/|I|) * ∑ᵢ∈I xᵢ
        Then for each missing value j: xⱼ ← μ

        Properties:
        - Preserves the sample mean: E[X] remains unchanged
        - Reduces variance: Var(X_imputed) ≤ Var(X_original)
        - Maintains sample size: n_imputed = n_original
        -------------------------------------------------------
        INPUT:
            column: (str) Name of column to impute

        OUTPUT:
            self: (pd.DataFrame) Method chaining
        """
        # Validate column existence
        if column not in self.dataframe.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame")

        # Validate column type
        if not pd.api.types.is_numeric_dtype(self.dataframe[column]):
            raise ValueError(f"""Column '{column}' must be numeric, got
                             {self.dataframe[column].dtype}""")

        # Create a copy to ensure we're working with original data
        df_copy = self.dataframe.copy()

        # Calculate mean of non-missing data
        column_mean = df_copy[column].mean()

        # Impute missing values
        df_copy[column] = df_copy[column].fillna(column_mean)

        # Copy of dataframe so as not to change the original
        self.dataframe = df_copy

        return self

    def imputing_vals_median(self, column):
        """
        Imputes missing values in a numerical column with the Median.
        - Cannot work on Categorical data
        --------------------------------------------------------
        INPUT:
            column: (str) Column name to impute

        OUTPUT:
        """
        if (column in self.dataframe.columns and
            pd.api.is_numeric_dtype(self.dataframe[column])) :
            self.dataframe[column].fillna(self.dataframe[column].median(),
                                          inplace=True)

        return self

    def time_interpolate(self, limit=3, order=1):
        """
        Linearly interpolate numeric gaps to 'limit' consecutive NaNs.
        --------------------------------------------------------
        INPUT:

        OUTPUT:
        """
        # Get numeric columns only 
        num_cols = self.dataframe.select_dtypes(include="number").columns

        # Interpolation
        self.dataframe[num_cols] = \
        self.dataframe[num_cols].interpolate(method="linear", limit=limit,
                                             limit_direction="both", # ?
                                             order=order)

        return self


    def imputing_group(self, group_via_col, target_col, average=True):
        """
        Imputes missing values in a target column by group
        --------------------------------------------------------
        INPUT:
            group_via_col: (str) Column to group by
            target_col: (str) Column with missing values
            average: (str) "Mean" or "Median"

        OUTPUT:

        """
        # Validate existence of group_via_col
        if group_via_col not in self.dataframe.columns:
            raise ValueError(
                f"""Grouping column '{group_via_col}' not found in dataframe"""
                            )

        # Ensure target column exists and is numerical
        if (target_col in self.dataframe.columns and
            pd.api.is_numeric_dtype(self.dataframe[target_col])):

            # Determine whether group filled in via mean or median
            if average:
                # Mean
                self.dataframe[target_col] = \
                self.dataframe[target_col].fillna(self.dataframe.groupby(group_via_col)[target_col].transform("mean"))

            else:
                # Median
                self.dataframe[target_col] = \
                self.dataframe[target_col].fillna(self.dataframe.groupby(group_via_col)[target_col].transform("median"))

            return self

    def imputing_categorical_cols(self, column):
        """
        Imputes missing values in a categorical column with the MODE (most
        frequent value).
        ------------------------------------------------
        INPUT:
            column: (str) Column name to impute.

        OUTPUT:
        """
        if (column in self.dataframe.columns and not
            pd.api.types.is_numeric_dtype(self.dataframe[column])):
            # Most frequent value
            mode_value = self.dataframe[column].mode()[0]

            self.dataframe[column].fillna(mode_value, inplace=True)

        return self

    def forward_fill(self):
        """
        Method to fill missing values by carrying forward the last observed
        non-missing value (ffil).
        Useful for;
            - Time series data
        -----------------------------------------------------
        INPUT:

        OUTPUT:
        """
        self.dataframe = self.dataframe.fillna(method="ffill")

        return self
        pass

    def backward_fill(self, threshold=0.5):
        """
        Method for filling in missing values via carrying backward the next
        observed non-missing value.
        --------------------------------------------------
        INPUT:

        OUTPUT:
        """
        self.dataframe = self.dataframe.fillna(method="bfill")

        return self
        pass
        #

    def reset_dataframe(self, original_dataframe):
        """
        Resets dataframe to its original state.
        """
        self.dataframe = original_dataframe.copy()

        return self

    def eda(self):
        """
        Prints dataset overview including:
            - Shape, missing values, column types
            - Summary statistics
        """
        print(f"Dataset Shape: {self.dataframe.shape}")
        print("\nColumn Data Types:")
        print(self.dataframe.dtypes)

        print("\nMissing Values Per Column:")
        print(self.dataframe.isnull().sum())

        print("\nSummary Statistics:")
        print(self.dataframe.describe())

    def detect_outliers(self, column):
        """
        Identifies potential outliers using the IQR method
        ------------------------------------------------
        INPUT:
            column: (str) Column name

        OUTPUT:
            outliers: ()
        """
        q1 = self.dataframe[column].quantile(0.25)
        q3 = self.dataframe[column].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outliers = self.dataframe[(self.dataframe[column] < lower_bound) | (self.dataframe[column] > upper_bound)]

        return outliers

    def plot_time_series(self, date_col, value_col):
        """
        Plots time series of energy consumption
        """
        plt.figure(figsize=(12, 5))
        plt.plot(self.dataframe[date_col], self.dataframe[value_col],
                 marker="o", linestyle="-")
        plt.xlabel("Date")
        plt.ylabel("Energy Consumption")
        plt.title("Monthly Energy Consumption Over Time")
        plt.xticks(rotation=45)
        plt.show()


if __name__ == "__main__":
    data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
    'Age': np.random.randint(20, 60, size=5),
    'City': ['New York', 'London', 'Paris', 'Tokyo', 'Sydney'],
    'Score': np.random.uniform(70, 100, size=5).round(2)
}

    # Create the DataFrame
    mock_df = pd.DataFrame(data)
    dc = DataCleaning(mock_df)
    dc.drop_missing_columns()

