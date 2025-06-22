#!/usr/bin/env python
import numpy as np
import pandas as pd

# ---------- ENERGY ----------
energy = energy_df.copy()

# 2.1  Replace sentinel -999 with proper NaN
energy.replace(-999, np.nan, inplace=True)

# 2.2  Build a monthly timestamp ---------------------------
# (adapt if you actually have daily data elsewhere)
energy["month"] = pd.PeriodIndex(year=energy["year"], period="M").to_timestamp("M")
# If you *do* have 'month' or 'day' columns, combine them instead:
# energy["month"] = pd.to_datetime(dict(year=energy["year"],
#                                       month=energy["month"],
#                                       day=1))

# 2.3  Optionally aggregate duplicates created by utilities
energy = (energy
          .groupby(["month", "county_name"], as_index=False)
          .agg({"value": "sum",              # pick what makes sense
                "number_of_accounts": "sum"}))

# ---------- WEATHER ----------
weather = weather_df.copy()

# The index is already YYYY-MM-01; move it into a column that
# matches the energy frame's name ('month') for a normal merge.
weather["month"] = weather.index.to_period("M").to_timestamp("M")
weather.reset_index(drop=True, inplace=True)

combined = (energy
            .merge(weather, on="month", how="left",
                   validate="m:1"))  # many energy rows -> 1 weather row

combined = (energy
            .merge(weather, on=["month", "county_name"],
                   how="left",
                   validate="m:1"))

combined = pd.merge_asof(
    energy.sort_values("date"),          # finer-grain DF
    weather.sort_values("month"),        # coarser-grain DF
    left_on="date",
    right_on="month",
    direction="backward"                 # use last completed month
)

from data_cleaning import DataCleaning

group_specs = [
    {"group_col": "county_name", "target_col": "value", "method": "mean"}
]

cleaner = DataCleaning(combined)
clean_df = (cleaner
            .auto_clean(col_thresh=0.6,
                        time_interp_first=True,
                        group_specs=group_specs)
            .dataframe)

print(clean_df.isna().sum().sort_values(ascending=False).head(10))



