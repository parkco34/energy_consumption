#!/usr/bin/env python
"""
# data_utils/__init__.py

# Import the DataCleaning class so it's available when someone imports the package
from .data_cleaning import DataCleaning

# You could define package-level variables
__version__ = '1.0.0'

# You could also run initialization code
print("Initializing data_utils package...")

# Option 1: Import the package
import data_utils
cleaner = data_utils.DataCleaning(df)  # If you imported DataCleaning in __init__.py

# Option 2: Import specific class
from data_utils import DataCleaning
cleaner = DataCleaning(df)

# Option 3: Import from specific module
from data_utils.data_cleaning import DataCleaning
cleaner = DataCleaning(df)
"""


print("Initializing data_utils package")


