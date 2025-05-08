#!/usr/bin/env python
#!/usr/bin/env python
"""
Energy Forecasting Module:
    - Time series analysis for energy consumption data
    - Predictive modeling incorporating weather variables
    - Evaluation metrics for model performance
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from datetime import datetime, timedelta

class EnergyForecaster:
    """
    Class for forecasting energy consumption using various time series models.
    """
    def __init__(self, dataframe=None, filepath=None):
        """
        Initialize with either a dataframe or a filepath to CSV
        """
        if dataframe is not None:
            self.data = dataframe.copy()
        elif filepath is not None:
            self.data = pd.read_csv(filepath)
            # Convert date column to datetime
            if 'date' in self.data.columns:
                self.data['date'] = pd.to_datetime(self.data['date'])
        else:
            raise ValueError("Either dataframe or filepath must be provided")

        # Store model and scaler for later use
        self.model = None
        self.scaler = StandardScaler()
        self.train_data = None
        self.test_data = None

    def prepare_time_series_features(self):
        """
        Extract time-based features from date column.
        Mathematical basis: Fourier decomposition of seasonal patterns
        """
        # Ensure date is datetime
        if 'date' in self.data.columns:
            # Extract time components
            self.data['month'] = self.data['date'].dt.month
            self.data['day_of_week'] = self.data['date'].dt.dayofweek
            self.data['day_of_year'] = self.data['date'].dt.dayofyear

            # Create cyclic features to represent seasonality
            # Using sine and cosine transformations to capture cyclical nature
            # This handles the circularity of time features (Dec-Jan transition)
            self.data['month_sin'] = np.sin(2 * np.pi * self.data['month']/12)
            self.data['month_cos'] = np.cos(2 * np.pi * self.data['month']/12)

            print("Time features added to dataset")
            return self
        else:
            print("No date column found")
            return self

    def analyze_correlations(self):
        """
        Compute correlation between energy consumption and weather variables
        """
        # List of potential predictor variables
        weather_vars = ['temp (°C)', 'humidity%', 'speed (m/s)', 'W/m²']
        available_vars = [var for var in weather_vars if var in self.data.columns]

        if 'value' in self.data.columns and available_vars:
            corr_matrix = self.data[['value'] + available_vars].corr()

            # Plot correlation heatmap
            plt.figure(figsize=(10, 8))
            plt.imshow(corr_matrix, cmap='coolwarm', interpolation='none', aspect='auto')
            plt.colorbar()
            plt.xticks(range(len(corr_matrix)), corr_matrix.columns, rotation=45)
            plt.yticks(range(len(corr_matrix)), corr_matrix.columns)

            # Add correlation values to the heatmap
            for i in range(len(corr_matrix)):
                for j in range(len(corr_matrix)):
                    plt.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                             ha='center', va='center',
                             color='white' if abs(corr_matrix.iloc[i, j]) > 0.5 else 'black')

            plt.title('Correlation Matrix: Energy vs Weather Variables')
            plt.tight_layout()
            plt.savefig('correlation_heatmap.png')
            plt.show()

            # Print insights about strongest correlations
            for var in available_vars:
                corr = corr_matrix.loc['value', var]
                print(f"Correlation between energy consumption and {var}: {corr:.3f}")

            return corr_matrix
        else:
            print("Required columns not found")
            return None

    def perform_seasonal_decomposition(self, column='value', county=None):
        """
        Decompose time series into trend, seasonal, and residual components
        """
        if county:
            # Filter data for specific county
            county_data = self.data[self.data['county_name'] == county].copy()
        else:
            county_data = self.data.copy()

        # Set date as index for time series analysis
        county_data = county_data.sort_values('date')
        county_data = county_data.set_index('date')

        # Check if we have enough data points
        if len(county_data) < 12:
            print("Not enough data points for seasonal decomposition")
            return None

        try:
            # Perform decomposition (additive model)
            decomposition = seasonal_decompose(county_data[column], model='additive', period=12)

            # Plot components
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(14, 12))

            decomposition.observed.plot(ax=ax1)
            ax1.set_title('Observed')

            decomposition.trend.plot(ax=ax2)
            ax2.set_title('Trend')

            decomposition.seasonal.plot(ax=ax3)
            ax3.set_title('Seasonal')

            decomposition.resid.plot(ax=ax4)
            ax4.set_title('Residual')

            plt.tight_layout()
            plt.savefig(f'{column}_seasonal_decomposition_{county or "all"}.png')
            plt.show()

            return decomposition

        except Exception as e:
            print(f"Error in seasonal decomposition: {str(e)}")
            return None

    def train_model(self, target_col='value', features=None, county=None, test_size=0.2):
        """
        Train a Random Forest model for energy consumption prediction
        """
        if county:
            # Filter data for specific county
            model_data = self.data[self.data['county_name'] == county].copy()
        else:
            model_data = self.data.copy()

        # Drop any rows with NaN in target column
        model_data = model_data.dropna(subset=[target_col])

        if features is None:
            # Default feature set: weather variables and time components
            potential_features = ['temp (°C)', 'humidity%', 'speed (m/s)', 'W/m²',
                                 'month', 'day_of_year', 'month_sin', 'month_cos']
            features = [f for f in potential_features if f in model_data.columns]

        print(f"Using features: {features}")

        # Check if we have enough data
        if len(model_data) < 10:
            print("Not enough data for modeling")
            return False

        # Prepare the data
        X = model_data[features]
        y = model_data[target_col]

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=42
        )

        # Store feature names for later use
        self.feature_names = features

        # Train Random Forest model
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)

        # Evaluate model
        y_pred = self.model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"Model Evaluation:")
        print(f"Mean Squared Error: {mse:.2f}")
        print(f"R² Score: {r2:.2f}")

        # Store test data for later analysis
        self.test_data = pd.DataFrame({
            'Actual': y_test,
            'Predicted': y_pred
        })

        # Feature importance
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1]

        plt.figure(figsize=(10, 6))
        plt.title('Feature Importance')
        plt.bar(range(X.shape[1]), importances[indices])
        plt.xticks(range(X.shape[1]), [features[i] for i in indices], rotation=90)
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        plt.show()

        return True

    def forecast(self, periods=12, county=None):
        """
        Generate future forecasts based on the trained model
        """
        if self.model is None:
            print("Model has not been trained yet")
            return None

        # Get the last date in our dataset
        last_date = self.data['date'].max()

        # Create future dates
        future_dates = [last_date + timedelta(days=30*i) for i in range(1, periods+1)]

        # Create a dataframe for future features
        future_df = pd.DataFrame({'date': future_dates})

        # Extract time features
        future_df['month'] = future_df['date'].dt.month
        future_df['day_of_year'] = future_df['date'].dt.dayofyear
        future_df['month_sin'] = np.sin(2 * np.pi * future_df['month']/12)
        future_df['month_cos'] = np.cos(2 * np.pi * future_df['month']/12)

        # For weather features, use historical averages for the same month
        if county:
            hist_data = self.data[self.data['county_name'] == county].copy()
        else:
            hist_data = self.data.copy()

        # For each month in future_df, get average weather values from historical data
        for feature in [f for f in self.feature_names if f not in ['month', 'day_of_year', 'month_sin', 'month_cos']]:
            if feature in hist_data.columns:
                # Group by month and get mean
                monthly_avg = hist_data.groupby(hist_data['date'].dt.month)[feature].mean()

                # Map these averages to future months
                future_df[feature] = future_df['month'].map(monthly_avg)

        # Prepare features for prediction
        future_features = future_df[self.feature_names]

        # Scale features
        future_features_scaled = self.scaler.transform(future_features)

        # Make predictions
        predictions = self.model.predict(future_features_scaled)

        # Add predictions to future dataframe
        future_df['predicted_value'] = predictions

        # Plot historical + forecasted values
        plt.figure(figsize=(14, 7))

        # Plot historical data
        if county:
            historical = self.data[self.data['county_name'] == county]
        else:
            historical = self.data

        plt.plot(historical['date'], historical['value'], 'b-', label='Historical Data')

        # Plot forecast
        plt.plot(future_df['date'], future_df['predicted_value'], 'r--', label='Forecast')

        plt.title(f'Energy Consumption Forecast for {county or "All Counties"}')
        plt.xlabel('Date')
        plt.ylabel('Energy Consumption')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'forecast_{county or "all"}.png')
        plt.show()

        return future_df

