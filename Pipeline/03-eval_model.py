import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, explained_variance_score, median_absolute_error

# Load your data
df = pd.read_csv('../Data/processed_cities_dataset.csv', delimiter=',')

# df = df[0:3000] Option for only taking sample

# Define features X and targets y
X = df.drop(columns=['latitude', 'longitude', 'Name', 'Country name EN'])  # Drop target and original name columns from features
y_lat = df['latitude']
y_lon = df['longitude']

# Split the data into training and testing sets
X_train, X_test, y_lat_train, y_lat_test, y_lon_train, y_lon_test = train_test_split(X, y_lat, y_lon, test_size=0.2, random_state=42)

# Below parameters obtained in 02-hyperparameter_tuning.py
rf_lat = RandomForestRegressor(n_estimators=86, max_features=None, max_depth=20, min_samples_split=10, min_samples_leaf=1)
rf_lon = RandomForestRegressor(n_estimators=159, max_features='log2', max_depth=None, min_samples_split=2, min_samples_leaf=1)

rf_lat.fit(X_train, y_lat_train)
rf_lon.fit(X_train, y_lon_train)

# Evaluate the best models using R^2 score
r2_score_lat = rf_lat.score(X_test, y_lat_test)
r2_score_lon = rf_lon.score(X_test, y_lon_test)
print(f"R^2 Score for Latitude: {r2_score_lat}")
print(f"R^2 Score for Longitude: {r2_score_lon}")

# Create a True/False table for measuring accuracy
lat_predictions = rf_lon.predict(X_test)
lon_predictions = rf_lon.predict(X_test)

lat_accuracy = np.abs(lat_predictions - y_lat_test) < 0.1 # Where tolerance = 0.1
lon_accuracy = np.abs(lon_predictions - y_lon_test) < 0.1 

accuracy_table = pd.DataFrame({
    'Actual Latitude': y_lat_test,
    'Predicted Latitude': lat_predictions,
    'Latitude Accuracy': lat_accuracy,
    'Actual Longitude': y_lon_test,
    'Predicted Longitude': lon_predictions,
    'Longitude Accuracy': lon_accuracy
})

print(accuracy_table)


# -------------------------------------------------------- Measuring Model Performance

# Assuming y_test and y_pred are your actual and predicted values respectively

y_pred = rf_lon.predict(X_test)

# Mean Absolute Error (MAE)
mae = mean_absolute_error(y_lon_test, y_pred)

# Mean Squared Error (MSE)
mse = mean_squared_error(y_lon_test, y_pred)

# Root Mean Squared Error (RMSE)
rmse = np.sqrt(mse)

# R-squared
r2 = r2_score(y_lon_test, y_pred)

# Explained Variance Score
explained_variance = explained_variance_score(y_lon_test, y_pred)

# Median Absolute Error
median_ae = median_absolute_error(y_lon_test, y_pred)

print(f"MAE: {mae}")
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"R-squared: {r2}")
print(f"Explained Variance: {explained_variance}")
print(f"Median Absolute Error: {median_ae}")