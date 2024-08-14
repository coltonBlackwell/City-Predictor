import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from scipy.stats import randint

df = pd.read_csv('../Data/processed_cities_dataset.csv', delimiter=',')

df = df[0:10000] # Using smaller sample size

X = df.drop(columns=['latitude', 'longitude', 'Name', 'Country name EN'])  # Drop target and original name columns from features
y_lat = df['latitude']
y_lon = df['longitude']

X_train, X_test, y_lat_train, y_lat_test, y_lon_train, y_lon_test = train_test_split(X, y_lat, y_lon, test_size=0.2, random_state=42)

rf_lat = RandomForestRegressor()
rf_lon = RandomForestRegressor()

rf_lat.fit(X_train, y_lat_train)
rf_lon.fit(X_train, y_lon_train)


param_dist = {
    'n_estimators': randint(50, 201),
    'max_features': ['sqrt', 'log2', None],
    'max_depth': [None, 10, 20, 30, 40],
    'min_samples_split': randint(2, 11), 
    'min_samples_leaf': randint(1, 5)    
}

# Use RandomizedSearchCV to find the best parameters for latitude model
random_search_lat = RandomizedSearchCV(estimator=rf_lat, param_distributions=param_dist, n_iter=40, cv=5, n_jobs=-1, verbose=2)
random_search_lat.fit(X_train, y_lat_train)
best_rf_lat = random_search_lat.best_estimator_

# Use RandomizedSearchCV to find the best parameters for longitude model
random_search_lon = RandomizedSearchCV(estimator=rf_lon, param_distributions=param_dist, n_iter=40, cv=5, n_jobs=-1, verbose=2)
random_search_lon.fit(X_train, y_lon_train)
best_rf_lon = random_search_lon.best_estimator_


print("Best parameters for Latitude model:")
print(random_search_lat.best_params_)
print("Best parameters for Longitude model:")
print(random_search_lon.best_params_)

'''Following results when ran: 

Best parameters for Latitude model:
{'max_depth': 20, 'max_features': None, 'min_samples_leaf': 1, 'min_samples_split': 10, 'n_estimators': 86}

Best parameters for Longitude model:
{'max_depth': None, 'max_features': 'log2', 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 159}
'''