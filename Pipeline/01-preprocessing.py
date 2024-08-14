import pandas as pd
import numpy as np
import os 
from scipy import stats

# there were only a couple lines where the number of fields exceeded the maximum (20) so those were eliminated

df = pd.read_csv("../Data/geonames-all-cities-with-a-population-1000.csv", delimiter=';')

# Extract latitude and longitude from the last column
df[['latitude', 'longitude']] = df['Coordinates'].str.split(',', expand=True).astype(float)

#ensure columns are of numericcal type 
df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')


# Extract numeric columns for analysis
numeric_columns = ['Population', 'Elevation', 'latitude', 'longitude']
df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')

# ----------------------------------------------------------------------------------- (DATA CLEANING)

# - Remove the Geoname ID and Alternate Names columns 

df = df.drop(['Geoname ID', 'Alternate Names', 'Admin1 Code', 'Admin2 Code', 'Admin3 Code', 'Modification date'], axis=1)

# # Remove rows with NaN values in numeric columns
# df = df.dropna(subset=numeric_columns)


# - Remove row if either lat or long is missing
location_data = df.dropna(subset=['latitude', 'longitude'], how='any')


df = df.fillna({
    'Population': -1,
    'Elevation': -1,
})


# Check for missing values
missing_values = df.isnull().sum()
print("Missing values per column:")
print(missing_values)

#As we can see both Country Code 2 and Admin4 Code have more than 120000 values missing each! Lets drop these...

df = df.drop(['Admin4 Code', 'Country Code 2'], axis=1)

#----------- (REMOVING OUTLIERS)-----------

z_scores = np.abs(stats.zscore(df[numeric_columns]))
outliers_z_score = (z_scores > 3.5).any(axis=1)
print("Potential outliers (Z-Score):")
print(df[outliers_z_score])

# - Remove unlikely Long/Lat from dataset

df = df[(df['latitude'] >= -90) & (df['latitude'] < 90)]
df = df[(df['longitude'] >= -180) & (df['longitude'] < 180)]

df = df.drop(['Coordinates'], axis=1)

df.to_csv('../Data/processed_cities_dataset.csv', index=False)

# ----------------------------------------------------------------------------------- (TRANSFORMING CATEGORIAL ATTRIBUTES)

# Convert 'Name' and 'Country name EN' to categorical codes and create mappings
df['Name_code'] = pd.Categorical(df['Name']).codes
df['Country_code'] = pd.Categorical(df['Country name EN']).codes

# Convert other categorical columns to numerical codes
categorical_columns = df.select_dtypes(include=['object']).columns
for col in categorical_columns:
    if col not in ['Name', 'Country name EN']:
        df[col] = pd.Categorical(df[col]).codes


df.to_csv('../Data/processed_cities_dataset.csv', index=False)