import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from scipy import stats

# There were only a few of records where the number of fields exceeded the maximum (20) so those were eliminated

df = pd.read_csv('../Data/geonames-all-cities-with-a-population-1000.csv', delimiter=';')

df[['latitude', 'longitude']] = df['Coordinates'].str.split(',', expand=True).astype(float)

numeric_columns = ['Population', 'Elevation', 'latitude', 'longitude']
df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')

# Remove rows with NaN values in numeric columns
df = df.dropna(subset=numeric_columns)

# ----------------------------------------------------------- (Statistical Method - Z-Score)

z_scores = np.abs(stats.zscore(df[numeric_columns]))
outliers_z_score = (z_scores > 3).any(axis=1)
print("Potential outliers (Z-Score):")
print(df[outliers_z_score])

# ----------------------------------------------------------- (Statistical Method - IQR (Interquartile Range))
def find_outliers_iqr(column):
    Q1 = column.quantile(0.25)
    Q3 = column.quantile(0.75)
    IQR = Q3 - Q1
    return column[(column < (Q1 - 1.5 * IQR)) | (column > (Q3 + 1.5 * IQR))]

print("\nPotential outliers (IQR) in each numeric column:")
for column in numeric_columns:
    outliers_iqr = find_outliers_iqr(df[column])
    print(f"{column}:")
    print(outliers_iqr)

# ----------------------------------------------------------- (Visualization - Box Plots)
plt.figure(figsize=(12, 8))
for i, column in enumerate(numeric_columns):
    plt.subplot(2, 2, i + 1)
    plt.boxplot(df[column])
    plt.title(f'Box Plot of {column}')
    plt.ylabel(column)
plt.tight_layout()
plt.show()

# ----------------------------------------------------------- (Visualization - Scatter Plot (Latitude vs Longitude))

gdf = gpd.GeoDataFrame(
    df, geometry=gpd.points_from_xy(df.longitude, df.latitude))

world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

fig, ax = plt.subplots(figsize=(10, 7))
world.plot(ax=ax, color='lightgray')
gdf.plot(ax=ax, color='red', markersize=5, alpha=0.7)

plt.title('City Coordinates on Global Map')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()