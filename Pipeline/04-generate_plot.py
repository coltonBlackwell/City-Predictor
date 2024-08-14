import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import plotly.graph_objects as go
from geopy.distance import geodesic
from sklearn.model_selection import train_test_split

# Load your data
df = pd.read_csv('../Data/processed_cities_dataset.csv', delimiter=',')

# df = df[0:5000]

# Create mappings for 'Name' and 'Country name EN'
name_mapping = dict(enumerate(pd.Categorical(df['Name']).categories))
country_mapping = dict(enumerate(pd.Categorical(df['Country name EN']).categories))

# Features and targets
city_names = df['Name']
country_codes = df['Country_code']

# Define features X and targets y
X = df.drop(columns=['latitude', 'longitude', 'Name', 'Country name EN'])  # Drop target and original name columns from features
y_lat = df['latitude']
y_lon = df['longitude']

# Split the data into training and testing sets
X_train, X_test, y_lat_train, y_lat_test, y_lon_train, y_lon_test = train_test_split(X, y_lat, y_lon, test_size=0.2, random_state=42)

# Initialize and train Random Forest Regressors
rf_lat = RandomForestRegressor(n_estimators=86, max_features=None, max_depth=20, min_samples_split=10, min_samples_leaf=1)
rf_lon = RandomForestRegressor(n_estimators=159, max_features='log2', max_depth=None, min_samples_split=2, min_samples_leaf=1)

rf_lat.fit(X_train, y_lat_train)
rf_lon.fit(X_train, y_lon_train)

# Get the number of points to predict from the user
num_points = int(input("Enter the number of predicted data points you want on the map: "))

# Use dynamic seed or omit it for randomness
np.random.seed()  # Initialize the numpy random number generator
selected_points = X_test.sample(n=num_points)

# Make predictions for the chosen points
predicted_lats = rf_lat.predict(selected_points)
predicted_lons = rf_lon.predict(selected_points)

# Get the actual latitudes and longitudes for these points
selected_points_indices = selected_points.index
actual_lats = y_lat_test.loc[selected_points_indices]
actual_lons = y_lon_test.loc[selected_points_indices]

# Calculate distances
distances_km = [geodesic((actual_lats.iloc[i], actual_lons.iloc[i]), (predicted_lats[i], predicted_lons[i])).km for i in range(num_points)]

# Get the actual city names and country names using the indices from the test set
actual_city_names = [city_names.iloc[idx] for idx in selected_points_indices]
actual_country_codes = [country_codes.iloc[idx] for idx in selected_points_indices]

# Handle missing mappings
actual_country_names = []
for code in actual_country_codes:
    if code in country_mapping:
        actual_country_names.append(country_mapping[code])
    else:
        actual_country_names.append("Unknown Country")

# Create custom hover text
hover_text = [f"Actual City: {actual_city_names[i]}<br>Actual Country: {actual_country_names[i]}<br>Distance: {distances_km[i]:.2f} km"
              for i in range(num_points)]



# ----------------------------------------------------------------------------------- (CREATING 3D MAP)

# Create a 3D globe with Plotly
fig = go.Figure()

# Add actual points
fig.add_trace(go.Scattergeo(
    lon=actual_lons,
    lat=actual_lats,
    mode='markers+text',
    text=actual_city_names,
    textposition='top center',
    marker=dict(size=8, color='red'),
    name='Actual',
    hovertext=hover_text
))

# Add predicted points
fig.add_trace(go.Scattergeo(
    lon=predicted_lons,
    lat=predicted_lats,
    mode='markers+text',
    textposition='top center',
    marker=dict(size=8, color='blue'),
    name='Predicted',
    hovertext=hover_text
))

# Add lines connecting actual and predicted points
for i in range(num_points):
    fig.add_trace(go.Scattergeo(
        lon=[actual_lons.iloc[i], predicted_lons[i]],
        lat=[actual_lats.iloc[i], predicted_lats[i]],
        mode='lines',
        line=dict(width=2, color='black', dash='dot'),
        name=f'Line {i}',  # Line name is optional
        showlegend=False
    ))

# Highlight the best prediction
best_idx = np.argmin(distances_km)
best_predicted_lat = predicted_lats[best_idx]
best_predicted_lon = predicted_lons[best_idx]
fig.add_trace(go.Scattergeo(
    lon=[best_predicted_lon],
    lat=[best_predicted_lat],
    mode='markers+text',
    text=['Best Predicted'],
    textposition='top center',
    marker=dict(size=15, color='orange', symbol='star'),
    name='Best Predicted'
))

fig.update_layout(
    title={
        'text': '3D Globe with Actual Vs. Predicted Cities',
        'y':0.95,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top',
        'font': {
            'size': 24
        }
    },
    legend=dict(
        x=0.8,
        y=0.1,
        traceorder='normal',
        font=dict(
            family='sans-serif',
            size=22,  # Legend text font size
            color='black'
        ),
        bgcolor='white',
        bordercolor='black',
        borderwidth=2
    ),
    geo=dict(
        projection_type='orthographic',
        showland=True,
        landcolor='lightgreen',
        showocean=True,
        oceancolor='aqua',
        showlakes=True,
        lakecolor='lightblue',
        showcountries=True,
        showcoastlines=True,
        showframe=True,
        framecolor='black',  # Black border
        framewidth=2  # Border width
    )
)

fig.show()
