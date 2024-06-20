#!/usr/bin/env python
# coding: utf-8

# In[6]:


import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import geopandas as gpd
import xarray as xr
import pandas as pd
import numpy as np
import regionmask
import plotly.graph_objs as go
import json

# Create the Dash app
app = dash.Dash(__name__)
server=app.server
india_geojson = 'gadm41_IND_1.json'
india_gdf = gpd.read_file(india_geojson)
states_to_remove = ['DadraandNagarHaveli', 'Lakshadweep', 'DamanandDiu', 'Puducherry']
india_gdf_f = india_gdf[~india_gdf['NAME_1'].isin(states_to_remove)]
unique_states = india_gdf_f.drop_duplicates(subset=['NAME_1'])

# Convert the GeoDataFrame to GeoJSON format
india_geojson_f = json.loads(india_gdf_f.to_json())

# Load datasets
ds_tmean_ref = xr.open_dataset('precomputed_mean.nc')
ds_tmean = xr.open_dataset('CPC_tmean_18june2024.nc')
per_90 = xr.open_dataset('precomputed_percentiles.nc')

# Calculate the number of days exceeding the 99th percentile for each state
def calculate_extreme_days(state):
    st = unique_states.loc[unique_states['NAME_1'] == state, :]
    mask = regionmask.mask_3D_geopandas(st, ds_tmean['lon'], ds_tmean['lat'])
    tmean_mask = ds_tmean.where(mask).mean(['lat', 'lon'])
    tmean_clim_mask = ds_tmean_ref.sel(state=state)
    tmean_mask = tmean_mask.sel(time=slice('2023-06-16', '2024-06-15')).squeeze()
    percentile_90 = per_90.sel(state=state).__xarray_dataarray_variable__

    start_date = np.datetime64('2023-06-16')
    day_of_year_start = (start_date - np.datetime64('2023-01-01')).astype('timedelta64[D]').astype(int)

    p90_shifted_values = np.roll(percentile_90.values, -day_of_year_start)
    length_of_p90 = len(percentile_90.dayofyear)
    new_time_index = pd.date_range(start='2023-06-16', periods=length_of_p90, freq='D')
    p90_clim_ind_s = xr.DataArray(p90_shifted_values, coords={'time': new_time_index}, dims=['time'])

    tmean_shifted_values = np.roll(tmean_mask.__xarray_dataarray_variable__.values, -day_of_year_start)
    tmean_ind_s = xr.DataArray(tmean_shifted_values.squeeze(), coords={'time': new_time_index}, dims=['time'])

    tmean_clim_shifted_values = np.roll(tmean_clim_mask.__xarray_dataarray_variable__.values, -day_of_year_start)
    tmean_clim_ind_s = xr.DataArray(tmean_clim_shifted_values, coords={'time': new_time_index}, dims=['time'])

    tmean_anomaly = tmean_mask.__xarray_dataarray_variable__.groupby('time.dayofyear') - tmean_clim_ind_s.groupby('time.dayofyear').mean('time')

    days_exceeding_99th = np.sum((tmean_anomaly.values + tmean_clim_ind_s.values) > p90_clim_ind_s.values)
    
    return days_exceeding_99th

# Create a dictionary with state names as keys and number of extreme days as values
extreme_days_dict = {state: calculate_extreme_days(state) for state in unique_states['NAME_1']}

# Define custom colorscale
custom_colorscale = [
    [0, 'rgb(255,255,204)'],
    [1/8, 'rgb(255,237,160)'],
    [2/8, 'rgb(254,217,118)'],
    [3/8, 'rgb(254,178,76)'],
    [4/8, 'rgb(253,141,60)'],
    [5/8, 'rgb(252,78,42)'],
    [6/8, 'rgb(227,26,28)'],
    [7/8, 'rgb(189,0,38)'],
    [1, 'rgb(128,0,38)']
]

# Define the layout
app.layout = html.Div([
    html.H1("Temperature Time Series Dashboard"),
    html.P("Caution: The displayed boundaries may not accurately represent political boundaries."),
    dcc.Graph(id='map-graph'),
    dcc.Graph(id='temp-graph')
])

@app.callback(
    Output('map-graph', 'figure'),
    Input('map-graph', 'figure')
)
def display_map(_):
    map_trace = go.Choropleth(
        geojson=india_geojson_f,
        locations=list(extreme_days_dict.keys()),
        z=list(extreme_days_dict.values()),
        colorscale=custom_colorscale,
        marker_line_width=0.5,
        showscale=True,
        featureidkey="properties.NAME_1",
        hoverinfo='location+z',
        colorbar=dict(
            title="Days > 99th Percentile",
            tickvals=list(range(20, 181, 20))
        )
    )

    map_layout = go.Layout(
        geo=dict(
            showcountries=False,
            showcoastlines=False,
            fitbounds="geojson",
            projection_scale=10  # Adjust this scale to fit India better
        ),
        clickmode='event+select',
        dragmode=False,  # Disable zooming and panning
        annotations=[
            dict(
                x=0.5,
                y=-0.1,
                xref='paper',
                yref='paper',
                text='Created by: Ligin and Lijo\nData Source: CPC NOAA',
                showarrow=False,
                font=dict(size=14)
            )
        ]
    )

    return go.Figure(data=[map_trace], layout=map_layout)

@app.callback(
    Output('temp-graph', 'figure'),
    Input('map-graph', 'clickData')
)
def update_graph(clickData):
    if clickData is None:
        return go.Figure()  # Return an empty figure if no state is clicked

    state = clickData['points'][0]['location']

    st = unique_states.loc[unique_states['NAME_1'] == state, :]
    mask = regionmask.mask_3D_geopandas(st, ds_tmean['lon'], ds_tmean['lat'])
    tmean_mask = ds_tmean.where(mask).mean(['lat', 'lon'])
    tmean_clim_mask = ds_tmean_ref.sel(state=state)
    tmean_mask = tmean_mask.sel(time=slice('2023-06-16', '2024-06-15')).squeeze()
    percentile_90 = per_90.sel(state=state).__xarray_dataarray_variable__

    start_date = np.datetime64('2023-06-16')
    day_of_year_start = (start_date - np.datetime64('2023-01-01')).astype('timedelta64[D]').astype(int)

    p90_shifted_values = np.roll(percentile_90.values, -day_of_year_start)
    length_of_p90 = len(percentile_90.dayofyear)
    new_time_index = pd.date_range(start='2023-06-16', periods=length_of_p90, freq='D')
    p90_clim_ind_s = xr.DataArray(p90_shifted_values, coords={'time': new_time_index}, dims=['time'])

    tmean_shifted_values = np.roll(tmean_mask.__xarray_dataarray_variable__.values, -day_of_year_start)
    tmean_ind_s = xr.DataArray(tmean_shifted_values.squeeze(), coords={'time': new_time_index}, dims=['time'])

    tmean_clim_shifted_values = np.roll(tmean_clim_mask.__xarray_dataarray_variable__.values, -day_of_year_start)
    tmean_clim_ind_s = xr.DataArray(tmean_clim_shifted_values, coords={'time': new_time_index}, dims=['time'])

    tmean_anomaly = tmean_mask.__xarray_dataarray_variable__.groupby('time.dayofyear') - tmean_clim_ind_s.groupby('time.dayofyear').mean('time')

    days_exceeding_99th = np.sum((tmean_anomaly.values + tmean_clim_ind_s.values) > p90_clim_ind_s.values)

    exceed_mask = (tmean_anomaly.values + tmean_clim_ind_s.values) > p90_clim_ind_s.values

    trace1 = go.Scatter(
        x=tmean_clim_ind_s.time.values,
        y=tmean_clim_ind_s.values,
        mode='lines',
        name='Mean',
        line=dict(color='black', width=2)
    )

    trace2 = go.Scatter(
        x=p90_clim_ind_s.time.values,
        y=p90_clim_ind_s.values,
        mode='lines',
        name='99th Percentile',
        fill='tonexty',
        line=dict(color='grey', width=1),
        fillcolor='rgba(200, 200, 200, 0.5)'
    )

    trace3 = go.Bar(
        x=tmean_anomaly.time.values[~exceed_mask],
        y=tmean_anomaly.values[~exceed_mask],
        base=tmean_clim_ind_s.values[~exceed_mask],
        marker=dict(color='grey'),
        name='Temperature Anomaly'
    )

    trace4 = go.Bar(
        x=tmean_anomaly.time.values[exceed_mask],
        y=tmean_anomaly.values[exceed_mask],
        base=tmean_clim_ind_s.values[exceed_mask],
        marker=dict(color='red'),
        name='Temperature Anomaly Exceeding 99th Percentile'
    )

    layout = go.Layout(
        title=f'{state} Daily Temperature (°C)',
        xaxis=dict(title='Date'),
        yaxis=dict(title='Temperature (°C)'),
        annotations=[
            dict(
                x=0.5,
                y=-0.3,
                xref='paper',
                yref='paper',
                text=f'Number of Days Exceeding 99th Percentile: {days_exceeding_99th}',
                showarrow=False,
                font=dict(size=14)
            ),
            dict(
                x=0.5,
                y=-0.4,
                xref='paper',
                yref='paper',
                text='Created by: Ligin and Lijo; \n Data Source: CPC NOAA',
                showarrow=False,
                font=dict(size=14)
            )
        ]
    )

    fig = go.Figure(data=[trace1, trace2, trace3, trace4], layout=layout)
    return fig

if __name__ == '__main__':
    app.run_server(debug=True, port=8060)


# In[ ]:






# In[ ]:




