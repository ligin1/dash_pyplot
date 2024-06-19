#!/usr/bin/env python
# coding: utf-8

# In[2]:


import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import geopandas as gpd
import xarray as xr
import pandas as pd
import numpy as np
import regionmask
import plotly.graph_objs as go

# Create the Dash app
app = dash.Dash(__name__)
india_geojson = 'gadm41_IND_1.json'#
india_gdf = gpd.read_file(india_geojson)
states_to_remove = ['DadraandNagarHaveli', 'Lakshadweep', 'DamanandDiu', 'Puducherry']
india_gdf_f = india_gdf[~india_gdf['NAME_1'].isin(states_to_remove)]
unique_states = india_gdf_f.drop_duplicates(subset=['NAME_1'])

# Define the layout
app.layout = html.Div([
    html.H1("Temperature Time Series Dashboard"),
    dcc.Dropdown(
        id='state-dropdown',
        options=[{'label': state, 'value': state} for state in unique_states['NAME_1'].unique()],
        value=unique_states['NAME_1'].unique()[0]
    ),
    dcc.Graph(id='temp-graph')
])

@app.callback(
    Output('temp-graph', 'figure'),
    [Input('state-dropdown', 'value')]
)
def update_graph(state):
    ds_tmean_ref = xr.open_dataset('precomputed_mean.nc')
    ds_tmean = xr.open_dataset('CPC_tmean_18june2024.nc')
    per_90 = xr.open_dataset('precomputed_percentiles.nc')
    
    india_geojson = 'gadm41_IND_1.json'
    india_gdf = gpd.read_file(india_geojson)
    states_to_remove = ['DadraandNagarHaveli', 'Lakshadweep', 'DamanandDiu', 'Puducherry']
    india_gdf_f = india_gdf[~india_gdf['NAME_1'].isin(states_to_remove)]
    unique_states = india_gdf_f.drop_duplicates(subset=['NAME_1'])
    
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

    # Separate the anomalies into two categories: exceeding and not exceeding the 99th percentile
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
                text='Created by: Ligin Joseph\nData Source: India',
                showarrow=False,
                font=dict(size=14)
            )
        ]
    )

    fig = go.Figure(data=[trace1, trace2, trace3, trace4], layout=layout)
    return fig


# In[ ]:




