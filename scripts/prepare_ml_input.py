# Import packages
import sys

import pandas as pd

import utils

# Country code
country_code = sys.argv[1]

# NA values needed for Porijõgi
na_values = -99

# Time interval
time_interval = sys.argv[2]

# Set project directory
utils.set_project_dir()

# Read flow data
flow_df = utils.read_flow_data(country_code, time_interval)

# Read predictor variable data
pred_var_df = utils.read_pred_var_data(country_code, time_interval, na_values)

# Get lags for predictor variables
variables = ['Pcp', 'Tmax', 'Tmin']
lag_width = 1
n_lags = utils.get_n_lags(time_interval)
pred_var_lags = utils.calc_lags(pred_var_df, variables, time_interval, lag_width, n_lags)

# Get rolling aggregates for predictor variables
window_width = utils.get_window_width(time_interval)
n_windows = 4
agg_func_list = ['mean', 'max', 'min']
pred_var_rolling_aggs = utils.calc_rolling_aggs(
    pred_var_lags, variables, time_interval, window_width, n_windows, agg_func_list
)

# Get lags for flow
lag_width = 1
n_lags = utils.get_n_lags(time_interval)
flow_lags = utils.calc_lags(flow_df, [f'Q'], time_interval, lag_width, n_lags)

# Get leads for flow
lead_width = 1
n_leads = utils.get_n_lags(time_interval)
flow_leads = utils.calc_leads(flow_lags, [f'Q'], time_interval, lead_width, n_leads)

# Merge predictor variables with flow
ml_input = flow_leads.merge(pred_var_rolling_aggs, on='Date', how='left')

# Add temporal dimension
if time_interval == 'd':
    ml_input['DOY'] = pd.to_datetime(ml_input['Date']).dt.dayofyear
elif time_interval == 'm':
    ml_input['MOY'] = pd.to_datetime(ml_input['Date']).dt.month

# Export ML input
ml_input.to_csv(f'ml/{country_code}/models/{country_code}_ml_input_{time_interval}.csv', index=False)
