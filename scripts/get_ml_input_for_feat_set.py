# Import packages
import sys

import pandas as pd

import utils

# Country code
country_code = sys.argv[1]

# Feature set
feat_set = sys.argv[2]

# Time interval
time_interval = feat_set[-1]

# Target
target = f'Q_{time_interval}+1'

# Set project directory
utils.set_project_dir()

# Read ML input
ml_input = pd.read_csv(f'ml/{country_code}/models/{country_code}_ml_input_{time_interval}.csv')

# Get ML input based on time interval and feature set
features = ['Date', target]
if time_interval.lower() == 'd':
    date_filter = ~(
        (pd.to_datetime(ml_input['Date']).dt.year == pd.to_datetime(ml_input['Date']).min().year) &
        (pd.to_datetime(ml_input['Date']).dt.month == pd.to_datetime(ml_input['Date']).min().month) |
        (pd.to_datetime(ml_input['Date']).dt.year == pd.to_datetime(ml_input['Date']).max().year) &
        (pd.to_datetime(ml_input['Date']).dt.month == pd.to_datetime(ml_input['Date']).max().month)
    )
    ml_input = ml_input[date_filter].reset_index(drop=True)
    features.append('DOY')
    if feat_set in [f'FS1_{time_interval}', f'FS3_{time_interval}']:
        for variable in ['Pcp', 'Tmax', 'Tmin']:
            features.append(f'{variable}_{time_interval}')
            for i in range(1, 8):
                features.append(f'{variable}_{time_interval}-{i}')
        for col in ml_input.columns:
            if 'rolling' in col:
                features.append(col)
        if feat_set == f'FS3_{time_interval}':
            features.append(f'Q_{time_interval}')
    else:
        features.append(f'Q_{time_interval}')
        for i in range(1, 8):
            features.append(f'Q_{time_interval}-{i}')
if time_interval.lower() == 'm':
    date_filter = ~(
        (pd.to_datetime(ml_input['Date']).dt.year == pd.to_datetime(ml_input['Date']).min().year) |
        (pd.to_datetime(ml_input['Date']).dt.year == pd.to_datetime(ml_input['Date']).max().year)
    )
    ml_input = ml_input[date_filter].reset_index(drop=True)
    features.append('MOY')
    if feat_set in [f'FS1_{time_interval}', f'FS3_{time_interval}']:
        for variable in ['Pcp', 'Tmax', 'Tmin']:
            features.append(f'{variable}_{time_interval}')
            for i in range(1, 13):
                features.append(f'{variable}_{time_interval}-{i}')
        for col in ml_input.columns:
            if 'rolling' in col:
                features.append(col)
        if feat_set == f'FS3_{time_interval}':
            features.append(f'Q_{time_interval}')
    elif feat_set == f'FS2_{time_interval}':
        features.append(f'Q_{time_interval}')
        for i in range(1, 13):
            features.append(f'Q_{time_interval}-{i}')
ml_input = ml_input[features]

# Export ML input to CSV
model_dir = utils.get_model_dir(country_code, target, feat_set)
ml_input.to_csv(f'{model_dir}/{country_code}_{target}_{feat_set}_ml_input.csv', index=False)
