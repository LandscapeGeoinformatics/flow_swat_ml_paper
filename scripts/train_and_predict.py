# Import packages
import sys
import time

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from joblib import dump

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

# Model directory
model_dir = utils.get_model_dir(country_code, target, feat_set)

# Read ML input
ml_input = pd.read_csv(f'{model_dir}/{country_code}_{target}_{feat_set}_ml_input.csv')

# Test size list
# test_size_list = [i / 10 for i in range(5, 0, -1)]
test_size_list = [0.5]

# Train and predict for each test size in list
results_list = []
for test_size in test_size_list:

    # Extract features and target
    X = ml_input.iloc[:, 2:]
    y = ml_input[target]

    # Split the data into training and test sets
    random_state = 1
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, shuffle=False
    )

    # Export training and test data to CSV
    X_train.to_csv(
        f'{model_dir}/{country_code}_{target}_{feat_set}_feat_train_{int(test_size*100)}.csv', index_label='Index'
    )
    X_test.to_csv(
        f'{model_dir}/{country_code}_{target}_{feat_set}_feat_test_{int(test_size*100)}.csv', index_label='Index'
    )
    y_train.to_csv(
        f'{model_dir}/{country_code}_{target}_{feat_set}_target_train_{int(test_size*100)}.csv', index_label='Index'
    )
    y_test.to_csv(
        f'{model_dir}/{country_code}_{target}_{feat_set}_target_test_{int(test_size*100)}.csv', index_label='Index'
    )

    # Train model
    start_time = time.time()
    regressor = RandomForestRegressor()
    regressor.fit(X_train, y_train)
    end_time = time.time() - start_time
    training_time = time.strftime('%H:%M:%S', time.gmtime(end_time))

    # Save model
    dump(regressor, f'{model_dir}/{country_code}_{target}_{feat_set}_{int(test_size * 100)}.joblib')

    # Predict
    Y_train_pred = regressor.predict(X_train)
    Y_test_pred = regressor.predict(X_test)

    # Create DataFrame from observed and predicted values
    y_test_df = pd.DataFrame(y_test)
    y_test_df[f'{target}_pred'] = Y_test_pred
    y_train_df = pd.DataFrame(y_train)
    y_train_df[f'{target}_pred'] = Y_train_pred
    obs_vs_pred = ml_input[['Date']].join(pd.concat([y_test_df, y_train_df]))

    # Calculate residuals between observed and predicted values
    obs_vs_pred['Residual'] = round(obs_vs_pred[target] - obs_vs_pred[f'{target}_pred'], 3)
    obs_vs_pred.to_csv(
        f'{model_dir}/{country_code}_{target}_{feat_set}_obs_vs_pred_{int(test_size * 100)}.csv', index=False
    )

    # Get parameters
    params = regressor.get_params()
    results = {
        'country_code': country_code,
        'target': target,
        'feat_set': feat_set,
        'model_version': f'{target}_{feat_set}',
        'n_features': len(X.columns),
        'test_size': test_size,
        'n_samples_train': len(y_train),
        'n_samples_test': len(y_test),
        'r2_train': round(regressor.score(X_train, y_train), 2),
        'r2_test': round(r2_score(y_test, Y_test_pred), 2),
        'training_time': training_time,
        'n_estimators': params.get('n_estimators')
    }
    df_results = pd.DataFrame([results.values()], columns=results.keys())
    results_list.append(df_results)

# Create DataFrame from results and export to CSV
pd.concat(results_list)\
    .reset_index(drop=True)\
    .to_csv(f'{model_dir}/{country_code}_{target}_{feat_set}_results.csv', index=False)
