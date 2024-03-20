# Import libraries
import sys
from joblib import Parallel, delayed

import utils

# Country code
country_code = sys.argv[1]

# Target
target = sys.argv[2]

# Set project directory
utils.set_project_dir()

# Feature set list
if target == 'Q_d+1':
    feat_set_list = ['FS3_d']
elif target == 'Q_m+1':
    feat_set_list = ['FS3_m']

# Test size list
# test_size_list = [i / 10 for i in range(5, 0, -1)]
test_size_list = [0.5]

# Create SHAP summary plots for each feature set
Parallel(n_jobs=4)(
    delayed(utils.plot_shap)(
        country_code, feat_set, test_size
    ) for feat_set in feat_set_list for test_size in test_size_list
)
