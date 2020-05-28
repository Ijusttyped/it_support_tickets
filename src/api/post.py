from pathlib import Path

import pandas as pd
import requests

from data_access.load_file import load_agg_data
from modeling.preparation import impute_times

filename = Path("C:/Users/marcelfe/PycharmProjects/gdsc2_app/gdsc2_app/data/agg_feature_table.csv")
final, times_open, times_closed = load_agg_data(filename=filename)

feature_cols = ["Accept_time", "Analyze_time", "Build_time", "Clarify_time", "Deploy_time", "Design_time",
                "Package_time", "Test_time", "platcomp_cluster", "work_type", "work_priority"]
times_open = impute_times(final, times_open, times_closed, columns=feature_cols, imputation_method="mean")
post_this = times_open[["work_item", *feature_cols]]

req = requests.post("http://localhost:5000/api/v1.0/duration", json=post_this.to_json())

print(req.json())
predictions = pd.read_json(req.json())
print(predictions.head())
