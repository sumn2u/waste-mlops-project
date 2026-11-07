import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

# Load reference (training) features and current (production) features
reference = pd.read_csv("reference_features.csv")
current = pd.read_csv("production_features.csv")

report = Report(metrics=[DataDriftPreset()])
report.run(reference=reference, current=current)

if report.as_dict()['metrics'][0]['result']['drift_detected']:
    print("ALERT: Drift detected - triggering retraining")
    # In a real setup, this would call the GitHub API to start a new run