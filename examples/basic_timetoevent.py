"""Basic example for using Octopus time-to-event (survival analysis)."""

# This example demonstrates how to use Octopus for time-to-event analysis,
# also known as survival analysis. Time-to-event analysis is used to model
# the time until an event of interest occurs (e.g., patient survival, equipment failure).
#
# Key concepts:
# - Duration: The time until the event occurs or until censoring
# - Event: Binary indicator (1 = event occurred, 0 = censored/event not yet observed)
# - Censoring: When we don't observe the event (e.g., study ends, patient lost to follow-up)
#
# We will use synthetic data for this example to demonstrate the workflow.
# Please ensure your dataset has:
# - A duration column (time until event or censoring)
# - An event column (1 for event, 0 for censored)
# - Numeric features with no missing values (NaN)

### Necessary imports for this example
import numpy as np
import pandas as pd

from octopus import OctoStudy
from octopus.modules import Octo

### Generate synthetic time-to-event dataset

# Set random seed for reproducibility
np.random.seed(42)

# Create synthetic survival dataset
n_samples = 200  # Number of observations
n_features = 8  # Number of predictor features

# Generate features
X = np.random.randn(n_samples, n_features)
features = [f"feature_{i}" for i in range(n_features)]
df = pd.DataFrame(X, columns=features)

# Generate survival times (duration) based on features
# Higher feature values lead to longer survival times
risk_score = X[:, 0] + 0.5 * X[:, 1] - 0.3 * X[:, 2] + 0.2 * X[:, 3]
baseline_hazard = 0.1
duration = np.random.exponential(scale=1.0 / (baseline_hazard * np.exp(risk_score)))

# Generate censoring times
# In real studies, censoring occurs when observation ends before event
censoring_time = np.random.exponential(scale=15, size=n_samples)

# Observed time is minimum of event time and censoring time
observed_time = np.minimum(duration, censoring_time)

# Event indicator: 1 if event occurred, 0 if censored
event = (duration <= censoring_time).astype(int)

# Add to dataframe
df["duration"] = observed_time
df["event"] = event
df = df.reset_index()

print(f"Dataset created with {n_samples} samples and {n_features} features")
print(f"Events observed: {event.sum()} ({100 * event.mean():.1f}%)")
print(f"Censored observations: {(1 - event).sum()} ({100 * (1 - event.mean()):.1f}%)")
print(f"Mean survival time: {observed_time.mean():.2f}")

### Create and run OctoStudy

# For time-to-event analysis, we need to specify:
# - target_columns: Both 'duration' and 'event' columns
# - target_assignments: Mapping to identify which column is which
#   {"duration": "duration", "event": "event"}

print("\nStarting Octopus time-to-event workflow...")

study = OctoStudy(
    name="basic_timetoevent_example",
    ml_type="timetoevent",  # Specify time-to-event analysis
    target_metric="CI",  # Concordance Index (C-index)
    feature_columns=features,
    target_columns=["duration", "event"],
    sample_id="index",
    target_assignments={"duration": "duration", "event": "event"},
    outer_parallelization=True,
    run_single_experiment_num=0,  # 0: only first experiment, -1 runs all experiments
    tasks=[
        Octo(
            task_id=0,
            depends_on_task=-1,
            description="step1_octo",
            models=[
                "ExtraTreesSurv",  # Extra Trees for survival analysis
            ],
            n_trials=20,  # Number of hyperparameter optimization trials
            fi_methods_bestbag=["shap"],  # Use SHAP for feature importance
        ),
    ],
)

study.fit(data=df)

print("\nWorkflow completed successfully!")
print(f"Results saved to: studies/{study.name}/")
