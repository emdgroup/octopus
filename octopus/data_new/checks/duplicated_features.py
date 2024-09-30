import pandas as pd


def check_duplicated_features(
    data: pd.DataFrame, feature_columns: list[str], sample_id: str
):
    """Check for duplicates (rows) in all features."""
    duplicated_features = data[feature_columns].duplicated().any()

    if sample_id is not None:
        duplicated_features_and_sample = (
            data[feature_columns + [sample_id]].duplicated().any()
        )
    else:
        duplicated_features_and_sample = None

    return duplicated_features, duplicated_features_and_sample
