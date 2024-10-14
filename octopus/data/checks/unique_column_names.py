"""Check unique column names."""


def check_unique_column_names(
    feature_columns: list[str], target_columns: list[str], row_id: str
):
    """Check unique column names.

    In the combined list of feature_columns and target_columns a column name
    must only appear once.
    """
    combined_list = feature_columns + target_columns + [row_id]
    return {name: False for name in combined_list if combined_list.count(name) > 1}
