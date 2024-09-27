def check_single_value(df):
    return {col: df[col].nunique() == 1 for col in df.columns}
