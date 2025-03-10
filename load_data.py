from pathlib import Path

import duckdb

# Define the base directory where your partitioned Parquet files are stored
path_study = Path("./studies/basic_regression_example")

# Use DuckDB to query the Parquet files with Hive-style partitioning
# query_optuna = f"""
# SELECT * FROM read_parquet('{path_study}/*/*/optuna*.parquet', hive_partitioning=true)
# """
query_predictions = f"""
SELECT * FROM read_parquet('{path_study}/*/*/predictions*.parquet', hive_partitioning=true)
"""
# Execute the query
# df_optuna = duckdb.query(query_optuna).to_df()
df_predictions = duckdb.query(query_predictions).to_df()

# Now `result_df` contains the combined data from all partitioned Parquet files
# print(df_optuna)
print(df_predictions)
