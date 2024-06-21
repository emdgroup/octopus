"""Sqlite API."""

import logging
import sqlite3
from pathlib import Path

import pandas as pd


class SqliteAPI:
    """Class to handle SQLite database operations."""

    def __init__(self, db_name="dashboard.db"):
        self.db_name = db_name

    def _connect(self):
        """Create a database connection and return it."""
        return sqlite3.connect(self.db_name)

    def query(self, SQL_command):
        """Execute a SQL query and return results in a dataframe."""
        with self._connect() as con:
            df = pd.read_sql_query(SQL_command, con)
        return df

    def insert_dataframe(self, table_name, df, index=None):
        """Insert a dataframe into a specified table in the database."""
        try:
            with self._connect() as con:
                if index is None:
                    df.to_sql(table_name, con, if_exists="replace", index=False)
                else:
                    df.set_index(index).to_sql(
                        table_name, con, if_exists="replace", index=False
                    )
        except Exception as e:
            logging.error(f"Error inserting dataframe into {table_name}: {e}")

    def delete_db(self):
        """Delete the database file if it exists."""
        database_path = Path(self.db_name)
        if database_path.exists():
            database_path.unlink()
            logging.info(f"Database {self.db_name} deleted successfully.")
        else:
            logging.info(f"No database file {self.db_name} found to delete.")
