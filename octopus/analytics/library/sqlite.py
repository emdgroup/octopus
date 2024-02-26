"""Sqlite API."""

import sqlite3

import pandas as pd

DB_NAME = "data.db"


def query(SQL_command):
    """Query."""
    con = sqlite3.connect(DB_NAME)
    df = pd.read_sql_query(SQL_command, con)
    con.close()
    return df


def free(SQL_command):
    """Execute SQL statement without return."""
    con = sqlite3.connect(DB_NAME)
    cursor = con.cursor()
    cursor.execute(SQL_command.replace("None", "NULL"))
    con.commit()
    cursor.close()
    con.close()


def insert_dataframe(table_name, df, index):
    """Insert dataframe."""
    con = sqlite3.connect(DB_NAME)
    df.set_index(index).to_sql(table_name, con, if_exists="replace")
    con.close()
