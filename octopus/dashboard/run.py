"""Run dashboard."""

from pathlib import Path

import dash
from attrs import define, field
from dash import Dash

from octopus.dashboard.components.appshell import create_appshell
from octopus.dashboard.library.api.sqlite import SqliteAPI
from octopus.dashboard.library.data_processor import (
    EDADataProcessor,
    ResultsDataProcessor,
)
from octopus.data.config import OctoData

dash._dash_renderer._set_react_version("18.2.0")


@define
class OctoDash:
    """Octo Dashboard."""

    data: OctoData | Path = field(default=None)
    port: int = field(default=8050)
    overwrite_db: bool = field(default=False)
    db_filename: str = field(default="")

    def __attrs_post_init__(self):
        # create db
        if isinstance(self.data, Path):
            self.db_filename = self.data.joinpath("dashboard.db")

            if self.overwrite_db or not self.db_filename.is_file():
                sqlite = SqliteAPI(self.db_filename)
                # create eda and results from study
                self.create_eda_tables(sqlite)
                self.create_results_tables(sqlite)
            else:
                print("Database already exists. Skipping creation.")

        elif isinstance(self.data, OctoData):
            self.db_filename = "dashboard.db"
            sqlite = SqliteAPI(self.db_filename)
            # create eda from octodata
            self.create_eda_tables(sqlite)

    def create_eda_tables(self, sqlite: SqliteAPI) -> None:
        """Create database."""
        eda_data_processor = EDADataProcessor(self.data)
        df_dataset, df_data_info = eda_data_processor.get_dataset()
        df_description = eda_data_processor.create_eda_data_description()
        df_col = eda_data_processor.get_eda_column_info()

        sqlite.insert_dataframe("description", df_description)
        sqlite.insert_dataframe("dataset", df_dataset)
        sqlite.insert_dataframe("dataset_info", df_data_info)
        sqlite.insert_dataframe("column_description", df_col)

    def create_results_tables(self, sqlite: SqliteAPI) -> None:
        """Load data for results."""
        results_data_processor = ResultsDataProcessor(self.data)
        df_predictions, df_scores = results_data_processor.get_predictions()
        df_feature_importances = results_data_processor.get_feature_importances()
        df_optuna = results_data_processor.get_optuna_trials()
        (
            df_config_study,
            df_config_manager,
            df_config_sequence,
        ) = results_data_processor.get_configs()

        sqlite.insert_dataframe("predictions", df_predictions)
        sqlite.insert_dataframe("scores", df_scores)
        sqlite.insert_dataframe("optuna_trials", df_optuna)
        sqlite.insert_dataframe("config_study", df_config_study)
        sqlite.insert_dataframe("config_manager", df_config_manager)
        sqlite.insert_dataframe("config_sequence", df_config_sequence)
        if not df_feature_importances.empty:
            sqlite.insert_dataframe("feature_importances", df_feature_importances)

    def run(self):
        """Start dashboard."""
        app = Dash(
            __name__,
            suppress_callback_exceptions=True,
            use_pages=True,
            update_title=None,
        )

        show_results = True if isinstance(self.data, Path) else False
        app.layout = create_appshell(
            data=dash.page_registry.values(),
            show_results=show_results,
            db_filename=str(self.db_filename),
        )
        app.run_server(debug=True, host="0.0.0.0", port=self.port)
