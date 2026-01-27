"""Utility functions for Jupyter notebooks."""

import json
import re
from pathlib import Path

import pandas as pd

from octopus.experiment import OctoExperiment
from octopus.metrics.utils import get_performance_from_model


def show_study_details(study_directory: str | Path, expected_ml_type: str | None = None, verbose: bool = True) -> dict:
    """Display and validate study details including configuration and structure.

    This function reads the study configuration, validates the study structure,
    and displays information about the workflow tasks and experiment directories.

    Args:
        study_directory: Path to the study directory.
        expected_ml_type: Expected ML type (e.g., 'classification',
            'regression'). If provided, validates that the study matches this type.
        verbose: If True, prints detailed information.

    Returns:
        Dictionary containing study information with keys:
            - 'path': Path object of the study directory
            - 'config': Study configuration dictionary
            - 'ml_type': Machine learning type
            - 'n_folds_outer': Number of outer folds
            - 'workflow_tasks': List of workflow task configurations
            - 'outersplit_dirs': List of outersplit directory paths
            - 'expected_task_ids': List of expected task IDs
            - 'octo_workflow_tasks': List of task IDs for octo modules
            - 'missing_outersplits': List of missing outersplit IDs
            - 'missing_workflow_dirs': List of missing workflow directories

    Raises:
        ValueError: If expected_ml_type is provided and doesn't match the study's ml_type,
            or if no outersplit directories are found, or if no workflow results are found.
        FileNotFoundError: If the study directory does not exist.

    Example:
        >>> from octopus.predict.notebook_utils import show_study_details
        >>> study_info = show_study_details("./studies/my_study/", expected_ml_type="classification")
        >>> print(f"Study has {len(study_info['workflow_tasks'])} workflow tasks")
    """
    path_study = Path(study_directory)

    # Display path status
    if not path_study.exists():
        raise FileNotFoundError(f"⚠️ WARNING: Study path does not exist: {path_study}")

    if verbose:
        print(f"Selected study path: {path_study}\n")

    # Study information and available sequence items
    config_path = path_study / "config.json"
    with open(config_path) as f:
        config = json.load(f)

    ml_type = config["ml_type"]
    n_folds_outer = config["n_folds_outer"]
    workflow_tasks = config["workflow"]

    # Validate study requirements
    if verbose:
        print("Validate study....")

    # Check 1: Verify ml_type if expected_ml_type is provided
    if expected_ml_type is not None:
        if ml_type != expected_ml_type:
            raise ValueError(
                f"❌ ERROR: This notebook is for {expected_ml_type} tasks only.\n"
                f"Found ml_type: '{ml_type}'\nExpected: '{expected_ml_type}'"
            )
        if verbose:
            print(f"ML Type: {ml_type}")
    elif verbose:
        print(f"ML Type: {ml_type}")

    # Check 2: Verify study has been run (check for outersplit directories)
    outersplit = sorted(
        [d for d in path_study.glob("outersplit*") if d.is_dir()],
        key=lambda x: int(x.name.replace("outersplit", "")),
    )

    if not outersplit:
        raise ValueError(
            f"❌ ERROR: No experiment directories found in study path.\n"
            f"Study path: {path_study}\nThe study may not have been run yet."
        )
    if verbose:
        print(f"Found {len(outersplit)} outersplit directory/directories")

    # Check that all expected outersplit directories exist
    expected_outersplit_ids = list(range(n_folds_outer))
    if verbose:
        print(f"Expected outersplit IDs: {expected_outersplit_ids}")

    missing_outersplits = []
    for split_id in expected_outersplit_ids:
        expected_split_dir = path_study / f"outersplit{split_id}"
        if not expected_split_dir.exists():
            if verbose:
                print(f"⚠️  WARNING: Missing directory 'outersplit{split_id}'")
            missing_outersplits.append(split_id)

    if missing_outersplits:
        if verbose:
            print(f"⚠️  {len(missing_outersplits)} outersplit directory/directories missing")
    elif verbose:
        print("All expected outersplit directories found")

    # Check 3: Verify experiments contain results
    # Extract task_ids from workflow_tasks
    expected_task_ids = [task["task_id"] for task in workflow_tasks]
    if verbose:
        print(f"Expected workflow task IDs: {expected_task_ids}")

    has_results = False
    missing_workflow_dirs = []

    for split_dir in outersplit:
        workflow_dirs = list(split_dir.glob("workflowtask*"))
        if workflow_dirs:
            has_results = True

        # Check that all expected workflow task directories exist
        for task_id in expected_task_ids:
            expected_dir = split_dir / f"workflowtask{task_id}"
            if not expected_dir.exists():
                if verbose:
                    print(f"⚠️  WARNING: Missing directory '{expected_dir.name}' in {split_dir.name}")
                missing_workflow_dirs.append(str(expected_dir.relative_to(path_study)))

    if not has_results:
        raise ValueError(
            "❌ ERROR: No workflow results found in experiments.\nThe study may not have completed successfully."
        )
    elif missing_workflow_dirs:
        if verbose:
            print("⚠️  Study has completed workflow tasks, but some directories are missing (see warnings above)")
    elif verbose:
        print("Study has completed workflow tasks - all expected directories found")

    # Display workflow task information
    if verbose:
        print("\nInformation on workflow tasks in this study")
        print(f"Number of workflow tasks: {len(workflow_tasks)}")

    # Get octo workflows
    octo_workflow_lst = []
    for _item in workflow_tasks:
        if verbose:
            print(f"Task {_item['task_id']}: {_item['module']}")
        if _item["module"] == "octo":
            octo_workflow_lst.append(_item["task_id"])

    if verbose:
        print(f"Octo workflow tasks: {octo_workflow_lst}")

    # Return all collected information
    return {
        "path": path_study,
        "config": config,
        "ml_type": ml_type,
        "n_folds_outer": n_folds_outer,
        "workflow_tasks": workflow_tasks,
        "outersplit_dirs": outersplit,
        "expected_task_ids": expected_task_ids,
        "octo_workflow_tasks": octo_workflow_lst,
        "missing_outersplits": missing_outersplits,
        "missing_workflow_dirs": missing_workflow_dirs,
    }


def show_target_metric_performance(study_info: dict, details: bool = False) -> pd.DataFrame:
    """Display performance metrics for all workflow tasks in a study.

    This function loads experiments from all workflow tasks across outer splits,
    extracts performance metrics, and displays aggregated results.

    Args:
        study_info: Dictionary returned by show_study_details() containing study information.
        details: If False, only shows performance overview. If True, shows performance
            overview first, then detailed information for each experiment.

    Returns:
        DataFrame containing performance metrics with columns:
            - OuterSplit: Outer split number
            - Workflow: Workflow task number
            - Workflow_name: Name of the workflow directory
            - Results_key: Key identifying the result
            - Scores_dict: Dictionary of performance scores
            - n_features: Number of selected features
            - Selected_features: List of selected feature names

    Example:
        >>> from octopus.predict.notebook_utils import show_study_details, show_target_metric_performance
        >>> study_info = show_study_details("./studies/my_study/")
        >>> df = show_target_metric_performance(study_info, details=False)
    """
    # Initialize results dataframe
    df = pd.DataFrame(
        columns=[
            "OuterSplit",
            "Task",
            "Task_name",
            "Results_key",
            "Scores_dict",
            "n_features",
            "Selected_features",
        ]
    )

    # Collect data silently (no prints during data collection)
    for path_split in study_info["outersplit_dirs"]:
        # Name of outer split
        split_name = path_split.name
        # Number of outer split
        match = re.search(r"\d+$", split_name)
        split_num = int(match.group()) if match else None

        # Workflows
        path_workflows = [f for f in path_split.glob("workflowtask*") if f.is_dir()]

        # Iterate through workflows
        for path_workflow in path_workflows:
            # Name of workflow task
            workflow_name = str(path_workflow.name)
            # Number of workflow task
            match = re.search(r"\d+", workflow_name)
            workflow_num = int(match.group()) if match else None
            path_exp_pkl = path_workflow.joinpath(f"exp{split_num}_{workflow_num}.pkl")

            if path_exp_pkl.exists():
                # Load experiment
                exp = OctoExperiment.from_pickle(path_exp_pkl)
                # Iterate through keys
                for key, result in exp.results.items():
                    new_row = pd.DataFrame(
                        [
                            {
                                "OuterSplit": split_num,
                                "Task": workflow_num,
                                "Task_name": workflow_name,
                                "Results_key": str(key),
                                "Scores_dict": result.scores,
                                "n_features": len(result.selected_features),
                                "Selected_features": sorted(result.selected_features),
                            }
                        ]
                    )
                    df = pd.concat([df, new_row], ignore_index=True)

    # Sort dataframe by Task, then by OuterSplit
    df = df.sort_values(by=["Task", "OuterSplit"], ignore_index=True)

    # Performance overview (always shown)
    for _item in study_info["workflow_tasks"]:
        print(f"\033[1mWorkflow task: {_item['task_id']}\033[0m")

        df_workflow = df[df["Task"] == _item["task_id"]]

        # Available results keys
        res_keys = sorted(set(df_workflow["Results_key"].tolist()))
        print("Available results keys:", res_keys)

        for _key in res_keys:
            print("Selected results key:", _key)
            df_workflow_selected = df_workflow.copy()
            df_workflow_selected = df_workflow_selected[df_workflow_selected["Results_key"] == _key]
            # Expand the Scores_dict column into separate columns
            scores_df = df_workflow_selected["Scores_dict"].apply(pd.Series)
            # Combine with the original DataFrame, setting 'OuterSplit' as the index
            result_df = df_workflow_selected[["OuterSplit"]].join(scores_df).set_index("OuterSplit")
            # Remove columns that do not contain numeric values
            result_df = result_df.select_dtypes(include="number")
            mean_values = {}
            # Iterate through the columns
            for column in result_df.columns:
                if result_df[column].dtype in ["float64", "int64"]:
                    mean_values[column] = result_df[column].mean()
                else:
                    mean_values[column] = ""
            # Append the mean values as a new row
            result_df.loc["Mean"] = mean_values
            print(result_df)

    # Detailed printout (only if details=True)
    if details:
        print("\n" + "=" * 80)
        print("DETAILED INFORMATION")
        print("=" * 80 + "\n")
        print("Listing of outer splits available in this study")

        # Iterate through outer splits again for detailed output
        for path_split in study_info["outersplit_dirs"]:
            split_name = path_split.name
            match = re.search(r"\d+$", split_name)
            split_num = int(match.group()) if match else None

            print(f"Processing split {split_num} at {path_split} ...")

            # Workflows
            path_workflows = [f for f in path_split.glob("workflowtask*") if f.is_dir()]

            # Iterate through workflows
            for path_workflow in path_workflows:
                workflow_name = str(path_workflow.name)
                match = re.search(r"\d+", workflow_name)
                workflow_num = int(match.group()) if match else None
                path_exp_pkl = path_workflow.joinpath(f"exp{split_num}_{workflow_num}.pkl")

                print(f"\tWorkflow Task {workflow_num} at {path_exp_pkl}")

                if path_exp_pkl.exists():
                    exp = OctoExperiment.from_pickle(path_exp_pkl)
                    for key, result in exp.results.items():
                        print(f"\t\t{key}: {'\n\t\t\t'.join(f'{m}: {s}' for m, s in result.scores.items())}")

    return df


def show_selected_features(
    df_performance: pd.DataFrame, sort_task: int | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Display the number of selected features across outer splits and tasks.

    This function creates two summary tables:
    1. Number of features per outer split and task
    2. Feature frequency table showing how often each feature appears across outer splits for each task

    Args:
        df_performance: DataFrame returned by show_target_metric_performance()
            containing performance metrics and feature information.
        sort_task: Task ID to use for sorting the feature frequency table.
            If provided, the table will be sorted by frequency for this task (highest first).
            If None, uses the first task in the dataframe.

    Returns:
        Tuple of two DataFrames:
        - feature_table: Outer splits as rows, tasks as columns, showing feature counts
        - frequency_table: Features as rows, tasks as columns, showing selection frequency

    Example:
        >>> from octopus.predict.notebook_utils import show_study_details, show_target_metric_performance, show_selected_features
        >>> study_info = show_study_details("./studies/my_study/")
        >>> df_performance = show_target_metric_performance(study_info)
        >>> feature_table, freq_table = show_selected_features(df_performance, sort_task=0)
    """
    # Create a pivot table with OuterSplit as rows and Task as columns
    feature_table = df_performance.pivot_table(
        index="OuterSplit", columns="Task", values="n_features", aggfunc=lambda x: x.iloc[0]
    )

    # Calculate mean for each task (column) and add as a new row
    mean_row = feature_table.mean(axis=0)
    feature_table.loc["Mean"] = mean_row

    # Display the table
    print("\n" + "=" * 80)
    print("NUMBER OF SELECTED FEATURES")
    print("=" * 80)
    print("Rows: OuterSplit | Columns: Task ID")
    print(feature_table)
    print("=" * 80 + "\n")

    # Create feature frequency table
    # Get all unique tasks
    tasks = sorted(df_performance["Task"].unique())

    # Determine which task to use for sorting
    if sort_task is None:
        sort_task = int(tasks[0])

    # Collect all unique features across all tasks
    all_features = set()
    for features_list in df_performance["Selected_features"]:
        all_features.update(features_list)

    # Create a dictionary to store frequency counts
    frequency_data: dict[int, dict[str, int]] = {int(task): {} for task in tasks}

    # Count feature occurrences for each task across all outer splits
    for task in tasks:
        task_int = int(task)
        task_data = df_performance[df_performance["Task"] == task]
        for _, row in task_data.iterrows():
            for feature in row["Selected_features"]:
                if feature not in frequency_data[task_int]:
                    frequency_data[task_int][feature] = 0
                frequency_data[task_int][feature] += 1

    # Create frequency table
    frequency_table = pd.DataFrame(frequency_data)
    frequency_table = frequency_table.fillna(0).astype(int)

    # Sort by the specified task's frequency (highest first)
    # Ensure sort_task is treated as the column name
    sort_col = sort_task
    if sort_col in list(frequency_table.columns):
        frequency_table = frequency_table.sort_values(by=sort_col, ascending=False)  # type: ignore[call-overload]

    # Display the frequency table
    print("\n" + "=" * 80)
    print("FEATURE FREQUENCY ACROSS OUTER SPLITS")
    print("=" * 80)
    print("Rows: Features | Columns: Task ID")
    print(f"Sorted by Task {sort_task} frequency (highest first)")
    print(frequency_table)
    print("=" * 80 + "\n")

    return feature_table, frequency_table


def testset_performance_overview(predictor: "OctoPredict", metrics: list[str]) -> pd.DataFrame:  # type: ignore[name-defined] # noqa: F821
    """Display test performance metrics across all experiments in a task predictor.

    This function evaluates each experiment's model on the test dataset using the
    specified metrics and creates a summary table showing performance across experiments.

    Args:
        predictor: OctoPredict object containing experiments with trained models.
        metrics: List of metric names to evaluate (e.g., ['roc_auc', 'accuracy']).

    Returns:
        DataFrame with experiments as rows (plus a 'Mean' row), metrics as columns,
        showing performance values for each metric-experiment combination.

    Example:
        >>> from octopus.predict import OctoPredict
        >>> from octopus.predict.notebook_utils import testset_performance_overview
        >>> task_predictor = OctoPredict(study_directory="./studies/my_study/", task_id=0)
        >>> df_test_perf = testset_performance_overview(predictor=task_predictor, metrics=["roc_auc", "accuracy"])
    """
    # Collect performance data
    data_list = []

    print("Performance on test dataset (pooling)")

    for exp_id, experiment in predictor.experiments.items():
        # Create a row dictionary for this experiment
        row_data = {"outersplit": exp_id}

        for metric in metrics:
            performance = get_performance_from_model(
                experiment.model,
                experiment.data_test,
                experiment.feature_columns,
                metric,
                experiment.target_assignments,
                positive_class=experiment.positive_class,
            )
            row_data[metric] = performance

        data_list.append(row_data)

    # Create DataFrame
    df = pd.DataFrame(data_list)

    # Set experiment_id as index
    df = df.set_index("outersplit")

    # Calculate mean for each metric and add as a new row
    mean_row = df.mean(axis=0)
    df.loc["Mean"] = mean_row

    print(df)
    return df
