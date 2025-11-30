import marimo

__generated_with = "0.18.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    # first draft version
    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    # 📊 Study Analysis for Classification Tasks

    **Notebook to analyse trained classification models from Octopus studies**

    This interactive notebook provides comprehensive analysis tools for evaluating classification models trained using the Octopus framework. It includes:

    ## 🎯 Key Features

    - **Model Performance Metrics**: AUCROC, Accuracy, F1-Score, AUCPR, and more
    - **ROC Curve Analysis**: Individual and averaged ROC curves across experiments
    - **Confusion Matrices**: Absolute and relative confusion matrices with visualizations
    - **Feature Importance**: Permutation Feature Importance (PFI) and SHAP values
    - **Cross-Validation Results**: Performance across all outer folds
    - **Interactive Visualizations**: Plotly and Matplotlib charts for detailed exploration


    ## 📋 Usage Instructions

    1. **Select Study**: Modify the `path_study` variable to point to your study directory
    2. **Select Task**: Set the `task_id` to analyze a specific workflow task
    3. **Run Cells**: Execute cells sequentially to generate analysis outputs
    4. **Customize**: Adjust metrics, thresholds, and visualization parameters as needed

    ## ⚠️ Requirements

    - Study must be a classification task (`ml_type == "classification"`)
    - Study directory must contain completed experiments with results
    - Required packages: octopus, pandas, numpy, matplotlib, seaborn, plotly, sklearn

    ---
    """)
    return


@app.cell
def _():
    # Setup - Import required libraries
    import os
    import re
    import socket
    from pathlib import Path

    import pandas as pd

    from octopus.config.core import OctoConfig
    from octopus.experiment import OctoExperiment

    # Configure pandas display options
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    return OctoConfig, OctoExperiment, Path, os, pd, re, socket


@app.cell
def _(os, socket):
    # System info
    print("Notebook kernel is running on server:", socket.gethostname())
    print("Conda environment on server:", os.environ.get("CONDA_DEFAULT_ENV", "Not set"))
    print("Working directory: ", os.getcwd())
    return


@app.cell
def _(Path):
    # INPUT: Select study
    # Handle both running from main directory and from examples directory
    current_dir = Path.cwd()
    if current_dir.name == "examples":
        # Running from examples directory, go up one level
        base_dir = current_dir.parent
    else:
        # Running from main directory
        base_dir = current_dir

    path_study = base_dir / "studies" / "20250826A_octo_intro"
    print("Current directory:", current_dir)
    print("Base directory:", base_dir)
    print("Selected study path:", path_study)

    if not path_study.exists():
        print(f"WARNING: Study path does not exist: {path_study}")
    return (path_study,)


@app.cell
def _(OctoConfig, path_study):
    # Study information and available sequence items
    path_config = path_study / "config"
    config = OctoConfig.from_pickle(path_config / "config.pkl")

    ml_type = config.study.ml_type
    workflow_tasks = config.workflow.tasks

    print("Information on workflow tasks in this study")
    print("Number of workflow tasks:", len(workflow_tasks))

    # get octo workflows
    octo_workflow_lst = []
    for _cnt, _item in enumerate(workflow_tasks):
        print(f"Task {_item.task_id}:  {_item.module}")
        if _item.module == "octo":
            octo_workflow_lst.append(_item.task_id)
    print("Octo workflow tasks:", octo_workflow_lst)
    print()
    return ml_type, workflow_tasks


@app.cell
def _(ml_type, path_study):
    # Validate study requirements

    # Check 1: Verify ml_type is classification
    if ml_type != "classification":
        error_msg = f"❌ ERROR: This notebook is for classification tasks only.\nFound ml_type: '{ml_type}'\nExpected: 'classification'"
        print(error_msg)
        raise ValueError(error_msg)
    else:
        print(f"✓ ML Type: {ml_type}")

    # Check 2: Verify study has been run (check for experiment directories)
    experiment_dirs = list(path_study.glob("experiment*"))
    experiment_dirs = [d for d in experiment_dirs if d.is_dir()]

    if not experiment_dirs:
        error_msg = f"❌ ERROR: No experiment directories found in study path.\nStudy path: {path_study}\nThe study may not have been run yet."
        print(error_msg)
        raise ValueError(error_msg)
    else:
        print(f"✓ Found {len(experiment_dirs)} experiment(s)")

    # Check 3: Verify experiments contain results
    has_results = False
    for exp_dir in experiment_dirs:
        workflow_dirs = list(exp_dir.glob("workflowtask*"))
        if workflow_dirs:
            has_results = True
            break

    if not has_results:
        error_msg = (
            "❌ ERROR: No workflow results found in experiments.\nThe study may not have completed successfully."
        )
        print(error_msg)
        raise ValueError(error_msg)
    else:
        print("✓ Study has completed workflow tasks")
    return


@app.cell
def _(OctoExperiment, path_study, pd, re):
    # Model performance overview and selected features
    # get all experiments
    path_experiments = [f for f in path_study.glob("experiment*") if f.is_dir()]

    # results df
    df = pd.DataFrame(
        columns=[
            "Experiment",
            "Workflow",
            "Workflow_name",
            "Results_key",
            "Scores_dict",
            "n_features",
            "Selected_features",
        ]
    )

    print("Listing of experiments available in this study")
    # iterate through experiments
    for path_exp in path_experiments:
        # name of experiment
        exp_name = str(path_exp.name)
        # number of experiment
        match = re.search(r"\d+", exp_name)
        exp_num = int(match.group()) if match else None

        # workflows
        path_workflows = [f for f in path_exp.glob("workflowtask*") if f.is_dir()]
        print("Processing....:", path_exp)

        # iterate through workflows
        for path_workflow in path_workflows:
            # name of workflow task
            workflow_name = str(path_workflow.name)
            # number of workflow task
            match = re.search(r"\d+", workflow_name)
            workflow_num = int(match.group()) if match else None

            path_exp_pkl = path_workflow.joinpath(f"exp{exp_num}_{workflow_num}.pkl")

            if path_exp_pkl.exists():
                # load experiment
                exp = OctoExperiment.from_pickle(path_exp_pkl)
                # iterate through keys
                for key in exp.results:
                    sel_features = exp.results[key].selected_features
                    df.loc[len(df)] = [
                        exp_num,
                        workflow_num,
                        workflow_name,
                        str(key),
                        exp.results[key].scores,
                        len(sel_features),
                        sel_features,
                    ]

    return (df,)


@app.cell
def _(df, pd, workflow_tasks):
    # Performance overview
    for _num_task, _item2 in enumerate(workflow_tasks):
        print(f"\033[1mWorkflow task: {_item2.task_id}({_item2.module})\033[0m")

        df_workflow = df[df["Workflow"] == _item2.task_id]

        # available results keys
        res_keys = sorted(set(df_workflow["Results_key"].tolist()))
        print("Available results keys:", res_keys)

        for _key3 in res_keys:
            print("Selected results key:", _key3)
            df_workflow_selected = df_workflow.copy()
            df_workflow_selected = df_workflow_selected[df_workflow_selected["Results_key"] == _key3]
            # Expand the Scores_dict column into separate columns
            scores_df = df_workflow_selected["Scores_dict"].apply(pd.Series)
            # Combine with the original DataFrame, setting 'Experiment' as the index
            result_df = df_workflow_selected[["Experiment"]].join(scores_df).set_index("Experiment")
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
    return


if __name__ == "__main__":
    app.run()
