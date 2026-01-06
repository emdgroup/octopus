import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
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
    ## TODO
    # - create predict directory
    # - create utility functions in separate file
    # - functionality:
    # (1) study overview: which workflow tasks, number of splits
    # (2) performance overview for certain given metric
    # (3) provide feature lists for each task
    return


@app.cell
def _():
    # Setup - Import required libraries
    from octopus.predict.notebook_utils import setup_notebook

    OctoExperiment, Path, json, pd, re = setup_notebook()
    return OctoExperiment, Path, json, pd, re


@app.cell
def _(Path):
    # INPUT: Select study
    study_directory = "../studies/workflow_sequential_tasks/"

    path_study = Path(study_directory)

    # Display path status
    if path_study.exists():
        print(f"✓ Selected study path: {path_study}")
    else:
        print(f"⚠️ WARNING: Study path does not exist: {path_study}")
    return (path_study,)


@app.cell
def _(json, path_study):
    # Study information and available sequence items
    with open(path_study / "config.json") as f:
        config = json.load(f)

    ml_type = config["ml_type"]
    n_folds_outer = config["n_folds_outer"]
    workflow_tasks = config["workflow"]

    print("Information on workflow tasks in this study")
    print("Number of workflow tasks:", len(workflow_tasks))
    print("Number of outer folds:", n_folds_outer)

    # get octo workflows
    octo_workflow_lst = []
    for _cnt, _item in enumerate(workflow_tasks):
        print(f"Task {_item['task_id']}: {_item['module']} ")
        if _item["module"] == "octo":
            octo_workflow_lst.append(_item["task_id"])
    print("Octo workflow tasks:", octo_workflow_lst)
    print()
    return ml_type, n_folds_outer, workflow_tasks


@app.cell
def _(ml_type, n_folds_outer, path_study, workflow_tasks):
    # Validate study requirements

    # Check 1: Verify ml_type is classification
    if ml_type != "classification":
        raise ValueError(
            f"❌ ERROR: This notebook is for classification tasks only.\nFound ml_type: '{ml_type}'\nExpected: 'classification'"
        )
    else:
        print(f"✓ ML Type: {ml_type}")

    # Check 2: Verify study has been run (check for outersplit directories)
    outersplit = sorted(
        [d for d in path_study.glob("outersplit*") if d.is_dir()], key=lambda x: int(x.name.replace("outersplit", ""))
    )

    if not outersplit:
        raise ValueError(
            f"❌ ERROR: No experiment directories found in study path.\nStudy path: {path_study}\nThe study may not have been run yet."
        )
    else:
        print(f"✓ Found {len(outersplit)} experiment directory/directories")

    # Check that all expected outersplit directories exist
    expected_outersplit_ids = list(range(n_folds_outer))
    print(f"Expected outersplit IDs: {expected_outersplit_ids}")

    missing_outersplits = []
    for split_id in expected_outersplit_ids:
        expected_split_dir = path_study / f"outersplit{split_id}"
        if not expected_split_dir.exists():
            print(f"⚠️  WARNING: Missing directory 'outersplit{split_id}'")
            missing_outersplits.append(split_id)

    if missing_outersplits:
        print(f"⚠️  {len(missing_outersplits)} outersplit directory/directories missing")
    else:
        print("✓ All expected outersplit directories found")

    # Check 3: Verify experiments contain results
    # Extract task_ids from workflow_tasks
    expected_task_ids = [task["task_id"] for task in workflow_tasks]
    print(f"Expected workflow task IDs: {expected_task_ids}")

    has_results = False
    missing_dirs_found = False

    for split_dir in outersplit:
        workflow_dirs = list(split_dir.glob("workflowtask*"))
        if workflow_dirs:
            has_results = True

        # Check that all expected workflow task directories exist
        for task_id in expected_task_ids:
            expected_dir = split_dir / f"workflowtask{task_id}"
            if not expected_dir.exists():
                print(f"⚠️  WARNING: Missing directory '{expected_dir.name}' in {split_dir.name}")
                missing_dirs_found = True

    if not has_results:
        raise ValueError(
            "❌ ERROR: No workflow results found in experiments.\nThe study may not have completed successfully."
        )
    elif missing_dirs_found:
        print("⚠️  Study has completed workflow tasks, but some directories are missing (see warnings above)")
    else:
        print("✓ Study has completed workflow tasks - all expected directories found")
    return (outersplit,)


@app.cell
def _(OctoExperiment, outersplit, pd, re):
    # Model performance overview and selected features

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

    print("Listing of outer splits available in this study")
    # iterate through outer splits
    for path_split in outersplit:
        # name of outer split
        split_name = path_split.name
        # number of outer split
        match = re.search(r"\d+$", split_name)
        split_num = int(match.group()) if match else None

        print(f"Processing split {split_num} at {path_split} ...")

        # workflows
        path_workflows = [f for f in path_split.glob("workflowtask*") if f.is_dir()]

        # iterate through workflows
        for path_workflow in path_workflows:
            # name of workflow task
            workflow_name = str(path_workflow.name)
            # number of workflow task
            match = re.search(r"\d+", workflow_name)
            workflow_num = int(match.group()) if match else None
            path_exp_pkl = path_workflow.joinpath(f"exp{split_num}_{workflow_num}.pkl")

            print(f"\tWorkflow Task {workflow_num} at {path_exp_pkl}")

            if path_exp_pkl.exists():
                # load experiment
                exp = OctoExperiment.from_pickle(path_exp_pkl)
                # iterate through keys
                for key, result in exp.results.items():
                    print(f"\t\t{key}: {'\n\t\t\t'.join(f'{m}: {s}' for m, s in result.scores.items())}")

                    df.loc[len(df)] = [
                        split_num,
                        workflow_num,
                        workflow_name,
                        str(key),
                        result.scores,
                        len(result.selected_features),
                        result.selected_features,
                    ]
    return (df,)


@app.cell
def _(df, pd, workflow_tasks):
    # Performance overview
    for _num_task, _item2 in enumerate(workflow_tasks):
        print(f"\033[1mWorkflow task: {_item2['task_id']}\033[0m")

        df_workflow = df[df["Workflow"] == _item2["task_id"]]

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


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
