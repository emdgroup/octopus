import marimo

__generated_with = "0.11.17"
app = marimo.App(width="medium")


@app.cell
def _():
    import altair as alt
    import duckdb
    import marimo as mo
    import numpy as np
    import plotly.graph_objects as go
    import polars as pl
    from sklearn.metrics import confusion_matrix

    return alt, confusion_matrix, duckdb, go, mo, np, pl


@app.cell(hide_code=True)
def _(mo):
    file_browser = mo.ui.file_browser(
        initial_path="studies",
        multiple=False,
        selection_mode="directory",
        label="Select your study by clicking on the folder icon.",
    )
    file_browser
    return (file_browser,)


@app.cell(hide_code=True)
def _(file_browser, mo):
    df_optuna = mo.sql(
        f"""
        SELECT * FROM read_parquet('{file_browser.path()}/*/*/optuna*.parquet', hive_partitioning=true)
        """,
        output=False,
    )
    return (df_optuna,)


@app.cell(hide_code=True)
def _(file_browser, mo, pl):
    df_predictions = mo.sql(
        f"""
        SELECT * FROM read_parquet('{file_browser.path()}/*/*/predictions*.parquet', hive_partitioning=true)
        """,
        output=False,
    ).with_columns(pl.col("prediction").cast(pl.Int64))

    df_predictions_pandas = mo.sql(
        f"""
        SELECT * FROM read_parquet('{file_browser.path()}/*/*/predictions*.parquet', hive_partitioning=true)
        """,
        output=False,
    ).to_pandas()
    return df_predictions, df_predictions_pandas


@app.cell(hide_code=True)
def _(file_browser, mo):
    df_configs = mo.sql(
        f"""
        SELECT * FROM read_parquet('{file_browser.path()}/*/config_study*.parquet', hive_partitioning=true)
        """,
        output=False,
    )
    return (df_configs,)


@app.cell(hide_code=True)
def _(file_browser, mo, pl):
    df_data_attrs = mo.sql(
        f"""
        SELECT * FROM read_parquet('{file_browser.path()}/*/data*.parquet', hive_partitioning=true)
        """,
        output=False,
    )
    target = df_data_attrs.filter(pl.col("Parameter") == "target_columns")["Value"][0]
    return df_data_attrs, target


@app.cell(hide_code=True)
def _(file_browser, mo, pl):
    df_feature_importances = (
        mo.sql(
            f"""
        SELECT * FROM read_parquet('{file_browser.path()}/*/*/feature-importance*.parquet', hive_partitioning=true)
        """,
            output=False,
        )
        .with_columns(pl.col("experiment_id").cast(pl.Int64))
        .with_columns(pl.col("task_id").cast(pl.Int64))
    )
    return (df_feature_importances,)


@app.cell(hide_code=True)
def _(df_predictions, mo, pl):
    unique_id_values = {
        k: sorted(v)
        for k, v in df_predictions.select(pl.all().cast(pl.Utf8))
        .select(["experiment_id", "task_id", "split_id"])
        .unique()
        .to_dict(as_series=False)
        .items()
    }

    dropdown_exp_id = mo.ui.dropdown(
        options=unique_id_values["experiment_id"],
        value=unique_id_values["experiment_id"][0],
        label="Experiment ID",
    )

    dropdown_seq_id = mo.ui.dropdown(
        options=unique_id_values["task_id"],
        value=unique_id_values["task_id"][0],
        label="Task ID ",
    )

    dropdown_split_id = mo.ui.dropdown(
        options=unique_id_values["split_id"],
        value=unique_id_values["split_id"][0],
        label="Split ID ",
    )
    return (
        dropdown_exp_id,
        dropdown_seq_id,
        dropdown_split_id,
        unique_id_values,
    )


@app.cell(hide_code=True)
def _(df_feature_importances, mo, pl):
    unique_id_values_feature_importance = {
        k: sorted(v)
        for k, v in df_feature_importances.select(pl.all().cast(pl.Utf8))
        .select(["fi_type"])
        .unique()
        .to_dict(as_series=False)
        .items()
    }

    dropdown_fi_types = mo.ui.dropdown(
        options=unique_id_values_feature_importance["fi_type"],
        value=unique_id_values_feature_importance["fi_type"][0],
        label="Feature importance type",
    )
    return dropdown_fi_types, unique_id_values_feature_importance


@app.cell(hide_code=True)
def _(df_optuna, mo, pl):
    unique_id_values_optuna = {
        k: sorted(v)
        for k, v in df_optuna.select(pl.all().cast(pl.Utf8))
        .select(["experiment_id", "task_id", "model_type"])
        .unique()
        .to_dict(as_series=False)
        .items()
    }

    dropdown_model = mo.ui.dropdown(
        options=unique_id_values_optuna["model_type"],
        value=unique_id_values_optuna["model_type"][0],
        label="Model",
    )
    return dropdown_model, unique_id_values_optuna


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Feature Importance""")
    return


@app.cell(hide_code=True)
def _(
    alt,
    df_feature_importances,
    dropdown_exp_id,
    dropdown_fi_types,
    dropdown_seq_id,
    dropdown_split_id,
    mo,
    pl,
):
    df_fi_plot = df_feature_importances.filter(
        (pl.col("experiment_id") == int(dropdown_exp_id.value))
        & (pl.col("task_id") == int(dropdown_seq_id.value))
        & (pl.col("split_id") == dropdown_split_id.value)
        & (pl.col("fi_type") == dropdown_fi_types.value)
    )

    chart_fi = (
        alt.Chart(df_fi_plot)
        .mark_bar()
        .encode(
            x=alt.X(
                "feature",
                title="Feature",
                sort=alt.SortField("importance", order="descending"),
            ),
            y=alt.Y("importance", title="Importance"),
            tooltip=["feature", "importance"],
        )
        .properties(title="Feature Importance")
    )

    mo.vstack(
        [
            mo.hstack(
                [
                    dropdown_exp_id,
                    dropdown_seq_id,
                    dropdown_split_id,
                    dropdown_fi_types,
                ]
            ),
            chart_fi,
        ],
        align="center",
    )
    return chart_fi, df_fi_plot


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Confusion Matrix for Test Split""")
    return


@app.cell(hide_code=True)
def _(
    alt,
    confusion_matrix,
    df_predictions,
    dropdown_exp_id,
    dropdown_seq_id,
    dropdown_split_id,
    mo,
    np,
    pl,
):
    df_confusion_matrix = df_predictions.filter(
        (pl.col("experiment_id") == int(dropdown_exp_id.value))
        & (pl.col("task_id") == int(dropdown_seq_id.value))
        & (pl.col("split_id") == dropdown_split_id.value)
        & (pl.col("split") == "test")
    )

    y_true = df_confusion_matrix["target"].to_numpy()
    y_pred = df_confusion_matrix["prediction"].to_numpy()

    class_labels = sorted(set(y_true) | set(y_pred))

    cm = confusion_matrix(y_true, y_pred)

    # Convert confusion matrix to a Polars DataFrame
    cm_df = pl.DataFrame(
        {
            "true_label": np.repeat(class_labels, len(class_labels)),
            "predicted_label": np.tile(class_labels, len(class_labels)),
            "value": cm.flatten(),
        }
    )

    # Create the Altair chart
    cm_chart = (
        alt.Chart(cm_df.to_pandas())
        .mark_rect()
        .encode(
            x=alt.X("predicted_label:N", title="Predicted label"),
            y=alt.Y(
                "true_label:N",
                title="True label",
                sort=alt.EncodingSortField(field="true_label", order="descending"),
            ),
            color=alt.Color("value:Q", scale=alt.Scale(scheme="blues")),
            tooltip=["true_label", "predicted_label", "value"],
        )
        .properties(title="Confusion Matrix", width=400, height=400)
    )

    # Add text labels
    cm_text = (
        alt.Chart(cm_df.to_pandas())
        .mark_text(baseline="middle")
        .encode(
            x="predicted_label:N",
            y="true_label:N",
            text="value:Q",
            color=alt.condition(
                alt.datum.value > cm_df["value"].max() / 2,
                alt.value("white"),
                alt.value("black"),
            ),
        )
    )

    # Combine the heatmap and text layers
    cm_final_chart = cm_chart + cm_text

    mo.vstack(
        [
            mo.hstack([dropdown_exp_id, dropdown_seq_id, dropdown_split_id]),
            cm_final_chart,
        ],
        align="center",
    )
    return (
        class_labels,
        cm,
        cm_chart,
        cm_df,
        cm_final_chart,
        cm_text,
        df_confusion_matrix,
        y_pred,
        y_true,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Optuna Insights""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Number of unique Trials by model type""")
    return


@app.cell(hide_code=True)
def _(alt, df_optuna, pl):
    # Group by experiment_id, task_id, and model_type
    df_chart_optuna_count = (
        df_optuna.group_by(["experiment_id", "task_id", "model_type"])
        .agg(pl.col("trial").n_unique().alias("trial_count"))
        .sort(["task_id", "experiment_id"])
    )

    # Create the base chart
    base = (
        alt.Chart(df_chart_optuna_count)
        .mark_bar()
        .encode(
            x=alt.X("model_type:N", title="Model Type", axis=alt.Axis(labelAngle=-45)),
            y=alt.Y("trial_count:Q", title="Number of Unique Trials"),
            color=alt.Color("model_type:N", legend=None),
        )
        .properties(width=180, height=120)
    )
    bars = base.mark_bar()

    # Create text labels for the bars
    text = base.mark_text(
        align="center",
        baseline="bottom",
        dy=-5,  # Slight offset to position the text above the bar
    ).encode(
        text="trial_count:Q"  # Display the trial count
    )

    # Combine the bars and text
    chart_optuna_count = (bars + text).properties(
        width=150,  # width of each individual chart
        height=120,  # height of each individual chart
    )

    # Create text labels for the bars
    text = base.mark_text(
        align="center",
        baseline="bottom",
        dy=-5,  # Slight offset to position the text above the bar
    ).encode(
        text="trial_count:Q"  # Display the trial count
    )

    # Create the faceted chart
    chart_optuna_count = base.facet(row="task_id:N", column="experiment_id:N").properties(
        title="Number of Unique Trials by Model Type, Task ID, and Experiment ID"
    )

    # Adjust the spacing of the facets
    chart_optuna_count = chart_optuna_count.configure_facet(spacing=10)
    chart_optuna_count
    return bars, base, chart_optuna_count, df_chart_optuna_count, text


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Optuna trails: Object value and best value""")
    return


@app.cell(hide_code=True)
def _(alt, df_optuna, dropdown_exp_id, dropdown_seq_id, mo, pl):
    def get_best_optuna_trials(df, direction="maximize"):
        if direction == "maximize":
            df_optuna_trials_best = (
                df.with_columns(pl.col("value").cum_max().alias("cummax"))
                .filter(pl.col("value") == pl.col("cummax"))
                .drop("cummax")
            )
        else:
            df_optuna_trials_best = (
                df.with_columns(pl.col("value").cum_min().alias("cummin"))
                .filter(pl.col("value") == pl.col("cummin"))
                .drop("cummin")
            )

        return df_optuna_trials_best

    df_optuna_filtered = df_optuna.filter(
        (pl.col("experiment_id") == int(dropdown_exp_id.value)) & (pl.col("task_id") == int(dropdown_seq_id.value))
    )

    df_best_optuna_trails = get_best_optuna_trials(df_optuna_filtered, "minimize")

    # Create the scatter plot for object values
    scatter = (
        alt.Chart(df_optuna_filtered)
        .mark_point(size=60)
        .encode(
            x="trial:Q",
            y=alt.Y("value:Q", scale=alt.Scale(type="log")),
            color=alt.Color("model_type:N", legend=alt.Legend(title="Model Type")),
            tooltip=["trial", "value", "model_type"],
        )
        .properties(width=600, height=400)
    )

    # Create the line plot for best values
    line = (
        alt.Chart(df_best_optuna_trails)
        .mark_line(color="green")
        .encode(x="trial:Q", y=alt.Y("value:Q", scale=alt.Scale(type="log")))
    )

    # Combine the scatter and line plots
    chart_optuna_best_value = (scatter + line).properties(title="Optuna Trials: Object Value and Best Value")

    mo.vstack(
        [mo.hstack([dropdown_exp_id, dropdown_seq_id]), chart_optuna_best_value],
        align="center",
    )
    return (
        chart_optuna_best_value,
        df_best_optuna_trails,
        df_optuna_filtered,
        get_best_optuna_trials,
        line,
        scatter,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Optuna hyperparameters""")
    return


@app.cell(hide_code=True)
def _(
    alt,
    df_optuna,
    dropdown_exp_id,
    dropdown_model,
    dropdown_seq_id,
    mo,
    pl,
):
    df_optuna_hp = df_optuna.filter(
        (pl.col("experiment_id") == int(dropdown_exp_id.value))
        & (pl.col("task_id") == int(dropdown_seq_id.value))
        & (pl.col("model_type") == dropdown_model.value)
    )

    param_list = df_optuna_hp.select(pl.col("hyper_param")).unique().to_series().to_list()
    param_list = sorted(param_list)
    num_groups = len(param_list)
    plots_per_row = 2
    num_rows = (num_groups // plots_per_row) + (num_groups % plots_per_row > 0)

    base_optuna_hp = (
        alt.Chart(df_optuna_hp.to_pandas())  # convert to pandas for Altair
        .mark_point()
        .encode(
            x=alt.X("param_value:Q", title="Parameter Value"),
            y=alt.Y("value:Q", title="Target Metric"),
            color=alt.Color(
                "trial:Q",
                scale=alt.Scale(scheme="blues"),
                legend=alt.Legend(title="Trial"),
            ),
            tooltip=["hyper_param", "param_value", "value", "trial"],
        )
    )

    charts_optuna_hp = alt.vconcat()
    for row in range(num_rows):
        row_charts = alt.hconcat()
        for col in range(plots_per_row):
            idx = row * plots_per_row + col
            if idx < num_groups:
                param = param_list[idx]
                chart_optuna_hp = base_optuna_hp.transform_filter(alt.datum.hyper_param == param).properties(
                    title=param, width=300, height=200
                )
                row_charts |= chart_optuna_hp
        charts_optuna_hp &= row_charts

    final_chart_optuna_hp = charts_optuna_hp.resolve_scale(color="independent")

    mo.vstack(
        [
            mo.hstack([dropdown_exp_id, dropdown_seq_id, dropdown_model]),
            final_chart_optuna_hp,
        ],
        align="center",
    )
    return (
        base_optuna_hp,
        chart_optuna_hp,
        charts_optuna_hp,
        col,
        df_optuna_hp,
        final_chart_optuna_hp,
        idx,
        num_groups,
        num_rows,
        param,
        param_list,
        plots_per_row,
        row,
        row_charts,
    )


if __name__ == "__main__":
    app.run()
