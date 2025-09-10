"""Test basic classification workflow."""

# import os

# import pandas as pd

# from octopus import OctoData, OctoML
# from octopus.config import ConfigManager, ConfigSequence, ConfigStudy
# from octopus.modules import Octo


# def test_basic_classification():
#     """Test basic regression."""
#     data_df = (
#         pd.read_csv(
#             os.path.join(os.getcwd(), "datasets", "titanic_openml.csv"), index_col=0
#         )
#         .astype({"age": float})
#         .assign(
#             age=lambda df_: df_["age"].fillna(df_["age"].median()).astype(int),
#             embarked=lambda df_: df_["embarked"].fillna(df_["embarked"].mode()[0]),
#             fare=lambda df_: df_["fare"].fillna(df_["fare"].median()),
#         )
#         .astype({"survived": bool})
#         .pipe(
#             lambda df_: df_.reindex(
#        columns=["survived"] + list([a for a in df_.columns if a != "survived"])
#             )
#         )
#         .pipe(
#             lambda df_: df_.reindex(
#                 columns=["name"] + list([a for a in df_.columns if a != "name"])
#             )
#         )
#         .pipe(pd.get_dummies, columns=["embarked", "sex"], drop_first=True, dtype=int)
#         .loc[0:100, :]
#     )

#     octo_data = OctoData(
#         data=data_df,
#         target_columns=["survived"],
#         feature_columns=[
#             "pclass",
#             "age",
#             "sibsp",
#             "parch",
#             "fare",
#             "embarked_Q",
#             "embarked_S",
#             "sex_male",
#         ],
#         sample_id="name",
#         datasplit_type="group_sample_and_features",
#         stratification_column="survived",
#     )

#     config_study = ConfigStudy(
#         name="test_basic_classification",
#         ml_type="classification",
#         target_metric="AUCROC",
#         silently_overwrite_study=True,
#         ignore_data_health_warning=True,
#     )

#     config_manager = ConfigManager(
#         outer_parallelization=False, run_single_experiment_num=0
#     )

#     config_sequence = ConfigSequence(
#         sequence_items=[
#             Octo(
#                 sequence_id=0,
#                 input_sequence_id=-1,
#                 description="step_1_octo",
#                 models=["RandomForestClassifier"],
#                 n_trials=1,
#             )
#         ]
#     )

#     octo_ml = OctoML(
#         octo_data,
#         config_study=config_study,
#         config_manager=config_manager,
#         config_sequence=config_sequence,
#     )
#     octo_ml.run_study()
#     success = True
#     assert success is True
