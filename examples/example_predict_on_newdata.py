"""Workflow example for prediction on new data."""

from pathlib import Path

from octopus.predict import OctoPredict

# setup

path_study = Path("./studies/20240110B/")
study = OctoPredict(path_study)

# (A) predict on internally available test data
print(study.predict_proba_test())

# (B) predict on new data
# - data needs to be in pd.dataframe
# - for classification use study.predict_proba
# - for everything else use study.predict
# this generate the ensembling predictions using
# the final models from all available experiments
# data_df = pd.DataFrame()
# print(study.predict_proba(data_df, return_df=True))
# print(study.predict_proba(data_df))


# (C) calculate permutation feature importances using final models (bag)
#     on test data
study.calculate_fi_test(fi_type="permutation")
# - fi tables are saved in the  study.results dictionary
# - pdf plots are saved in the results directory

# (D) calculate shap feature importances using final models (bag)
#     on test data
study.calculate_fi_test(fi_type="shap", shap_type="exact")
# - shap_type could be ["exact", "permutation"]
# - shap_type "exact" does not scale well with number of features
# - shap_type "permutation" scales better than "exact" but
#   takes longer for a small number of features
# - fi tables are saved in the  study.results dictionary
# - pdf plots are saved in the results directory


# (E) calculate feature importances using final models (bag)
#     on new data
# study.calculate_fi(data_df, fi_type="permutation")
# study.calculate_fi(data_df, fi_type="shap")
# - fi tables are saved in the  study.results dictionary
# - pdf plots are saved in the results directory
