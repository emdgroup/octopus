"""Model parameters.

"default": default for optuna model parameter settings
"translate": dictionary to translate global parameters into
model specific parameters
"""

# ExtraTree Classifier parameter settings
extratree_class = {
    "default": [
        ("int", {"name": "max_depth", "low": 2, "high": 32}),
        ("int", {"name": "min_samples_split", "low": 2, "high": 100}),
        ("int", {"name": "min_samples_leaf", "low": 1, "high": 50}),
        ("float", {"name": "max_features", "low": 0.1, "high": 1}),
        ("int", {"name": "n_estimators", "low": 100, "high": 500, "log": False}),
        ("categorical", {"name": "class_weight", "choices": [None, "balanced"]}),
        ("fixed", {"name": "criterion", "value": "entropy"}),
    ],
    "translate": {
        "n_jobs": "n_jobs",
        "model_seed": "random_state",
    },
}

# ExtraTree Regressor parameter settings
extratree_reg = {
    "default": [
        ("int", {"name": "max_depth", "low": 2, "high": 32}),
        ("int", {"name": "min_samples_split", "low": 2, "high": 100}),
        ("int", {"name": "min_samples_leaf", "low": 1, "high": 50}),
        ("float", {"name": "max_features", "low": 0.1, "high": 1}),
        ("int", {"name": "n_estimators", "low": 100, "high": 500, "log": False}),
    ],
    "translate": {
        "n_jobs": "n_jobs",
        "model_seed": "random_state",
    },
}

# Random Forest Classifier parameter settings
rf_class = {
    "default": [
        ("int", {"name": "max_depth", "low": 2, "high": 32}),
        ("int", {"name": "min_samples_split", "low": 2, "high": 100}),
        ("int", {"name": "min_samples_leaf", "low": 1, "high": 50}),
        ("float", {"name": "max_features", "low": 0.1, "high": 1}),
        ("int", {"name": "n_estimators", "low": 100, "high": 500, "log": False}),
        ("categorical", {"name": "class_weight", "choices": [None, "balanced"]}),
    ],
    "translate": {
        "n_jobs": "n_jobs",
        "model_seed": "random_state",
    },
}

# Random Forest Regressor parameter settings
rf_reg = {
    "default": [
        ("int", {"name": "max_depth", "low": 2, "high": 32}),
        ("int", {"name": "min_samples_split", "low": 2, "high": 100}),
        ("int", {"name": "min_samples_leaf", "low": 1, "high": 50}),
        ("float", {"name": "max_features", "low": 0.1, "high": 1}),
        ("int", {"name": "n_estimators", "low": 100, "high": 500, "log": False}),
    ],
    "translate": {
        "n_jobs": "n_jobs",
        "model_seed": "random_state",
    },
}

# Random Forest Classifier parameter settings
xgboost_class = {
    "default": [
        ("float", {"name": "learning_rate", "low": 1e-4, "high": 0.3, "log": True}),
        ("int", {"name": "min_child_weight", "low": 2, "high": 15}),
        ("float", {"name": "subsample", "low": 0.15, "high": 1.0}),
        ("int", {"name": "n_estimators", "low": 30, "high": 200}),
        ("int", {"name": "max_depth", "low": 3, "high": 9, "step": 2}),
        ("fixed", {"name": "validate_parameters", "value": True}),
        # ("float", {"name": "lambda", "low": 1e-8, "high": 1, "log": True}),
        # missing: pos_class_weight
    ],
    "translate": {
        "n_jobs": "n_jobs",
        "model_seed": "random_state",
    },
}

# Random Forest Regressor parameter settings
xgboost_reg = {
    "default": [
        ("float", {"name": "learning_rate", "low": 1e-4, "high": 0.3, "log": True}),
        ("int", {"name": "min_child_weight", "low": 2, "high": 15}),
        ("float", {"name": "subsample", "low": 0.15, "high": 1.0}),
        ("int", {"name": "n_estimators", "low": 30, "high": 500}),
        ("int", {"name": "max_depth", "low": 3, "high": 9, "step": 2}),
        ("fixed", {"name": "validate_parameters", "value": True}),
        # ("float", {"name": "lambda", "low": 1e-8, "high": 1, "log": True}),
    ],
    "translate": {
        "n_jobs": "n_jobs",
        "model_seed": "random_state",
    },
}

# Linear ridge regression parameter settings
ridge_reg = {
    "default": [
        ("float", {"name": "alpha", "low": 1e-5, "high": 1e5, "log": True}),
        ("categorical", {"name": "fit_intercept", "choices": [True, False]}),
        ("fixed", {"name": "solver", "value": "svd"}),
    ],
    "translate": {
        "n_jobs": "NA",  # NA=ignore, model does not support this key
        "model_seed": "random_state",
    },
}

# Linear ridge regression parameter settings
ard_reg = {
    "default": [
        ("float", {"name": "alpha_1", "low": 1e-10, "high": 1e-3, "log": True}),
        ("float", {"name": "alpha_2", "low": 1e-10, "high": 1e-3, "log": True}),
        ("float", {"name": "lambda_1", "low": 1e-10, "high": 1e-3, "log": True}),
        ("float", {"name": "lambda_2", "low": 1e-10, "high": 1e-3, "log": True}),
        ("float", {"name": "threshold_lambda", "low": 1e3, "high": 1e5, "log": True}),
        ("float", {"name": "tol", "low": 1e-5, "high": 1e-1, "log": True}),
        ("fixed", {"name": "fit_intercept", "value": True}),
        # ("categorical", {"name": "fit_intercept", "choices": [True, False]}),
        # ("fixed", {"name": "copy_X", "value": False}),
    ],
    "translate": {
        "n_jobs": "NA",  # NA=ignore, model does not support this key
        "model_seed": "NA",  # NA=ignore, model does not support this key
    },
}

# Linear ridge regression parameter settings
elastic_reg = {
    "default": [
        ("float", {"name": "alpha", "low": 1e-10, "high": 1e2, "log": True}),
        ("float", {"name": "l1_ratio", "low": 0, "high": 1, "log": False}),
        ("categorical", {"name": "fit_intercept", "choices": [True, False]}),
        ("float", {"name": "tol", "low": 1e-5, "high": 1e-1, "log": True}),
        ("fixed", {"name": "max_iter", "value": 4000}),
        ("fixed", {"name": "selection", "value": "random"}),
    ],
    "translate": {
        "n_jobs": "NA",  # NA=ignore, model does not support this key
        "model_seed": "random_state",
    },
}

# ExtraTree Classifier parameter settings
extratrees_surv = {
    "default": [
        ("int", {"name": "max_depth", "low": 2, "high": 32}),
        ("int", {"name": "min_samples_split", "low": 2, "high": 100}),
        ("int", {"name": "min_samples_leaf", "low": 1, "high": 50}),
        ("float", {"name": "max_features", "low": 0.1, "high": 1}),
        ("int", {"name": "n_estimators", "low": 100, "high": 500, "log": False}),
    ],
    "translate": {
        "n_jobs": "n_jobs",
        "model_seed": "random_state",
    },
}


## GradientBoostingRegressor
gb_reg = {
    "default": [
        ("float", {"name": "learning_rate", "low": 0.01, "high": 1, "log": True}),
        ("int", {"name": "min_samples_leaf", "low": 1, "high": 200}),
        ("int", {"name": "max_leaf_nodes", "low": 3, "high": 2047}),
        ("int", {"name": "max_depth", "low": 3, "high": 9, "step": 2}),
        ("int", {"name": "n_estimators", "low": 30, "high": 500}),
        ("float", {"name": "max_features", "low": 0.1, "high": 1}),
        ("fixed", {"name": "loss", "value": "squared_error"}),
    ],
    "translate": {
        "n_jobs": "NA",
        "model_seed": "random_state",
    },
}

svr_reg = {
    "default": [
        ("float", {"name": "C", "low": 0.03125, "high": 32768, "log": True}),
        ("float", {"name": "epsilon", "low": 0.001, "high": 1, "log": True}),
        ("float", {"name": "tol", "low": 1e-5, "high": 1e-1, "log": True}),
    ],
    "translate": {
        "n_jobs": "NA",
        "model_seed": "NA",
    },
}

parameters_inventory = {
    "ExtraTreesClassifier": extratree_class,
    "ExtraTreesRegressor": extratree_reg,
    "RandomForestClassifier": rf_class,
    "RandomForestRegressor": rf_reg,
    "XGBClassifier": xgboost_class,
    "XGBRegressor": xgboost_reg,
    "RidgeRegressor": ridge_reg,
    "ARDRegressor": ard_reg,
    "ElasticNetRegressor": elastic_reg,
    "ExtraTreesSurv": extratrees_surv,
    "GradientBoostingRegressor": gb_reg,
    "SvrRegressor": svr_reg,
}
