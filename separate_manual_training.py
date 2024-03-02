"""Manual Training."""

from pathlib import Path

import optuna
from sklearn.linear_model import ARDRegression
from sklearn.metrics import make_scorer, mean_absolute_error
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline

from octopus.experiment import OctoExperiment

# from sklearn.preprocessing import Normalizer

# from sklearn.preprocessing import Normalizer


# random.seed(42)
# np.random.seed(42)

path_study = Path("./studies/20240214A_Martin_wf2_octofull_7x6_global_ardreg_single3/")

path_experiment = path_study.joinpath("experiment3", "sequence0", "exp3_0.pkl")

experiment = OctoExperiment.from_pickle(path_experiment)


x_train = experiment.data_traindev[experiment.feature_columns]
y_train = (
    experiment.data_traindev[experiment.target_assignments.values()].to_numpy().ravel()
)


x_test = experiment.data_test[experiment.feature_columns]
y_test = experiment.data_test[experiment.target_assignments.values()]


# Define the objective function for Optuna optimization
def objective(trial):
    """Optuna objective function."""
    # alpha = trial.suggest_float("alpha", 1e-5, 1e5, log=True)
    # fit_intercept = trial.suggest_categorical("fit_intercept", [True, False])
    tol = trial.suggest_float("tol", 1e-5, 1e-1, log=True)
    alpha_1 = trial.suggest_float("alpha_1", 1e-10, 1e-3, log=True)
    alpha_2 = trial.suggest_float("alpha_2", 1e-10, 1e-3, log=True)
    lambda_1 = trial.suggest_float("lambda_1", 1e-10, 1e-3, log=True)
    lambda_2 = trial.suggest_float("lambda_2", 1e-10, 1e-3, log=True)
    threshold_lambda = trial.suggest_float("threshold_lambda", 1e3, 1e5, log=True)
    fit_intercept = trial.suggest_categorical("fit_intercept", [True, False])

    # Create the pipeline
    pipeinner = Pipeline(
        [
            # ('scaler', StandardScaler()),
            # ("normalizer", Normalizer()),
            # ("ridge", Ridge(alpha=alpha, fit_intercept=fit_intercept, solver="svd")),
            (
                "ardreg",
                ARDRegression(
                    tol=tol,
                    alpha_1=alpha_1,
                    alpha_2=alpha_2,
                    lambda_1=lambda_1,
                    lambda_2=lambda_2,
                    threshold_lambda=threshold_lambda,
                    fit_intercept=fit_intercept,  # ,
                ),
            ),
        ]
    )

    # Define the cross-validation strategy
    kfold = KFold(n_splits=6, shuffle=True, random_state=5)

    # Specify the MAE scorer
    scorer = make_scorer(mean_absolute_error)

    # Perform cross-validation with MAE
    cross_val_scores = cross_val_score(
        pipeinner, x_train, y_train, cv=kfold, scoring=scorer
    )

    # Calculate the mean MAE score
    score = cross_val_scores.mean()
    print("Objective score:", score)
    print("Objective values:", cross_val_scores)

    return score


# Run the Optuna optimization
sampler = optuna.samplers.TPESampler(multivariate=True, group=True, seed=0)
study = optuna.create_study(direction="minimize", sampler=sampler)

study.enqueue_trial(
    {
        "tol": 1e-3,
        "alpha_1": 1e-6,
        "alpha_2": 1e-6,
        "lambda_1": 1e-6,
        "lambda_2": 1e-6,
        "threshold_lambda": 1e4,
        "fit_intercept": True,
    }
)
study.optimize(objective, n_trials=15, n_jobs=1)

# Print the best hyperparameters and best score
best_score = study.best_value
best_params = study.best_params
print("Best score: ", best_score)
print("Best hyperparameters: ", best_params)

# Create the pipeline with the best hyperparameters
best_alpha1 = study.best_params["alpha_1"]
best_alpha2 = study.best_params["alpha_2"]
best_lambda1 = study.best_params["lambda_1"]
best_lambda2 = study.best_params["lambda_2"]
best_tol = study.best_params["tol"]
best_threshold_lambda = study.best_params["threshold_lambda"]
best_fit_intercept = study.best_params["fit_intercept"]

pipe = Pipeline(
    [
        # ('scaler', StandardScaler()),
        # ("normalizer", Normalizer()),
        # ("ridge", Ridge(alpha=alpha, fit_intercept=fit_intercept, solver="svd")),
        (
            "ardreg",
            ARDRegression(
                tol=best_tol,
                alpha_1=best_alpha1,
                alpha_2=best_alpha2,
                lambda_1=best_lambda1,
                lambda_2=best_lambda2,
                threshold_lambda=best_threshold_lambda,
                fit_intercept=best_fit_intercept,
            ),
        ),
    ]
)
# Fit the pipeline on the training data
# ! Watch out - this is a refit!
pipe.fit(x_train, y_train)

# Evaluate the model on the test data
y_pred_train = pipe.predict(x_train)
y_pred_test = pipe.predict(x_test)

print("Mean Absolute Error on train set: ", mean_absolute_error(y_train, y_pred_train))
print("Mean Absolute Error on test set: ", mean_absolute_error(y_test, y_pred_test))
