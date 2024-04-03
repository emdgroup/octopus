""""Objective function for optuna optimization."""

from octopus.models.parameters import parameters_inventory
from octopus.models.utils import create_trialparams_from_config
from octopus.modules.octo.bag import Bag
from octopus.modules.octo.training import Training
from octopus.modules.utils import optuna_direction


class ObjectiveOptuna:
    """Callable optuna objective.

    A single solution for global and individual HP optimizations.
    """

    def __init__(
        self,
        experiment,
        data_splits,
        study_name,
        save_trials,
    ):
        self.experiment = experiment
        self.data_splits = data_splits
        self.study_name = study_name
        self.save_trials = save_trials
        # parameters potentially used for optimizations
        self.ml_model_types = self.experiment.ml_config["models"]
        self.dim_red_methods = self.experiment.ml_config["dim_red_methods"]
        self.max_outl = self.experiment.ml_config["max_outl"]
        self.max_features = self.experiment.ml_config["max_features"]
        self.penalty_factor = self.experiment.ml_config["penalty_factor"]
        # fixed parameters
        self.ml_seed = self.experiment.ml_config["model_seed"]
        self.ml_jobs = self.experiment.ml_config["n_jobs"]
        # training parameters
        self.parallel_execution = self.experiment.ml_config["inner_parallelization"]
        self.num_workers = self.experiment.ml_config["n_workers"]

    def __call__(self, trial):
        """Call.

        We have different types of parameters:
        (a) non-model parameters that are needed in
            the training
        (b) model parameters that are varied by optuna
            (defined by default or optuna_model_settings)
        (c) global parameters that have to be translated
            in fixed model specific parameters
        """
        # get non-model parameters
        # (1) dimension reduction
        if len(self.dim_red_methods) > 1:
            dim_reduction = trial.suggest_categorical(
                name="dim_red_method", choices=self.dim_red_methods
            )
        else:
            dim_reduction = self.dim_red_methods[0]

        # (2) ml_model_type
        if len(self.ml_model_types) > 1:
            ml_model_type = trial.suggest_categorical(
                name="ml_model_type", choices=self.ml_model_types
            )
        else:
            ml_model_type = self.ml_model_types[0]

        # (3) number of outliers to be detected
        if self.max_outl > 0:
            num_outl = trial.suggest_int(name="num_outl", low=0, high=self.max_outl)
        else:
            num_outl = 0

        # get model parameters
        optuna_model_settings = None  # use default
        settings_default = parameters_inventory[ml_model_type]["default"]

        if optuna_model_settings is None:
            # use default model parameter settings
            model_params = create_trialparams_from_config(
                trial, settings_default, ml_model_type
            )
        else:
            # use model parameter settings as provided by config
            model_params = create_trialparams_from_config(
                trial, optuna_model_settings, ml_model_type
            )

        # overwrite model parameters specified by global settings
        fixed_global_parameters = {
            "n_jobs": self.ml_jobs,
            "model_seed": self.ml_seed,
        }
        translate = parameters_inventory[ml_model_type]["translate"]
        for key, value in fixed_global_parameters.items():
            if translate[key] != "NA":  # NA=ignore
                model_params[translate[key]] = value

        config_training = {
            "dim_reduction": dim_reduction,
            "outl_reduction": num_outl,
            "ml_model_type": ml_model_type,
            "ml_model_params": model_params,
        }

        # create trainings
        trainings = list()
        for key, split in self.data_splits.items():
            trainings.append(
                Training(
                    training_id=self.experiment.id + "_" + str(key),
                    ml_type=self.experiment.ml_type,
                    target_assignments=self.experiment.target_assignments,
                    feature_columns=self.experiment.feature_columns,
                    row_column=self.experiment.row_column,
                    data_train=split["train"],  # inner datasplit, train
                    data_dev=split["test"],  # inner datasplit, dev
                    data_test=self.experiment.data_test,
                    config_training=config_training,
                    target_metric=self.experiment.config["target_metric"],
                    max_features=self.experiment.ml_config["max_features"],
                )
            )

        # create bag with all provided trainings
        bag_trainings = Bag(
            bag_id=self.experiment.id + "_" + str(trial),
            trainings=trainings,
            target_assignments=self.experiment.target_assignments,
            parallel_execution=self.parallel_execution,
            num_workers=self.num_workers,
            target_metric=self.experiment.config["target_metric"],
            row_column=self.experiment.row_column,
            # path?
        )

        # train all models in bag
        bag_trainings.fit()

        # save bag if desired
        if self.save_trials:
            path_save = self.experiment.path_study.joinpath(
                self.experiment.path_sequence_item,
                "trials",
                f"study{self.study_name}trial{trial.number}_bag.pkl",
            )
            bag_trainings.to_pickle(path_save)

        # evaluate trainings using target metric
        scores = bag_trainings.get_scores()

        # get number of features used in bag
        n_features_mean = bag_trainings.n_features_used_mean

        # add scores info to the optuna trial
        for key, value in scores.items():
            trial.set_user_attr(key, value)

        # add config_training to user attributes
        trial.set_user_attr("config_training", config_training)

        # print scores info to console
        print(f"Trial scores for metric: {self.experiment.config['target_metric']}")
        for key, value in scores.items():
            if isinstance(value, list):
                print(f"{key}:{value}")
            else:
                print(f"{key}:{value:.3f}")

        # define optuna target
        optuna_target = scores["dev_avg"]

        # adjust direction, optuna in octofull always minimizes
        if optuna_direction(self.experiment.config["target_metric"]) == "maximize":
            optuna_target = -optuna_target

        # add penaltiy for n_features > max_features if configured
        if self.max_features > 0:
            diff_nfeatures = n_features_mean - self.max_features
            # only consider if n_features_mean > max_features
            if diff_nfeatures < 0:
                diff_nfeatures = 0
            n_features = len(self.experiment.feature_columns)
            optuna_target = (
                optuna_target + self.penalty_factor * diff_nfeatures / n_features
            )

        print("Otarget:", optuna_target)
        print("Number of features used:", int(n_features_mean))

        return optuna_target
