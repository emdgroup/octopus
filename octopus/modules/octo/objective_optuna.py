""""Objective function for optuna optimization."""

import heapq

# from octopus.models.parameters import parameters_inventory
# from octopus.models.utils import create_trialparams_from_config
from octopus.models.machine_learning.core import model_inventory
from octopus.modules.octo.bag import Bag
from octopus.modules.octo.training import Training
from octopus.modules.utils import optuna_direction


class ObjectiveOptuna:
    """Callable optuna objective.

    A single solution for global and individual HP optimizations.
    """

    def __init__(self, experiment, data_splits, study_name, top_trials):
        self.experiment = experiment
        self.data_splits = data_splits
        self.study_name = study_name
        self.top_trials = top_trials
        # saving trials
        self.ensel = self.experiment.ml_config.ensemble_selection
        self.n_save_trials = self.experiment.ml_config.ensel_n_save_trials
        # parameters potentially used for optimizations
        self.ml_model_types = self.experiment.ml_config.models
        self.dim_red_methods = self.experiment.ml_config.dim_red_methods
        self.max_outl = self.experiment.ml_config.max_outl
        self.max_features = self.experiment.ml_config.max_features
        self.penalty_factor = self.experiment.ml_config.penalty_factor
        self.hyper_parameters = self.experiment.ml_config.hyperparameters
        # fixed parameters
        self.ml_seed = self.experiment.ml_config.model_seed
        self.ml_jobs = self.experiment.ml_config.n_jobs
        # training parameters
        self.parallel_execution = self.experiment.ml_config.inner_parallelization
        self.num_workers = self.experiment.ml_config.n_workers

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

        # get hyperparameters for selected model
        model_item = model_inventory.get_model_by_name(ml_model_type)

        if ml_model_type in self.hyper_parameters.keys():
            hyperparameters = self.hyper_parameters[ml_model_type]
        else:
            model_item.hyperparameters

        model_params = model_inventory.create_optuna_parameters(
            trial,
            ml_model_type,
            hyperparameters,
            model_item.translate,
            {
                "n_jobs": self.ml_jobs,
                "model_seed": self.ml_seed,
            },
        )

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
                    target_metric=self.experiment.configs.study.target_metric,
                    max_features=self.experiment.ml_config.max_features,
                    feature_groups=self.experiment.feature_groups,
                )
            )

        # create bag with all provided trainings
        bag_trainings = Bag(
            bag_id=self.experiment.id + "_" + str(trial.number),
            trainings=trainings,
            target_assignments=self.experiment.target_assignments,
            parallel_execution=self.parallel_execution,
            num_workers=self.num_workers,
            target_metric=self.experiment.configs.study.target_metric,
            row_column=self.experiment.row_column,
            # path?
        )

        # train all models in bag
        # print("config_training", config_training)
        bag_trainings.fit()

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
        print(f"Trial scores for metric: {self.experiment.configs.study.target_metric}")
        for key, value in scores.items():
            if isinstance(value, list):
                print(f"{key}:{value}")
            else:
                print(f"{key}:{value:.3f}")

        # define optuna target
        optuna_target = scores["dev_avg"]

        # adjust direction, optuna in octofull always minimizes
        if optuna_direction(self.experiment.configs.study.target_metric) == "maximize":
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

        # save bag if we plan to run ensemble selection
        if self.ensel:
            self._save_topn_trials(bag_trainings, optuna_target, trial.number)

        print("Otarget:", optuna_target)
        print("Number of features used:", int(n_features_mean))

        return optuna_target

    def _save_topn_trials(self, bag, target_value, n_trial):
        max_n_trials = self.experiment.ml_config.ensel_n_save_trials
        path_save = self.experiment.path_study.joinpath(
            self.experiment.path_sequence_item,
            "trials",
            f"study{self.study_name}trial{n_trial}_bag.pkl",
        )

        # saving top n_trials to disk
        # the optuna target_value will always be minimized. Heappop removes the lowest
        # value, therefore target_value needs to be negated.
        heapq.heappush(self.top_trials, (-1 * target_value, path_save))
        bag.to_pickle(path_save)
        if len(self.top_trials) > max_n_trials:
            # delete trial with lowest perfomrmance in n_trials
            _, path_delete = heapq.heappop(self.top_trials)
            if path_delete.is_file():
                path_delete.unlink()
            else:
                raise FileNotFoundError("Problem deleting trial-pkl file")
