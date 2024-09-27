"""RFE2 core function."""

from attrs import define

from octopus.modules.octo.core import OctoCore

# RF2 TOBEDONE:
#  - train bag in standard octo way
#  - give bag and hyperparameters to rfe-process
#  - (start) train bag und standard hyperparameters
#  - calculate group_pfi and shap
# - remove least important feature
#   -- automatically remove not used features
#   -- deal with group features, needs to be intelligent
#   -- deal with negative feature importances
#   -- consider count information
# - record performance
# - go back to start and repeat
# - select the model, different approaches
#   -- persimonial
#   -- best model
# - model retraining after n removal, or start use module several times
# - autogluon: add 3-5 random feature and remove all feature below the lowest random
# - rewrite using iheritance from OctoCore, overwrite run_experiment


@define
class Rfe2Core(OctoCore):
    """Rfe2 Core."""

    @property
    def config(self) -> dict:
        """Module configuration."""
        return self.experiment.ml_config

    def run_experiment(self):
        """Run experiment."""
        # (1) train and optimize model
        self._optimize_splits(self.data_splits)

        # create best bag in results directory
        # - attach best bag to experiment
        # - attach best bag scores to experiment
        self._create_best_bag()

        # (2) run RFE
        print("config:", self.config)

        # model should be found here: self.experiment.results["best"]

        return self.experiment
