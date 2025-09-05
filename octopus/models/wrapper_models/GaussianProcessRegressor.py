"""Wrapper for Gaussian Process Regressor."""

from collections.abc import Callable
from typing import Any

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Kernel, Matern, RationalQuadratic
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


class GPRegressorWrapper(BaseEstimator, RegressorMixin):
    """Wrapper for Gaussian Process Regressor."""

    def __init__(
        self,
        kernel: str | Kernel = "RBF",
        alpha: float = 1e-10,
        optimizer: str | Callable | None = "fmin_l_bfgs_b",
        n_restarts_optimizer: int = 0,
        normalize_y: bool = False,
        copy_X_train: bool = True,
        random_state: int | None = None,
    ) -> None:
        self.kernel = kernel
        self.alpha = alpha
        self.optimizer = optimizer
        self.n_restarts_optimizer = n_restarts_optimizer
        self.normalize_y = normalize_y
        self.copy_X_train = copy_X_train
        self.random_state = random_state

    def fit(self, X: Any, y: Any) -> "GPRegressorWrapper":
        """Fit the Gaussian Process model.

        Args:
            X: Training data.
            y: Target values.

        Returns:
            Fitted model.
        """
        X, y = check_X_y(X, y)
        kernel = self._get_kernel(self.kernel)
        self.model_ = GaussianProcessRegressor(
            kernel=kernel,
            alpha=self.alpha,
            optimizer=self.optimizer,
            n_restarts_optimizer=self.n_restarts_optimizer,
            normalize_y=self.normalize_y,
            copy_X_train=self.copy_X_train,
            random_state=self.random_state,
        )
        self.model_.fit(X, y)
        return self

    def predict(self, X: Any) -> Any:
        """Predict using the Gaussian Process model.

        Args:
            X: Input data.

        Returns:
            Predicted values.
        """
        check_is_fitted(self, "model_")
        X = check_array(X)
        return self.model_.predict(X)

    def _get_kernel(self, kernel_str: str | Kernel) -> Kernel:
        """Get the kernel object based on the kernel string.

        Args:
            kernel_str: Kernel string or object.

        Returns:
            Kernel object.

        Raises:
            ValueError: If any kernel is not known.

        """
        if isinstance(kernel_str, Kernel):
            return kernel_str
        elif kernel_str == "RBF":
            return RBF()
        elif kernel_str == "Matern":
            return Matern()
        elif kernel_str == "RationalQuadratic":
            return RationalQuadratic()
        else:
            raise ValueError(f"Unknown kernel: {kernel_str}")

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Get parameters for this estimator.

        Args:
            deep: Whether to return the parameters for this estimator and
                contained subobjects.

        Returns:
            Parameter names mapped to their values.
        """
        return {
            "kernel": self.kernel,
            "alpha": self.alpha,
            "optimizer": self.optimizer,
            "n_restarts_optimizer": self.n_restarts_optimizer,
            "normalize_y": self.normalize_y,
            "copy_X_train": self.copy_X_train,
            "random_state": self.random_state,
        }

    def set_params(self, **params: Any) -> "GPRegressorWrapper":
        """Set the parameters of this estimator.

        Args:
            **params: Estimator parameters.

        Returns:
            Estimator instance.
        """
        for param, value in params.items():
            setattr(self, param, value)
        return self
