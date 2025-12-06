from __future__ import annotations

from .src.base import TreeClassifier

from typing import List, Optional
from numpy.typing import ArrayLike, NDArray
import numpy as np
from math import comb
import warnings


def formatwarning(message, category, filename, lineno, line=None, **kwargs):
    return f"UserWarning: {message}\n"


warnings.formatwarning = formatwarning


class BaseTree(TreeClassifier):
    """
    Base class for decision tree classifiers and regressors.

    This class provides foundational functionality for building decision trees,
    including parameter validation, data preprocessing, and interfacing with the
    underlying `TreeClassifier`. It handles both classification and regression
    tasks based on the `task` parameter.

    Parameters
    ----------
    task : bool
        - If `True`, construct regression tree.
        - If `False`, construct classification tree.

    max_depth : int
        Maximum depth of the tree. Controls model complexity and prevents overfitting.

        - If `-1`: Expands until leaves are pure or contain fewer than `min_samples_split` samples.
        - If `int > 0`: Limits the tree to the specified depth.

    min_samples_leaf : int
        Minimum number of samples required at leaf nodes.

    min_samples_split : int
        Minimum number of samples required to split an internal node.

    min_impurity_decrease : float
        Minimum required decrease in impurity to create a split.

    ccp_alpha : float
        Complexity parameter for Minimal Cost-Complexity Pruning.

    categories : List[int]
        Indices of categorical features in the dataset.

    use_oblique : bool
        - If `True`, enables oblique splits using linear combinations of features.
        - If `False`, uses traditional axis-aligned splits only.

    random_state : int
        Seed for random number generation in oblique splits.

        - Only used when `use_oblique=True`.

    n_pair : int
        Number of features to combine in oblique splits.

        - Only used when `use_oblique=True`.

    gamma : float
        Separation strength parameter for oblique splits.

        - Only used when `use_oblique=True`.

    max_iter : int
        Maximum iterations for L-BFGS optimization in oblique splits.

        - Only used when `use_oblique=True`.

    relative_change : float
        Early stopping threshold for L-BFGS optimization.

        - Only used when `use_oblique=True`.
    """

    def __init__(
        self,
        task: bool,
        max_depth: int,
        min_samples_leaf: int,
        min_samples_split: int,
        min_impurity_decrease: float,
        ccp_alpha: float,
        categories: Optional[List[int]],
        use_oblique: bool,
        random_state: Optional[int],
        n_pair: int,
        gamma: float,
        max_iter: int,
        relative_change: float,
    ) -> None:
        # Validate and assign parameters
        self.task = task
        self.use_oblique = self._validate_use_oblique(use_oblique)
        self.max_depth = self._validate_max_depth(max_depth)
        self.min_samples_leaf = self._validate_min_samples_leaf(min_samples_leaf)
        self.min_samples_split = self._validate_min_samples_split(min_samples_split)
        self.min_impurity_decrease = self._validate_min_impurity_decrease(
            min_impurity_decrease
        )
        self.ccp_alpha = self._validate_ccp_alpha(ccp_alpha)
        self.n_pair = self._validate_n_pair(n_pair)
        self.gamma = self._validate_gamma(gamma)
        self.max_iter = self._validate_max_iter(max_iter)
        self.relative_change = self._validate_relative_change(
            relative_change, self.use_oblique
        )
        self.random_state = self._validate_random_state(random_state)
        self.categories = self._validate_categories(categories)
        self._fit = False
        self._categories: dict[int, NDArray]

        # Initialize the TreeClassifier
        super().__init__(
            self.max_depth,
            self.min_samples_leaf,
            self.min_samples_split,
            self.min_impurity_decrease,
            self.random_state,
            self.n_pair,
            self.gamma,
            self.max_iter,
            self.relative_change,
            self.categories,
            self.ccp_alpha,
            self.use_oblique,
            self.task,
            1,
        )

    def __getstate__(self):
        """Return the state for pickling."""
        state = super().__getstate__()
        state["_fit"] = self._fit

        return state

    def __setstate__(self, state):
        """Restore the state from pickle."""
        # Extract special attributes
        _fit = state.pop("_fit", False)
        super().__setstate__(state)

        # Restore state directly without re-initialization
        self.__dict__.update(state)
        self._fit = _fit

    def __repr__(self):
        param_str = (
            f"use_oblique={getattr(self, 'use_oblique', None)}, "
            f"max_depth={getattr(self, 'max_depth', None)}, "
            f"min_samples_leaf={getattr(self, 'min_samples_leaf', None)}, "
            f"min_samples_split={getattr(self, 'min_samples_split', None)}, "
            f"min_impurity_decrease={getattr(self, 'min_impurity_decrease', None)}, "
            f"ccp_alpha={getattr(self, 'ccp_alpha', None)}, "
            f"categories={getattr(self, 'categories', None)}, "
            f"random_state={getattr(self, 'random_state', None)}, "
            f"n_pair={getattr(self, 'n_pair', None)}, "
            f"gamma={getattr(self, 'gamma', None)}, "
            f"max_iter={getattr(self, 'max_iter', None)}, "
            f"relative_change={getattr(self, 'relative_change', None)}"
        )
        return f"{self.__class__.__name__}({param_str})"

    def _validate_max_depth(self, max_depth: int) -> int:
        if not isinstance(max_depth, int):
            raise ValueError("max_depth must be an integer")
        if max_depth < -1:
            raise ValueError("max_depth must be >= -1")
        return 255 if max_depth == -1 else min(max_depth, 255)

    def _validate_min_samples_leaf(self, min_samples_leaf: int) -> int:
        if not isinstance(min_samples_leaf, int):
            raise ValueError("min_samples_leaf must be an integer")
        if min_samples_leaf < 1:
            raise ValueError("min_samples_leaf must be >= 1")
        return min_samples_leaf

    def _validate_min_samples_split(self, min_samples_split: int) -> int:
        if not isinstance(min_samples_split, int):
            raise ValueError("min_samples_split must be an integer")
        if min_samples_split < 2:
            raise ValueError("min_samples_split must be >= 2")
        return min_samples_split

    def _validate_min_impurity_decrease(self, min_impurity_decrease: float) -> float:
        if not isinstance(min_impurity_decrease, (int, float)):
            raise ValueError("min_impurity_decrease must be a number")
        if min_impurity_decrease < 0.0:
            raise ValueError("min_impurity_decrease must be >= 0.0")
        return float(min_impurity_decrease)

    def _validate_ccp_alpha(self, ccp_alpha: float) -> float:
        if not isinstance(ccp_alpha, (int, float)):
            raise ValueError("ccp_alpha must be a number")
        if ccp_alpha < 0.0:
            raise ValueError("ccp_alpha must be >= 0.0")
        return float(ccp_alpha)

    def _validate_n_pair(self, n_pair: int) -> int:
        if not isinstance(n_pair, int):
            raise ValueError("n_pair must be an integer")
        if n_pair < 2:
            raise ValueError("n_pair must be >= 2")
        return n_pair

    def _validate_gamma(self, gamma: float) -> float:
        if not isinstance(gamma, (int, float)):
            raise ValueError("gamma must be a number")
        if gamma <= 0.0:
            raise ValueError("gamma must be > 0.0")
        return float(gamma)

    def _validate_max_iter(self, max_iter: int) -> int:
        if not isinstance(max_iter, int):
            raise ValueError("max_iter must be an integer")
        if max_iter < 1:
            raise ValueError("max_iter must be >= 1")
        return max_iter

    def _validate_relative_change(
        self, relative_change: float, use_oblique: bool
    ) -> float:
        if not isinstance(relative_change, (int, float)):
            raise ValueError("relative_change must be a number")
        if relative_change < 0.0:
            raise ValueError("relative_change must be >= 0.0")
        if use_oblique and relative_change <= 1e-5:
            warnings.warn(
                "relative_change is set very low. This may prolong the oblique training time."
            )
        return float(relative_change)

    def _validate_random_state(self, random_state: Optional[int]) -> int:
        if random_state is not None and not isinstance(random_state, int):
            raise ValueError("random_state must be None or an integer")
        return (
            random_state
            if random_state is not None
            else np.random.randint(0, np.iinfo(np.int32).max)
        )

    def _validate_categories(self, categories: Optional[List[int]]) -> List[int]:
        if categories is not None:
            if not isinstance(categories, (list, tuple)):
                raise ValueError("categories must be None or a list/tuple of integers")
            if not all(isinstance(x, int) for x in categories):
                raise ValueError("All elements in categories must be integers")
            if any(x < 0 for x in categories):
                raise ValueError(
                    "All elements in categories must be non-negative integers"
                )
            return list(categories)
        return []

    def _validate_use_oblique(self, use_oblique: bool) -> bool:
        if not isinstance(use_oblique, bool):
            raise ValueError("use_oblique must be a boolean")
        return use_oblique

    def fit(
        self, X: ArrayLike, y: ArrayLike, sample_weight: Optional[ArrayLike] = None
    ) -> "BaseTree":
        """
        Fit the decision tree to the training data.

        Parameters
        ----------
        X : ArrayLike
            Training input samples of shape (n_samples, n_features).
        y : ArrayLike
            Target values of shape (n_samples,).
        sample_weight : Optional[ArrayLike], default=None
            Sample weights of shape (n_samples,). If None, all samples are given equal weight.

        Returns
        -------
        self : BaseTree
            Fitted estimator.

        Raises
        ------
        ValueError
            If input data is invalid or contains NaN/Inf values where not allowed.
        """
        X = np.asarray(X, order="F", dtype=np.float64)
        y = np.asarray(y, order="C", dtype=np.float64)

        if X.ndim != 2:
            raise ValueError(
                f"Expected a 2D array for input samples, but got an array with {X.ndim} dimensions."
            )

        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"The number of samples in `X` ({X.shape[0]}) does not match the number of target values in `y` ({y.shape[0]})."
            )

        # Validate target vector
        self._validate_target(y)

        # Validate sample weights
        sample_weight = self._process_sample_weight(sample_weight, y.shape[0])

        # Validate feature matrix
        self._validate_features(X)

        # Classification or Regression setup
        self.n_classes = self._setup_task(y)

        # Validate categorical features
        self._validate_categories_in_data(X, is_fit=True)

        # Warn if the number of feature combinations is too large for oblique splits
        if self.use_oblique:
            self._warn_large_combinations(X.shape[1] - len(self.categories))

        super().fit(X, y, sample_weight)

        self._fit = True

        return self

    def _validate_target(self, y: NDArray) -> None:
        if y.ndim != 1:
            raise ValueError("y must be 1-dimensional")

        if self.task:  # Regression
            return
        else:  # Classification
            unique_labels = np.unique(y)
            expected_labels = np.arange(len(unique_labels))
            if not np.array_equal(unique_labels, expected_labels):
                raise ValueError(
                    "Classification labels must start from 0 and increment by 1"
                )

    def _process_sample_weight(
        self, sample_weight: Optional[ArrayLike], n_samples: int
    ) -> NDArray:
        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight, order="C", dtype=np.float64)

            if sample_weight.shape != (n_samples,):
                raise ValueError(
                    f"sample_weight has incompatible shape: {sample_weight.shape} "
                    f"while y has shape ({n_samples},)"
                )

            if (
                np.any(np.isnan(sample_weight))
                or np.any(np.isinf(sample_weight))
                or np.any(sample_weight < 0)
            ):
                raise ValueError(
                    "sample_weight cannot contain negative, NaN or inf values"
                )

            min_val = np.min(sample_weight)
            if min_val != 1:
                sample_weight = sample_weight / min_val

        else:
            sample_weight = np.ones(n_samples, dtype=np.float64)

        return sample_weight

    def _validate_features(self, X: NDArray) -> None:
        if self.use_oblique:
            if np.any(np.isnan(X)) or np.any(np.isinf(X)):
                raise ValueError(
                    "X cannot contain NaN or Inf values when use_oblique is True"
                )

        max_possible_pairs = (
            X.shape[1] - len(self.categories) if self.categories else X.shape[1]
        )

        if self.categories:
            if max_possible_pairs < 2:
                warnings.warn(
                    f"Total features: {X.shape[1]}, categorical features: {len(self.categories)}. "
                    f"The number of possible feature pairs ({max_possible_pairs}) is less than 2. "
                    f"As a result, 'use_oblique' set 'False'."
                )
                self.use_oblique = False

            elif self.n_pair > max_possible_pairs:
                warnings.warn(
                    f"Total features: {X.shape[1]}, categorical features: {len(self.categories)}. "
                    f"n_pair ({self.n_pair}) exceeds the usable features, adjusting n_pair to {max_possible_pairs}."
                )
                self.n_pair = max_possible_pairs
        else:  # If there are no categorical features
            if self.n_pair > X.shape[1]:
                warnings.warn(
                    f"n_pair ({self.n_pair}) exceeds the total features ({X.shape[1]}). "
                    f"Adjusting n_pair to {X.shape[1]}."
                )
                self.n_pair = X.shape[1]

    def _setup_task(self, y: NDArray) -> int:
        if not self.task:
            n_classes = len(np.unique(y))
            return n_classes
        else:
            return 1  # Regression

    def _validate_categories_in_data(self, X: NDArray, is_fit: bool) -> None:
        if self.categories:
            for col_idx in self.categories:
                # Kategori indeksi matris boyutlarını aşmamalı
                if col_idx >= X.shape[1]:
                    raise ValueError(
                        f"Category column index {col_idx} exceeds X dimensions ({X.shape[1]} features)."
                    )

            # Kategorik sütunlardaki değerler negatif olmamalı
            if (X[:, self.categories] < 0).any():
                raise ValueError(
                    "X contains negative values in the specified category columns, which are not allowed."
                )

            if np.isnan(X[:, self.categories]).any():
                raise ValueError(
                    "X contains null values in the specified category columns. Please encode them before passing."
                )

            if is_fit:
                self._categories = {
                    idx: np.unique(X[:, idx]) for idx in self.categories
                }

            else:
                for idx in self.categories:
                    unknown = np.setdiff1d(np.unique(X[:, idx]), self._categories[idx])
                    if len(unknown) > 0:
                        raise ValueError(
                            f"Unknown categories in column {idx}: {unknown}. "
                            f"Available categories: {self._categories[idx]}"
                        )

    def _warn_large_combinations(self, n_features: int) -> None:
        total_combinations = comb(n_features, self.n_pair)
        if total_combinations > 1000:  # Optimal threshold can be adjusted
            warnings.warn(
                "The number of feature combinations for oblique splits is very large, which may lead to long training times. "
                "Consider reducing `n_pair` or the number of features."
            )

    def predict(self, X: ArrayLike) -> NDArray:
        """
        Predict target values for the input samples.

        Parameters
        ----------
        X : ArrayLike
            Input samples of shape (n_samples, n_features).

        Returns
        -------
        NDArray
            Predicted values.

        Raises
        ------
        ValueError
            If the model has not been fitted yet.
        """
        if not self._fit:
            raise ValueError(
                "The model has not been fitted yet. Please call `fit` first."
            )

        X = np.asarray(X, order="F", dtype=np.float64)

        if X.ndim != 2:
            raise ValueError(
                f"Expected a 2D array for input samples, but got an array with {X.ndim} dimensions. "
            )

        self._validate_categories_in_data(X, is_fit=False)

        return super().predict(X)


class Classifier(BaseTree):
    def __init__(
        self,
        use_oblique: bool = True,
        max_depth: int = -1,
        min_samples_leaf: int = 1,
        min_samples_split: int = 2,
        min_impurity_decrease: float = 0.0,
        ccp_alpha: float = 0.0,
        categories: Optional[List[int]] = None,
        random_state: Optional[int] = None,
        n_pair: int = 2,
        gamma: float = 1.0,
        max_iter: int = 100,
        relative_change: float = 0.001,
    ):
        """
        A decision tree classifier supporting both traditional axis-aligned and oblique splits.

        This advanced decision tree classifier extends traditional regression trees by supporting oblique
        splits (linear combinations of features) alongside conventional axis-aligned splits. It offers enhanced
        flexibility in modeling continuous outputs while maintaining the interpretability of decision trees.

        Parameters
        ----------
        use_oblique : bool, default=True
            - If `True`, enables oblique splits using linear combinations of features.
            - If `False`, uses traditional axis-aligned splits only.

        max_depth : int, default=-1
            Maximum depth of the tree. Controls model complexity and prevents overfitting.

            - If `-1`: Expands until leaves are pure or contain fewer than `min_samples_split` samples.
            - If `int > 0`: Limits the tree to the specified depth.

        min_samples_leaf : int, default=1
            Minimum number of samples required at leaf nodes.

        min_samples_split : int, default=2
            Minimum number of samples required to split an internal node.

        min_impurity_decrease : float, default=0.0
            Minimum required decrease in impurity to create a split.

        ccp_alpha : float, default=0.0
            Complexity parameter for Minimal Cost-Complexity Pruning.

        categories : List[int], default=None
            Indices of categorical features in the dataset.

        random_state : int, default=None
            Seed for random number generation in oblique splits.

            - Only used when `use_oblique=True`.

        n_pair : int, default=2
            Number of features to combine in oblique splits.

            - Only used when `use_oblique=True`.

        gamma : float, default=1.0
            Separation strength parameter for oblique splits.

            - Only used when `use_oblique=True`.

        max_iter : int, default=100
            Maximum iterations for L-BFGS optimization in oblique splits.

            - Only used when `use_oblique=True`.

        relative_change : float, default=0.001
            Early stopping threshold for L-BFGS optimization.

            - Only used when `use_oblique=True`.
        """
        super().__init__(
            task=False,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            min_samples_split=min_samples_split,
            min_impurity_decrease=min_impurity_decrease,
            ccp_alpha=ccp_alpha,
            categories=categories,
            use_oblique=use_oblique,
            random_state=random_state,
            n_pair=n_pair,
            gamma=gamma,
            max_iter=max_iter,
            relative_change=relative_change,
        )

    def fit(
        self, X: ArrayLike, y: ArrayLike, sample_weight: Optional[ArrayLike] = None
    ) -> "Classifier":
        """
        Build a decision tree classifier from the training set (X, y).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,)
            Target values (class labels).
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.

        Returns
        -------
        self : Classifier
            Fitted estimator.
        """
        return super().fit(X, y, sample_weight)

    def predict(self, X: ArrayLike) -> NDArray:
        """
        Predict regression target for X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples to predict.

        Returns
        -------
        y : NDArray of shape (n_samples,)
            The predicted values.
        """
        return np.argmax(super().predict(X), axis=1)

    def predict_proba(self, X: ArrayLike) -> NDArray:
        """
        Predict class probabilities for X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        proba : NDArray of shape (n_samples, n_classes)
            The class probabilities of the input samples.
        """
        return super().predict(X)


class Regressor(BaseTree):
    def __init__(
        self,
        use_oblique: bool = True,
        max_depth: int = -1,
        min_samples_leaf: int = 1,
        min_samples_split: int = 2,
        min_impurity_decrease: float = 0.0,
        ccp_alpha: float = 0.0,
        categories: Optional[List[int]] = None,
        random_state: Optional[int] = None,
        n_pair: int = 2,
        gamma: float = 1.0,
        max_iter: int = 100,
        relative_change: float = 0.001,
    ):
        """
        A decision tree regressor supporting both traditional axis-aligned and oblique splits.

        This advanced decision tree regressor extends traditional regression trees by supporting oblique
        splits (linear combinations of features) alongside conventional axis-aligned splits. It offers enhanced
        flexibility in modeling continuous outputs while maintaining the interpretability of decision trees.

        Parameters
        ----------
        use_oblique : bool, default=True
            - If `True`, enables oblique splits using linear combinations of features.
            - If `False`, uses traditional axis-aligned splits only.

        max_depth : int, default=-1
            Maximum depth of the tree. Controls model complexity and prevents overfitting.

            - If `-1`: Expands until leaves are pure or contain fewer than `min_samples_split` samples.
            - If `int > 0`: Limits the tree to the specified depth.

        min_samples_leaf : int, default=1
            Minimum number of samples required at leaf nodes.

        min_samples_split : int, default=2
            Minimum number of samples required to split an internal node.

        min_impurity_decrease : float, default=0.0
            Minimum required decrease in impurity to create a split.

        ccp_alpha : float, default=0.0
            Complexity parameter for Minimal Cost-Complexity Pruning.

        categories : List[int], default=None
            Indices of categorical features in the dataset.

        random_state : int, default=None
            Seed for random number generation in oblique splits.

            - Only used when `use_oblique=True`.

        n_pair : int, default=2
            Number of features to combine in oblique splits.

            - Only used when `use_oblique=True`.

        gamma : float, default=1.0
            Separation strength parameter for oblique splits.

            - Only used when `use_oblique=True`.

        max_iter : int, default=100
            Maximum iterations for L-BFGS optimization in oblique splits.

            - Only used when `use_oblique=True`.

        relative_change : float, default=0.001
            Early stopping threshold for L-BFGS optimization.

            - Only used when `use_oblique=True`.
        """
        super().__init__(
            task=True,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            min_samples_split=min_samples_split,
            min_impurity_decrease=min_impurity_decrease,
            ccp_alpha=ccp_alpha,
            categories=categories,
            use_oblique=use_oblique,
            random_state=random_state,
            n_pair=n_pair,
            gamma=gamma,
            max_iter=max_iter,
            relative_change=relative_change,
        )

    def fit(
        self, X: ArrayLike, y: ArrayLike, sample_weight: Optional[ArrayLike] = None
    ) -> "Regressor":
        """
        Build a decision tree regressor from the training set (X, y).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,)
            Target values.
        sample_weight : array-like of shape (n_samples,), optional, default=None
            Sample weights.

        Returns
        -------
        self : Regressor
            Fitted estimator.
        """
        return super().fit(X, y, sample_weight)

    def predict(self, X: ArrayLike) -> NDArray:
        """
        Predict regression target for X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples to predict.

        Returns
        -------
        y : NDArray of shape (n_samples,)
            The predicted values.
        """
        return super().predict(X).ravel()
