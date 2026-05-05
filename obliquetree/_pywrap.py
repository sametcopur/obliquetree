from .src.base import TreeClassifier

from typing import List, Optional
from numpy.typing import ArrayLike, NDArray
import numpy as np
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

    top_k : int or None
        Number of numeric features kept after cheap oblique feature screening.

        - If `None`, an internal heuristic is used.
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

    linear_leaf : bool
        If `True`, fit a small parametric model inside every terminal leaf
        instead of returning a single constant. See ``Notes`` for the per-task
        formulation.

    leaf_l2 : float
        L2 (ridge) penalty added to the leaf-level normal equations / Hessian
        diagonal when ``linear_leaf=True``. See ``Notes`` for the regression
        vs. classification semantics.

    leaf_max_iter : int
        Maximum Newton / IRLS iterations per leaf when ``linear_leaf=True``.
        Each iteration solves a Cholesky system; iteration stops early when
        the parameter step falls below ``1e-6`` (in absolute value). Used
        only for binary logistic IRLS and multiclass softmax Newton;
        regression OLS is closed-form and does not iterate.

    Notes
    -----
    **Linear leaves (``linear_leaf=True``)**

    A standard decision tree returns a single constant at each terminal leaf
    (the weighted mean for regression, the class frequency for
    classification). With ``linear_leaf=True`` each leaf instead fits a small
    parametric model on the training samples that fall into it. The tree
    splits still partition the input space; the leaf model only refines the
    prediction inside that local region. This combines the local
    piecewise-constant behaviour of trees with the smooth interpolation of a
    linear model and is useful when the target is approximately
    piecewise-linear within tree-segmented regions.

    Only **numeric** features participate in the leaf model. Columns listed
    in ``categories`` are excluded from the leaf coefficients because the
    tree splits already capture categorical structure; mixing them into the
    leaf regression would double-count their effect.

    The leaf model is selected based on the task:

    - **Regression**: weighted ordinary least squares (OLS) on the leaf
      samples, solved in closed form via a centered Cholesky decomposition.
      Prediction: ``intercept + Σ coef · x_numeric``.
    - **Binary classification**: weighted logistic regression fit by
      iteratively reweighted least squares (IRLS) for up to
      ``leaf_max_iter`` iterations (each iteration is the same Cholesky
      solve as the regression case). Prediction:
      ``sigmoid(intercept + Σ coef · x_numeric)`` interpreted as
      ``P(class = 1)``.
    - **Multiclass classification (n_classes > 2)**: multinomial softmax
      regression with K-1 reference-class parametrization (the last class
      logit is fixed at 0 to remove the rank-1 redundancy of the symmetric
      K-class form). Fit by Newton-Raphson on the full (K-1)·(d+1) Hessian
      for up to ``leaf_max_iter`` iterations. Prediction: ``softmax`` over
      the K logits.

    Both iterative schemes break early once the largest absolute parameter
    step falls below ``1e-6``; the last finite iterate is accepted, so
    reaching ``leaf_max_iter`` without strict convergence is not an error.
    A leaf falls back to the constant prediction (mean / frequency) when:

    - the leaf has fewer samples than coefficients (``n_samples < d + 1``),
    - the Hessian/Gram matrix is singular (e.g. correlated numeric features
      with ``leaf_l2 = 0`` in the regression case),
    - the Newton/IRLS step produces non-finite parameters,
    - any used numeric feature is NaN/Inf at predict time.

    **``leaf_l2``** adds an L2 penalty ``λ · I`` to the coefficient block of
    the normal equations (intercept excluded). For regression
    ``leaf_l2 = 0`` is honored exactly (true unregularized OLS); on
    rank-deficient numeric features the leaf falls back to the mean. For
    classification a tiny internal floor of ``1e-10`` is always applied
    because the K-class softmax Hessian is rank-deficient by construction
    without it; ``leaf_l2 = 0`` therefore behaves as ``1e-10`` for
    classifiers. Tree leaves are typically small, so a larger value
    (e.g. ``0.1`` - ``1.0``) often improves probability calibration without
    harming accuracy.
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
        top_k: Optional[int],
        gamma: float,
        max_iter: int,
        relative_change: float,
        linear_leaf: bool = False,
        leaf_l2: float = 1e-6,
        leaf_max_iter: int = 100,
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
        self.top_k = self._validate_top_k(top_k, self.n_pair)
        self.gamma = self._validate_gamma(gamma)
        self.max_iter = self._validate_max_iter(max_iter)
        self.relative_change = self._validate_relative_change(
            relative_change, self.use_oblique
        )
        self.random_state = self._validate_random_state(random_state)
        self.categories = self._validate_categories(categories)
        self.linear_leaf = self._validate_linear_leaf(linear_leaf)
        self.leaf_l2 = self._validate_leaf_l2(leaf_l2)
        self.leaf_max_iter = self._validate_leaf_max_iter(leaf_max_iter)
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
            self.top_k,
            self.gamma,
            self.max_iter,
            self.relative_change,
            self.categories,
            self.ccp_alpha,
            self.use_oblique,
            self.task,
            1,
            self.linear_leaf,
            self.leaf_l2,
            self.leaf_max_iter,
        )

    def __getstate__(self):
        """Return the state for pickling."""
        state = super().__getstate__()
        state["_fit"] = self._fit
        cats = getattr(self, "_categories", None)
        if cats is not None:
            state["_categories"] = {
                int(k): np.asarray(v) for k, v in cats.items()
            }
        return state

    def __setstate__(self, state):
        """Restore the state from pickle."""
        # Extract special attributes
        _fit = state.pop("_fit", False)
        _categories = state.pop("_categories", None)
        super().__setstate__(state)

        # Restore state directly without re-initialization
        self.__dict__.update(state)
        self._fit = _fit
        if _categories is not None:
            self._categories = _categories

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
            f"top_k={getattr(self, 'top_k', None)}, "
            f"gamma={getattr(self, 'gamma', None)}, "
            f"max_iter={getattr(self, 'max_iter', None)}, "
            f"relative_change={getattr(self, 'relative_change', None)}, "
            f"linear_leaf={getattr(self, 'linear_leaf', None)}, "
            f"leaf_l2={getattr(self, 'leaf_l2', None)}, "
            f"leaf_max_iter={getattr(self, 'leaf_max_iter', None)}"
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

    def _validate_top_k(self, top_k: Optional[int], n_pair: int) -> int:
        if top_k is None:
            return 0
        if not isinstance(top_k, int):
            raise ValueError("top_k must be an integer or None")
        if top_k < n_pair:
            raise ValueError("top_k must be >= n_pair")
        return top_k

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

    def _validate_linear_leaf(self, linear_leaf: bool) -> bool:
        if not isinstance(linear_leaf, bool):
            raise ValueError("linear_leaf must be a boolean")
        return linear_leaf

    def _validate_leaf_l2(self, leaf_l2: float) -> float:
        if not isinstance(leaf_l2, (int, float)) or isinstance(leaf_l2, bool):
            raise ValueError("leaf_l2 must be a number")
        if not np.isfinite(leaf_l2):
            raise ValueError("leaf_l2 must be finite (no NaN/Inf)")
        if leaf_l2 < 0.0:
            raise ValueError("leaf_l2 must be >= 0.0")
        return float(leaf_l2)

    def _validate_leaf_max_iter(self, leaf_max_iter: int) -> int:
        if not isinstance(leaf_max_iter, int) or isinstance(leaf_max_iter, bool):
            raise ValueError("leaf_max_iter must be an integer")
        if leaf_max_iter < 1:
            raise ValueError("leaf_max_iter must be >= 1")
        return leaf_max_iter

    def _coerce_feature_matrix(self, X: ArrayLike) -> NDArray:
        X = np.asarray(X, order="F", dtype=np.float64)

        if X.ndim != 2:
            raise ValueError(
                f"Expected a 2D array for input samples, but got an array with {X.ndim} dimensions."
            )

        return X

    def _prepare_inference_input(self, X: ArrayLike) -> NDArray:
        if not self._fit:
            raise ValueError(
                "The model has not been fitted yet. Please call `fit` first."
            )

        X = self._coerce_feature_matrix(X)
        self._validate_categories_in_data(X, is_fit=False)
        return X

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
        X = self._coerce_feature_matrix(X)
        y = np.asarray(y, order="C", dtype=np.float64)

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
            if len(unique_labels) < 2:
                raise ValueError(
                    "Classification requires at least 2 distinct classes in y; "
                    f"got {len(unique_labels)}."
                )
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

            positive_mask = sample_weight > 0
            if not positive_mask.any():
                raise ValueError(
                    "sample_weight must contain at least one positive value; "
                    "all zeros leaves no effective training samples."
                )

            if positive_mask.any():
                min_val = np.min(sample_weight[positive_mask])
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

        if self.linear_leaf and self.categories:
            numeric_cols = [c for c in range(X.shape[1]) if c not in set(self.categories)]
            if numeric_cols:
                X_num = X[:, numeric_cols]
                if np.any(np.isnan(X_num)) or np.any(np.isinf(X_num)):
                    raise ValueError(
                        "Numeric columns of X cannot contain NaN or Inf values when linear_leaf is True"
                    )
        elif self.linear_leaf:
            if np.any(np.isnan(X)) or np.any(np.isinf(X)):
                raise ValueError(
                    "X cannot contain NaN or Inf values when linear_leaf is True"
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

            category_values = X[:, self.categories]

            # Kategorik sütunlardaki değerler negatif olmamalı
            if (category_values < 0).any():
                raise ValueError(
                    "X contains negative values in the specified category columns, which are not allowed."
                )

            if np.isnan(category_values).any():
                raise ValueError(
                    "X contains null values in the specified category columns. Please encode them before passing."
                )

            if np.isinf(category_values).any():
                raise ValueError(
                    "X contains Inf values in the specified category columns, which are not allowed."
                )

            if is_fit:
                self._categories = {}
                for offset, idx in enumerate(self.categories):
                    self._categories[idx] = np.unique(category_values[:, offset])

            else:
                for offset, idx in enumerate(self.categories):
                    unknown = np.setdiff1d(
                        np.unique(category_values[:, offset]), self._categories[idx]
                    )
                    if len(unknown) > 0:
                        raise ValueError(
                            f"Unknown categories in column {idx}: {unknown}. "
                            f"Available categories: {self._categories[idx]}"
                        )

    def apply(self, X: ArrayLike) -> NDArray:
        """
        Return the index of the leaf that each sample is predicted as.

        Nodes are numbered using pre-order (depth-first) traversal, consistent
        with scikit-learn's ``DecisionTreeClassifier.apply``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        X_leaves : NDArray of shape (n_samples,)
            For each datapoint x in X, return the index of the leaf x
            ends up in. Indices are in ``[0, n_nodes)``.

        Raises
        ------
        ValueError
            If the model has not been fitted yet.
        """
        X = self._prepare_inference_input(X)

        return super().apply(X)

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
        X = self._prepare_inference_input(X)

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
        top_k: Optional[int] = None,
        gamma: float = 1.0,
        max_iter: int = 100,
        relative_change: float = 0.001,
        linear_leaf: bool = False,
        leaf_l2: float = 1e-6,
        leaf_max_iter: int = 100,
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

        top_k : int or None, default=None
            Number of numeric features kept after cheap oblique feature screening.

            - If `None`, an internal heuristic is used.
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

        linear_leaf : bool, default=False
            If `True`, replace the constant leaf with a small parametric
            model fit on the leaf samples:

            - Binary classification → weighted logistic regression (IRLS,
              25 iters); predict returns ``sigmoid(intercept + coef · x)``
              as ``P(class=1)``.
            - Multiclass (``n_classes > 2``) → multinomial softmax
              regression with K-1 reference-class parametrization
              (Newton-Raphson on the full Hessian, 50 iters); predict
              returns the softmax over per-class logits.

            Only numeric features participate in the leaf coefficients
            (categorical features in ``categories`` are excluded; the
            tree splits already capture their structure). See ``Notes``
            on ``BaseTree`` for the full mechanism, fallback rules, and
            iteration policy.

        leaf_l2 : float, default=1e-6
            L2 (ridge) penalty on the leaf coefficients. For
            classification a tiny internal floor of ``1e-10`` is always
            applied (the K-class softmax Hessian is rank-deficient
            without it), so ``leaf_l2=0.0`` is effectively ``1e-10`` for
            classifiers. A larger value (e.g. ``0.1`` - ``1.0``) often
            improves probability calibration without harming accuracy.

        leaf_max_iter : int, default=100
            Maximum IRLS / Newton iterations per leaf. Iteration stops
            early when the largest absolute parameter step falls below
            ``1e-6``; with well-conditioned data convergence is typically
            reached in 5-15 iterations. Increase (e.g. to ``500``) for
            harder leaves where the default cap might leave the iterate
            short of the optimum.
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
            top_k=top_k,
            gamma=gamma,
            max_iter=max_iter,
            relative_change=relative_change,
            linear_leaf=linear_leaf,
            leaf_l2=leaf_l2,
            leaf_max_iter=leaf_max_iter,
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
        Predict class labels for X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples to predict.

        Returns
        -------
        y : NDArray of shape (n_samples,)
            The predicted class labels.
        """
        return np.argmax(super().predict(X), axis=1)

    def apply(self, X: ArrayLike) -> NDArray:
        """
        Return the index of the leaf that each sample ends up in.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        X_leaves : NDArray of shape (n_samples,)
            For each datapoint x in X, return the index of the leaf x
            ends up in. Nodes are numbered using pre-order (depth-first)
            traversal.
        """
        return super().apply(X)

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
        top_k: Optional[int] = None,
        gamma: float = 1.0,
        max_iter: int = 100,
        relative_change: float = 0.001,
        linear_leaf: bool = False,
        leaf_l2: float = 1e-6,
        leaf_max_iter: int = 100,
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

        top_k : int or None, default=None
            Number of numeric features kept after cheap oblique feature screening.

            - If `None`, an internal heuristic is used.
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

        linear_leaf : bool, default=False
            If `True`, replace the constant leaf (the weighted mean of
            ``y``) with a weighted ordinary-least-squares model fit on
            the leaf samples. Predict returns
            ``intercept + sum(coef * x_numeric)``.

            Only numeric features participate in the leaf coefficients
            (categorical features in ``categories`` are excluded; the
            tree splits already capture their structure). See ``Notes``
            on ``BaseTree`` for the full mechanism and fallback rules.

        leaf_l2 : float, default=1e-6
            L2 (ridge) penalty on the leaf coefficients. ``0.0`` is
            honored exactly (true unregularized OLS); on rank-deficient
            numeric features the affected leaf falls back to the mean
            instead. Pass a small positive value (e.g. ``1e-8``) to
            keep the fit on correlated features.

        leaf_max_iter : int, default=100
            Unused for regression (OLS is closed-form), kept for API
            symmetry with ``Classifier`` so ``Regressor`` and
            ``Classifier`` share the same constructor surface.
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
            top_k=top_k,
            gamma=gamma,
            max_iter=max_iter,
            relative_change=relative_change,
            linear_leaf=linear_leaf,
            leaf_l2=leaf_l2,
            leaf_max_iter=leaf_max_iter,
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

    def apply(self, X: ArrayLike) -> NDArray:
        """
        Return the index of the leaf that each sample ends up in.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        X_leaves : NDArray of shape (n_samples,)
            For each datapoint x in X, return the index of the leaf x
            ends up in. Nodes are numbered using pre-order (depth-first)
            traversal.
        """
        return super().apply(X)
