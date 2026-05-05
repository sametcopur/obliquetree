from libc.stdlib cimport malloc, free, calloc
import numpy as np
cimport numpy as np

from .tree cimport (
    build_tree_recursive,
    finalize_tree_metadata,
    predict,
    apply,
    fit_linear_leaves,
    SortItem,
    CategoryStat,
)
from .ccp cimport prune_tree
from .utils cimport analyze_X, export_tree, deserialize_tree

cdef void free_tree(TreeNode* node) noexcept nogil:
    if node:
        # First recursively free child nodes
        if not node.is_leaf:
            free_tree(node.left)
            free_tree(node.right)

        # Free all member pointers first
        if node.x != NULL:
            free(node.x)
            node.x = NULL

        if node.pair != NULL:
            free(node.pair)
            node.pair = NULL

        if node.categories_go_left != NULL:
            free(node.categories_go_left)
            node.categories_go_left = NULL

        if node.value_multiclass != NULL:
            free(node.value_multiclass)
            node.value_multiclass = NULL

        if node.leaf_coef != NULL:
            free(node.leaf_coef)
            node.leaf_coef = NULL

        if node.leaf_intercept_buf != NULL:
            free(node.leaf_intercept_buf)
            node.leaf_intercept_buf = NULL

        # Finally free the node itself
        free(node)

cdef class TreeClassifier:
    def __init__(self,  unsigned char max_depth,
                        int min_samples_leaf,
                        int min_samples_split,
                        double min_impurity_decrease,
                        int random_state,
                        int n_pair,
                        int top_k,
                        double gamma,
                        int max_iter,
                        double relative_change,
                        list categories,
                        double ccp_alpha,
                        bint use_oblique,
                        bint task,
                        int n_classes,
                        bint linear_leaf,
                        double leaf_l2,
                        int leaf_max_iter) -> None:
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.random_state = random_state
        self.n_pair = n_pair
        self.top_k = top_k
        self.gamma = gamma
        self.max_iter = max_iter
        self.relative_change = relative_change
        self.use_oblique = use_oblique
        self.categories = categories
        self.min_samples_split = min_samples_split
        self.ccp_alpha = ccp_alpha
        self.task = task
        self.n_classes = n_classes
        self.linear_leaf = linear_leaf
        self.leaf_l2 = leaf_l2
        self.leaf_max_iter = leaf_max_iter

        self.cat_ = len(categories) > 0
        self.rng_ = np.random.default_rng(random_state)

    def __cinit__(self, ):
        self.root = NULL
        self.numeric_features_ = NULL
        self.n_numeric_features_ = 0

    def __dealloc__(self):
        if self.root != NULL:
            free_tree(self.root)
        if self.numeric_features_ != NULL:
            free(self.numeric_features_)
            self.numeric_features_ = NULL
            
    def __getstate__(self):
        return export_tree(self)

    def __setstate__(self, state):
        """Pickle'dan durum değerlerini yükle."""
        cdef dict params = state["params"]
        cdef dict tree = state["tree"]
        cdef list numeric_features
        cdef int i

        self.max_depth = params.get('max_depth', 255)
        self.min_samples_leaf = params.get('min_samples_leaf', 1)
        self.min_samples_split = params.get('min_samples_split', 2)
        self.min_impurity_decrease = params.get('min_impurity_decrease', 0.0)
        self.random_state = params.get('random_state', 42)
        self.n_pair = params.get('n_pair', 2)
        self.top_k = params.get('top_k', 0)
        self.gamma = params.get('gamma', 1.0)
        self.max_iter = params.get('max_iter', 100)
        self.relative_change = params.get('relative_change', 0.001)
        self.categories = params.get('categories', [])
        self.ccp_alpha = params.get('ccp_alpha', 0.0)
        self.use_oblique = params.get('use_oblique', True)
        self.linear_leaf = params.get('linear_leaf', False)
        # Accept legacy 'leaf_ridge' key for pickles produced before the rename.
        self.leaf_l2 = params.get('leaf_l2', params.get('leaf_ridge', 0.0))
        self.leaf_max_iter = params.get('leaf_max_iter', 100)

        self.cat_ = params.get('cat_', False)
        self.rng_ = np.random.default_rng(self.random_state)

        self.task = params["task"]
        self.n_classes = params['n_classes']
        self.n_features = params['n_features']

        if self.numeric_features_ != NULL:
            free(self.numeric_features_)
            self.numeric_features_ = NULL
        self.n_numeric_features_ = 0

        numeric_features = params.get('numeric_features', None)
        if numeric_features is not None and len(numeric_features) > 0:
            self.n_numeric_features_ = len(numeric_features)
            self.numeric_features_ = <int*>malloc(self.n_numeric_features_ * sizeof(int))
            if self.numeric_features_ == NULL:
                raise MemoryError()
            for i in range(self.n_numeric_features_):
                self.numeric_features_[i] = numeric_features[i]

        self.root = deserialize_tree(tree, self.n_features, self.n_classes)
        if self.root != NULL:
            finalize_tree_metadata(self.root)

    cpdef fit(self, double[::1, :] X, double[::1] y, double[::1] sample_weight):
        cdef int n_samples = X.shape[0]
        cdef int n_columns = X.shape[1]
        cdef int i
        cdef int n_numeric

        # Temel bellek ayırma işlemleri
        cdef SortItem* sort_buffer = NULL
        cdef int* sample_indices = NULL
        cdef int* nan_indices = NULL
        cdef bint* is_categorical = NULL
        cdef CategoryStat* categorical_stats = NULL
        cdef bint* is_integer = NULL


        try:
            self.n_features = n_columns

            if self.root != NULL:
                free_tree(self.root)
                self.root = NULL

            if self.numeric_features_ != NULL:
                free(self.numeric_features_)
                self.numeric_features_ = NULL
            self.n_numeric_features_ = 0

            # Bellek ayırma işlemleri
            sort_buffer = <SortItem*>malloc(n_samples * sizeof(SortItem))
            sample_indices = <int*>malloc(n_samples * sizeof(int))
            nan_indices = <int*>malloc(n_samples * sizeof(int))
            is_categorical = <bint*>calloc(n_columns, sizeof(bint))
            is_integer = <bint*>malloc(n_columns * sizeof(bint))

            if ((sort_buffer == NULL) or (sample_indices == NULL) or
                (nan_indices == NULL) or (is_categorical == NULL) or
                 (is_integer == NULL)
                ):
                raise MemoryError()

            analyze_X(X, is_integer)

            if self.cat_:
                categorical_stats = <CategoryStat*>malloc(n_samples * sizeof(CategoryStat))

                if categorical_stats == NULL:
                    raise MemoryError()

            # İndeksleri hazırla
            for i in range(n_samples):
                sample_indices[i] = i

            # Kategorik değişkenleri işaretle
            for i in self.categories:
                is_categorical[i] = 1

            # Numeric feature index list (used by linear-leaf fit + predict)
            n_numeric = 0
            for i in range(n_columns):
                if not is_categorical[i]:
                    n_numeric += 1
            if n_numeric > 0:
                self.numeric_features_ = <int*>malloc(n_numeric * sizeof(int))
                if self.numeric_features_ == NULL:
                    raise MemoryError()
                self.n_numeric_features_ = n_numeric
                n_numeric = 0
                for i in range(n_columns):
                    if not is_categorical[i]:
                        self.numeric_features_[n_numeric] = i
                        n_numeric += 1

            # Ağacı oluştur
            self.root = build_tree_recursive(
                self.task,
                self.n_classes,
                X, y,
                0,
                self.max_depth,
                self.min_samples_leaf,
                self.min_samples_split,
                self.min_impurity_decrease,
                sort_buffer,
                sample_indices,
                nan_indices,
                n_samples,
                self.n_pair,
                self.top_k,
                self.gamma,
                self.max_iter,
                self.relative_change,
                self.rng_,
                self.use_oblique,
                is_categorical,
                categorical_stats,
                sample_weight,
                is_integer,
            )

            if self.ccp_alpha > 0.0:
                prune_tree(self.root, self.ccp_alpha)

            if (self.linear_leaf and
                    self.root != NULL and self.n_numeric_features_ > 0):
                fit_linear_leaves(
                    self.root,
                    X,
                    y,
                    sample_weight,
                    self.numeric_features_,
                    self.n_numeric_features_,
                    self.leaf_l2,
                    self.leaf_max_iter,
                    n_samples,
                    self.n_classes,
                    self.task,
                )

            if self.root != NULL:
                finalize_tree_metadata(self.root)

        finally:
            free(sort_buffer)
            free(sample_indices)
            free(nan_indices)
            free(is_categorical)
            free(is_integer)

            if categorical_stats != NULL:
                free(categorical_stats)

        return self

    cpdef apply(self, double[::1, :] X):
        cdef int n_samples = X.shape[0]
        cdef int n_features = X.shape[1]
        cdef np.ndarray[int, ndim=1] out

        if self.root == NULL:
            raise ValueError("The model has not been fitted yet. Call the 'fit' method before using this model.")

        if self.n_features != n_features:
            raise ValueError(f"Mismatch in number of features: expected {self.n_features}, but got {n_features}.")

        out = np.empty(n_samples, dtype=np.intc)
        apply(self.root, X, out, n_samples)
        return out

    cpdef predict(self, double[::1, :] X):
        """
        Return an (n_samples, 2) array: [prob_of_class_0, prob_of_class_1].
        'node.value' is interpreted as the probability for class 1.
        """
        cdef int n_samples = X.shape[0]
        cdef int n_features = X.shape[1]
        cdef np.ndarray[double, ndim=2] out
        cdef np.ndarray[double, ndim=2] proba 
        cdef int i

        if self.root == NULL:
            raise ValueError("The model has not been fitted yet. Call the 'fit' method before using this model.")

        if self.n_features != n_features:
            raise ValueError(f"Mismatch in number of features: expected {self.n_features}, but got {n_features}.")

        if self.n_classes > 2:
            out = np.zeros((n_samples, self.n_classes), dtype=np.float64)
        else:
            out = np.zeros((n_samples, 1), dtype=np.float64)

        predict(
            self.root,
            X,
            out,
            n_samples,
            self.n_classes,
            self.linear_leaf,
            self.numeric_features_,
            self.n_numeric_features_,
            self.task == 0,
        )

        if (self.n_classes <= 2) and (self.task == 0):
            proba = np.empty((n_samples, self.n_classes), dtype=np.float64)
            for i in range(n_samples):
                proba[i, 1] = out[i, 0]           # prob of class 1
                proba[i, 0] = 1.0 - out[i, 0]     # prob of class 0

            return proba

        return out
