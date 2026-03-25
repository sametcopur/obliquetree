from libc.stdlib cimport malloc, free, calloc
import numpy as np
cimport numpy as np

from .tree cimport build_tree_recursive, predict, apply, SortItem, CategoryStat
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
            
        # Finally free the node itself
        free(node)

cdef class TreeClassifier:
    def __init__(self,  unsigned char max_depth, 
                        int min_samples_leaf, 
                        int min_samples_split,
                        double min_impurity_decrease,  
                        int random_state, 
                        int n_pair, 
                        double gamma,
                        int max_iter,
                        double relative_change,
                        list categories,
                        double ccp_alpha,
                        bint use_oblique,
                        bint task,
                        int n_classes) -> None:
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.random_state = random_state
        self.n_pair = n_pair
        self.gamma = gamma
        self.max_iter = max_iter
        self.relative_change = relative_change
        self.use_oblique = use_oblique
        self.categories = categories
        self.min_samples_split = min_samples_split
        self.ccp_alpha = ccp_alpha
        self.task = task
        self.n_classes = n_classes

        self.cat_ = len(categories) > 0
        self.rng_ = np.random.default_rng(random_state)

    def __cinit__(self, ):
        self.root = NULL

    def __dealloc__(self):
        if self.root != NULL:
            free_tree(self.root)
            
    def __getstate__(self):
        return export_tree(self)

    def __setstate__(self, state):
        """Pickle'dan durum değerlerini yükle."""
        cdef dict params = state["params"]
        cdef dict tree = state["tree"]

        self.max_depth = params.get('max_depth', 255)
        self.min_samples_leaf = params.get('min_samples_leaf', 1)
        self.min_samples_split = params.get('min_samples_split', 2)
        self.min_impurity_decrease = params.get('min_impurity_decrease', 0.0)
        self.random_state = params.get('random_state', 42)
        self.n_pair = params.get('n_pair', 2)
        self.gamma = params.get('gamma', 1.0)
        self.max_iter = params.get('max_iter', 100)
        self.relative_change = params.get('relative_change', 0.001)
        self.categories = params.get('categories', [])
        self.ccp_alpha = params.get('ccp_alpha', 0.0)
        self.use_oblique = params.get('use_oblique', True)

        self.cat_ = params.get('cat_', False)
        self.rng_ = np.random.default_rng(self.random_state)

        self.task = params["task"]
        self.n_classes = params['n_classes']
        self.n_features = params['n_features']

        self.root = deserialize_tree(tree, self.n_features, self.n_classes)

    cpdef fit(self, double[::1, :] X, double[::1] y, double[::1] sample_weight):
        cdef int n_samples = X.shape[0]
        cdef int n_columns = X.shape[1]
        cdef int i
        cdef int max_unique_range = 0
        
        # Temel bellek ayırma işlemleri
        cdef SortItem* sort_buffer = NULL
        cdef int* sample_indices = NULL
        cdef int* nan_indices = NULL
        cdef bint* is_categorical = NULL
        cdef CategoryStat* categorical_stats = NULL
        cdef int* count_unique_array = NULL
        cdef int* min_values = NULL
        cdef bint* is_integer = NULL
        cdef SortItem* count_sort_buffer = NULL
        
    
        try:
            self.n_features = n_columns

            if self.root != NULL:
                free_tree(self.root)
                self.root = NULL

            # Bellek ayırma işlemleri
            sort_buffer = <SortItem*>malloc(n_samples * sizeof(SortItem))
            sample_indices = <int*>malloc(n_samples * sizeof(int))
            nan_indices = <int*>malloc(n_samples * sizeof(int))
            is_categorical = <bint*>calloc(n_columns, sizeof(bint))
            min_values = <int*>malloc(n_columns * sizeof(int))
            is_integer = <bint*>malloc(n_columns * sizeof(bint))

            if ((sort_buffer == NULL) or (sample_indices == NULL) or 
                (nan_indices == NULL) or (is_categorical == NULL) or
                 (min_values == NULL) or (is_integer == NULL)
                ):
                raise MemoryError()

            analyze_X(X, is_integer, min_values, &max_unique_range)

            if max_unique_range != 0:
                count_unique_array = <int*>malloc(max_unique_range * sizeof(int))
                count_sort_buffer = <SortItem*>malloc(n_samples * sizeof(SortItem))

                if (count_unique_array == NULL) or (count_sort_buffer == NULL):
                    raise MemoryError()

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
                self.gamma,
                self.max_iter, 
                self.relative_change,
                self.rng_,
                self.use_oblique,
                is_categorical,
                categorical_stats,
                sample_weight,
                min_values,
                max_unique_range,
                count_unique_array,
                count_sort_buffer,
                is_integer,
            )
            
            if self.ccp_alpha > 0.0:
                prune_tree(self.root, self.ccp_alpha)
                
        finally:
            free(sort_buffer)
            free(sample_indices)
            free(nan_indices)
            free(is_categorical)
            free(min_values)
            free(is_integer)

            if count_sort_buffer != NULL:
                free(count_sort_buffer)

            if count_unique_array != NULL:
                free(count_unique_array)

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
        cdef int* indices = NULL
    
        try:
            if self.root == NULL:
                raise ValueError("The model has not been fitted yet. Call the 'fit' method before using this model.")

            if self.n_features != n_features:
                raise ValueError(f"Mismatch in number of features: expected {self.n_features}, but got {n_features}.")

            if self.n_classes > 2:
                out = np.zeros((n_samples, self.n_classes), dtype=np.float64)
            else:
                out = np.zeros((n_samples, 1), dtype=np.float64)

            indices = <int*>malloc(n_samples * sizeof(int))
            if not indices:
                raise MemoryError()

            for i in range(n_samples):
                indices[i] = i

            predict(self.root, X, out, indices, n_samples, self.n_classes)

            if (self.n_classes <= 2) and (self.task == 0):
                proba = np.empty((n_samples, self.n_classes), dtype=np.float64)
                for i in range(n_samples):
                    proba[i, 1] = out[i, 0]           # prob of class 1
                    proba[i, 0] = 1.0 - out[i, 0]     # prob of class 0

                return proba
            else:
                return out
        finally:
            free(indices)
