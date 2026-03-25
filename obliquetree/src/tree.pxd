from .utils cimport SortItem

cdef struct TreeNode:
    bint is_leaf
    double value
    double* value_multiclass
    int n_classes
    int feature_idx
    double threshold
    TreeNode* left
    TreeNode* right
    int n_pair 
    double* x
    int* pair
    int missing_go_left
    int n_category
    int* categories_go_left
    double impurity
    int n_samples

cdef struct CategoryStat:
    double value
    double y_sum           # Weighted sum of y values
    double* class_weights # Multiclass için sınıf ağırlıkları
    double weight
    int count

cdef void predict(
    const TreeNode* node,
    const double[::1, :] X,
    double[:, ::1] out,
    const int* indices,
    const int n_samples,
    const int n_classes,
) noexcept nogil

cdef void apply(
    const TreeNode* node,
    const double[::1, :] X,
    int[::1] out,
    const int n_samples,
) noexcept nogil

cdef TreeNode* build_tree_recursive(
    const bint task,
    const int n_classes,
    const double[::1, :] X,
    const double[::1] y,
    const unsigned char depth,
    const unsigned char max_depth,
    const int min_samples_leaf,
    const int min_samples_split,
    const double min_impurity_decrease,
    SortItem* sort_buffer,
    int* sample_indices,
    int* nan_indices,
    const int n_samples,
    const int n_pair,
    const double gamma,
    const int max_iter,
    const double relative_change,
    object rng,
    const bint use_oblique,
    const bint* is_categorical,
    CategoryStat* categorical_stats,
    const double[::1] sample_weight,
    const int* min_values,
    const int max_unique_range,
    int* count_unique_array,
    SortItem* count_sort_buffer,
    const bint* is_integer,
)  noexcept nogil
