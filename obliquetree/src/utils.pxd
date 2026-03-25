from .base cimport TreeClassifier
from .tree cimport TreeNode, CategoryStat

cdef struct SortItem:
    double value
    int index

#typedefs 
ctypedef unsigned char uint8_t
ctypedef unsigned int uint32_t
ctypedef unsigned long long uint64_t
ctypedef Py_ssize_t intp_t
ctypedef float float32_t
ctypedef double float64_t
ctypedef signed char int8_t
ctypedef signed int int32_t
ctypedef signed long long int64_t

cpdef dict export_tree(TreeClassifier tree)
cdef void sort_pointer_array(SortItem* items, const int n_samples) noexcept nogil
cdef TreeNode* deserialize_tree(dict tree_dict, int n_features, int n_classes)
cdef void sort_category_stats(CategoryStat* stats, const int unique_count) noexcept nogil
cdef void sort_category_stats_multiclass(
    CategoryStat* stats, 
    const int* permutation_order,
    const int unique_count, 
    const int n_classes) noexcept nogil
cdef void populate_indices(
    const bint missing_go_left,
    const int n_samples,
    const int left_count,
    const int right_count,
    const int n_nans,
    int* left_indices,
    int* right_indices,
    const int* best_sorted_indices,
    const int* best_nan_indices) noexcept nogil

cdef void sort_pointer_array_count(SortItem* items, int* count, SortItem* output, const int n_samples, const int n_unique, const int offset) noexcept nogil
cdef void analyze_X(const double[::1, :] X, bint* is_integer)
cdef int** generate_permutations(const int n_classes, int* perm_count) noexcept nogil
