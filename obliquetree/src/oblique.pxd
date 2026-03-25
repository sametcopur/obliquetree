cimport numpy as np
from .tree cimport SortItem

cdef int* prepare_candidate_pairs(
                const bint task,
                const int n_classes,
                np.ndarray[double, ndim=2] X,
                np.ndarray[double, ndim=1] y,
                np.ndarray[double, ndim=1] sample_weight,
                const int* sample_indices,
                const int n_samples,
                const int n_pair,
                const int top_k,
                const bint* is_categorical,
                Py_ssize_t* out_n_fp) noexcept

cdef void fill_oblique_sort_buffer(
                np.ndarray[double, ndim=2] X,
                const int* sample_indices,
                SortItem* sort_buffer,
                const int n_samples,
                const int n_pair,
                const int* pair_idx,
                const double* x) noexcept

cdef tuple[double*, int*] analyze_from_pairs(
                const bint task,
                const int n_classes,
                const bint linear,
                np.ndarray[double, ndim=2] X, 
                np.ndarray[double, ndim=1] y,
                np.ndarray[double, ndim=1] sample_weight, 
                const int* sample_indices,
                SortItem* sort_buffer,
                const int n_samples,
                const int n_pair,
                const int* pair_idx,
                const Py_ssize_t n_fp,
                object rng,
                const double gamma, 
                const int maxiter,
                const double relative_change) noexcept

cdef void analyze_both_from_pairs(
                const bint task,
                const int n_classes,
                np.ndarray[double, ndim=2] X,
                np.ndarray[double, ndim=1] y,
                np.ndarray[double, ndim=1] sample_weight,
                const int* sample_indices,
                SortItem* sort_buffer,
                const int n_samples,
                const int n_pair,
                const int* pair_idx,
                const Py_ssize_t n_fp,
                object rng,
                const double gamma,
                const int maxiter,
                const double relative_change,
                double** out_oblique_x,
                int** out_oblique_pair,
                double** out_linear_x,
                int** out_linear_pair) noexcept

cdef tuple[double*, int*] analyze(
                const bint task,
                const int n_classes,
                const bint linear,
                np.ndarray[double, ndim=2] X, 
                np.ndarray[double, ndim=1] y,
                np.ndarray[double, ndim=1] sample_weight, 
                const int* sample_indices,
                SortItem* sort_buffer,
                const int n_samples,
                const int n_pair,
                const int top_k,
                const bint* is_categorical,
                object rng,
                const double gamma, 
                const int maxiter,
                const double relative_change) noexcept
