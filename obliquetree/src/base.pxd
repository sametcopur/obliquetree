from .tree cimport TreeNode


cdef class TreeClassifier:
    cdef TreeNode* root
    cdef public unsigned char max_depth
    cdef public int min_samples_leaf
    cdef public int min_samples_split
    cdef public double min_impurity_decrease
    cdef public int random_state
    cdef public int n_pair
    cdef public int top_k
    cdef public object rng_
    cdef public double gamma
    cdef public int max_iter 
    cdef public double relative_change
    cdef public bint use_oblique
    cdef public list categories
    cdef public double ccp_alpha
    cdef public bint task
    cdef public int n_classes
    cdef bint cat_ 
    cdef public int n_features

    cpdef fit(self, double[::1, :] X, double[::1] y, double[::1] sample_weight)
    cpdef apply(self, double[::1, :] X)
    cpdef predict(self, double[::1, :] X)
