from libc.stdlib cimport malloc, free
from libc.string cimport memcpy
from libc.math cimport INFINITY

from .oblique cimport analyze
from .utils cimport sort_pointer_array, populate_indices, sort_pointer_array_count
from .metric cimport (calculate_impurity,
                        calculate_node_gini, 
                        calculate_node_value, 
                        calculate_node_mse,

                        calculate_node_value_multiclass)


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
    int *nan_indices,
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
    const bint* is_integer) noexcept nogil:
    """
    Recursively build a binary decision tree.
    """
    # Create a node
    cdef TreeNode* node = <TreeNode*>malloc(sizeof(TreeNode))
    if node == NULL:
        with gil:
            raise MemoryError()

    # Initialize as leaf
    node.is_leaf = True
    node.value = 0.0
    node.feature_idx = -1
    node.threshold = 0.0
    node.left = NULL
    node.right = NULL
    node.n_pair = n_pair
    node.pair = NULL
    node.x = NULL
    node.missing_go_left = True
    node.n_category = 0
    node.categories_go_left = NULL
    node.n_samples = n_samples
    node.n_classes = n_classes
    node.value_multiclass = NULL

    if n_classes > 2:
        calculate_node_value_multiclass(sample_weight, y, sample_indices, n_samples, n_classes, &node.value_multiclass)
    else:
        node.value = calculate_node_value(sample_weight, y, sample_indices, n_samples)

    # If 1 sample => leaf
    if n_samples == 1:
        return node

    if task == 0:
        node.impurity = calculate_node_gini(sample_indices, sample_weight, y, n_samples, n_classes)
    else:
        node.impurity = calculate_node_mse(sample_indices, sample_weight, y, n_samples)


    # Check stopping criteria
    if (
        node.impurity == 0.0 or
        depth >= max_depth or
        n_samples < 2 * min_samples_leaf or 
        n_samples < min_samples_split
    ):
        return node

    # We'll store the sorted indices for whichever feature gave the best split
    cdef int* best_sorted_indices = <int*>malloc(n_samples * sizeof(int))
    cdef int* best_nan_indices = <int*>malloc(n_samples * sizeof(int))
    cdef int* best_categorical_indices = <int*>malloc(n_samples * sizeof(int))

    if best_sorted_indices == NULL:
        with gil:
            raise MemoryError()

    cdef double min_impurity = INFINITY
    cdef int best_feature = -1
    cdef double best_threshold = 0.0 
    cdef int best_left_count = 0
    cdef bint best_missing_go_left = True
    cdef int best_n_categorical = 0
    cdef double impurity_c = 0.0, threshold_c = 0.0
    cdef int left_count_c, f_idx, n_features = X.shape[1]
    cdef double* best_x
    cdef int* pair
    cdef int n_nans, n_non_nans, best_n_nans
    cdef bint missing_go_left = True
    cdef double x_val
    cdef int i, unique, idx

    if use_oblique:
        node.feature_idx = best_feature = -2
        best_n_nans = 0

        with gil:
            best_x, pair = analyze(
                            task,
                            n_classes,
                            False,
                            X.base, 
                            y.base, 
                            sample_weight.base, 
                            sample_indices, 
                            sort_buffer, 
                            n_samples, 
                            n_pair, 
                            is_categorical, 
                            rng, 
                            gamma, 
                            max_iter, 
                            relative_change,
                            )

        min_impurity = calculate_impurity(
            False,
            n_classes,
            sort_buffer,
            &sample_weight[0],
            &y[0],
            nan_indices,
            categorical_stats,
            n_samples,
            0,
            min_samples_leaf,
            &best_threshold,
            &best_left_count,
            &best_missing_go_left,
            task)


        for i in range(n_samples):
            best_sorted_indices[i] = sort_buffer[i].index

        with gil:
            best_x_linear, pair_linear = analyze(
                            task,
                            n_classes,
                            True,
                            X.base, 
                            y.base, 
                            sample_weight.base, 
                            sample_indices, 
                            sort_buffer, 
                            n_samples, 
                            n_pair, 
                            is_categorical, 
                            rng, 
                            gamma, 
                            max_iter, 
                            relative_change,
                            )

        impurity_c = calculate_impurity(
            False,
            n_classes,
            sort_buffer,
            &sample_weight[0],
            &y[0],
            nan_indices,
            categorical_stats,
            n_samples,
            0,
            min_samples_leaf,
            &threshold_c,
            &left_count_c,
            &best_missing_go_left,
            task)

        # Eğer linear split daha iyiyse
        if impurity_c < min_impurity:
            # Önceki non-linear sonuçları temizle
            if node.x != NULL:
                free(node.x)
            if node.pair != NULL:
                free(node.pair)
            
            node.x = best_x_linear
            node.pair = pair_linear

            # Önceki malloc edilmiş belleği temizle
            free(best_x)
            free(pair)

            min_impurity = impurity_c
            best_threshold = threshold_c
            best_left_count = left_count_c

            for i in range(n_samples):
                best_sorted_indices[i] = sort_buffer[i].index

        else:
            # Linear split daha kötüyse, linear split sonuçlarını temizle
            if best_x_linear != NULL:
                free(best_x_linear)
            if pair_linear != NULL:
                free(pair_linear)

            node.x = best_x
            node.pair = pair


    for f_idx in range(n_features):
        n_nans = 0
        n_non_nans = 0

        for i in range(n_samples):
            idx = sample_indices[i]
            x_val = X[idx, f_idx] 

            if x_val != x_val:
                nan_indices[n_nans] = idx
                n_nans += 1 

            else:
                sort_buffer[n_non_nans].value = x_val
                sort_buffer[n_non_nans].index = idx
                n_non_nans += 1
                
        if n_non_nans < 2 * min_samples_leaf:
            continue

        if is_integer[f_idx]:
            sort_pointer_array_count(sort_buffer, count_unique_array, count_sort_buffer, n_non_nans, max_unique_range, min_values[f_idx])
        else:
            sort_pointer_array(sort_buffer, n_non_nans)


        impurity_c = calculate_impurity(
                    is_categorical[f_idx], 
                    n_classes,
                    sort_buffer,
                    &sample_weight[0],
                    &y[0],
                    nan_indices,
                    categorical_stats,
                    n_samples,
                    n_nans,
                    min_samples_leaf,
                    &threshold_c,
                    &left_count_c,
                    &missing_go_left,
                    task)

        if impurity_c < min_impurity:
            min_impurity = impurity_c
            best_feature = f_idx
            best_threshold = threshold_c
            best_left_count = left_count_c
            best_missing_go_left = missing_go_left
            best_n_nans = n_nans

            if is_categorical[f_idx]:
                best_n_categorical = <int>threshold_c

                for i in range(<int>threshold_c):
                    best_categorical_indices[i] = <int>categorical_stats[i].value

            else:
                best_n_categorical = 0

            for i in range(n_non_nans):
                best_sorted_indices[i] = sort_buffer[i].index
            
            memcpy(best_nan_indices, nan_indices, n_nans * sizeof(int))

    cdef double improvement = node.impurity - min_impurity
    cdef int right_count = n_samples - best_left_count

    if ((best_feature == -1 or improvement <= 0.0) or 
            (improvement < min_impurity_decrease) or 
            (best_left_count == 0 or right_count == 0)):

        free(best_sorted_indices)
        free(best_nan_indices)
        free(best_categorical_indices)
        return node

    cdef int* left_indices = <int*>malloc(best_left_count * sizeof(int))
    cdef int* right_indices = <int*>malloc(right_count * sizeof(int))

    if left_indices == NULL or right_indices == NULL:
        if left_indices != NULL:
            free(left_indices)
        if right_indices != NULL:
            free(right_indices)
        free(best_sorted_indices)
        free(best_nan_indices)
        free(best_categorical_indices)
        with gil:
            raise MemoryError()

    # Mark node as split
    node.is_leaf = False
    node.feature_idx = best_feature
    node.threshold = best_threshold
    node.missing_go_left = best_missing_go_left
    node.n_category = best_n_categorical

    if node.n_category != 0:
        node.categories_go_left = <int*>malloc(node.n_category * sizeof(int))

        for i in range(node.n_category):
            node.categories_go_left[i] = best_categorical_indices[i]

        
    populate_indices(
        best_missing_go_left, 
        n_samples, 
        best_left_count, 
        right_count, 
        best_n_nans, 
        left_indices, 
        right_indices, 
        best_sorted_indices, 
        best_nan_indices)

    free(best_sorted_indices)
    free(best_nan_indices)
    free(best_categorical_indices)

    # Recurse left
    node.left = build_tree_recursive(
        task,
        n_classes,
        X, y,
        depth + 1,
        max_depth,
        min_samples_leaf,
        min_samples_split,
        min_impurity_decrease,
        sort_buffer,
        left_indices,
        nan_indices,
        best_left_count,
        n_pair,
        gamma,
        max_iter,
        relative_change,
        rng,
        use_oblique,
        is_categorical,
        categorical_stats,
        sample_weight,
        min_values,
        max_unique_range,
        count_unique_array,
        count_sort_buffer,
        is_integer
    )

    # Recurse right
    node.right = build_tree_recursive(
        task,
        n_classes,
        X, y,
        depth + 1,
        max_depth,
        min_samples_leaf,
        min_samples_split,
        min_impurity_decrease,
        sort_buffer,
        right_indices,
        nan_indices,
        right_count,
        n_pair,
        gamma,
        max_iter,
        relative_change,
        rng,
        use_oblique,
        is_categorical,
        categorical_stats,
        sample_weight, 
        min_values,
        max_unique_range,
        count_unique_array,
        count_sort_buffer,
        is_integer
    )

    free(left_indices)
    free(right_indices)

    return node

cdef void predict(
    const TreeNode* node,
    const double[::1, :] X,
    double[:, ::1] out,
    const int* indices,
    const int n_samples,
    const int n_classes) noexcept nogil:
    """
    Recursively fill out[indices[i]] with node.value if leaf,
    otherwise route samples to left/right.
    """
    cdef int i, j, idx

    if node.is_leaf:
        if n_classes <= 2:

            for i in range(n_samples):
                out[indices[i], 0] = node.value

        else:
            for i in range(n_samples):
                idx = indices[i]
                for j in range(n_classes):
                    out[idx, j] = node.value_multiclass[j]
               
        return


    cdef int left_count = 0
    cdef int right_count = 0
    cdef double calc
    cdef int ind
    cdef double x_val
    cdef bint goes_left

    cdef int* left_indices = <int*>malloc(n_samples * sizeof(int))
    cdef int* right_indices = <int*>malloc(n_samples * sizeof(int))

    for i in range(n_samples):
        ind = indices[i]

        if node.feature_idx == -2:  # Oblique split
            calc = 0.0
            for j in range(node.n_pair):
                x_val = X[ind, node.pair[j]]
                calc += x_val * node.x[j]
            
            goes_left = calc <= node.threshold
            
        elif node.n_category > 0:  # Kategorik split
            x_val = X[ind, node.feature_idx]
            goes_left = False
            
            for j in range(node.n_category):
                if x_val == node.categories_go_left[j]:
                    goes_left = True
                    break
                        
        else:  # Normal numerik split
            x_val = X[ind, node.feature_idx]
            if x_val != x_val:
                goes_left = node.missing_go_left
            else:
                goes_left = x_val <= node.threshold

        if goes_left:
            left_indices[left_count] = ind
            left_count += 1
        else:
            right_indices[right_count] = ind
            right_count += 1

    if left_count > 0:
        predict(node.left, X, out, left_indices, left_count ,n_classes)
    if right_count > 0:
        predict(node.right, X, out, right_indices, right_count, n_classes)

    free(left_indices)
    free(right_indices)

