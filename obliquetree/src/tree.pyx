from cython.parallel cimport prange

from libc.math cimport INFINITY
from libc.stdlib cimport malloc, free

from .metric cimport (
    calculate_impurity,
    calculate_node_gini,
    calculate_node_mse,
    calculate_node_value,
    calculate_node_value_multiclass,
)
from .oblique cimport analyze
from .utils cimport sort_pointer_array, sort_pointer_array_count


cdef inline void free_oblique_split(double** x, int** pair) noexcept nogil:
    if x[0] != NULL:
        free(x[0])
        x[0] = NULL
    if pair[0] != NULL:
        free(pair[0])
        pair[0] = NULL


cdef inline bint sample_goes_left(
    const double[::1, :] X,
    const int sample_idx,
    const int feature_idx,
    const double threshold,
    const bint missing_go_left,
    const int n_pair,
    const double* x,
    const int* pair,
    const int n_category,
    const int* categories_go_left,
) noexcept nogil:
    cdef int j
    cdef double x_val
    cdef double calc

    if feature_idx == -2:
        calc = 0.0
        for j in range(n_pair):
            calc += X[sample_idx, pair[j]] * x[j]
        return calc <= threshold

    x_val = X[sample_idx, feature_idx]
    if x_val != x_val:
        return missing_go_left

    if n_category > 0:
        for j in range(n_category):
            if x_val == categories_go_left[j]:
                return True
        return False

    return x_val <= threshold


cdef int partition_samples_inplace(
    const double[::1, :] X,
    int* sample_indices,
    const int n_samples,
    const int feature_idx,
    const double threshold,
    const bint missing_go_left,
    const int n_pair,
    const double* x,
    const int* pair,
    const int n_category,
    const int* categories_go_left,
) noexcept nogil:
    cdef int left = 0
    cdef int right = n_samples - 1
    cdef int tmp

    while left <= right:
        while left <= right and sample_goes_left(
            X,
            sample_indices[left],
            feature_idx,
            threshold,
            missing_go_left,
            n_pair,
            x,
            pair,
            n_category,
            categories_go_left,
        ):
            left += 1

        while left <= right and not sample_goes_left(
            X,
            sample_indices[right],
            feature_idx,
            threshold,
            missing_go_left,
            n_pair,
            x,
            pair,
            n_category,
            categories_go_left,
        ):
            right -= 1

        if left < right:
            tmp = sample_indices[left]
            sample_indices[left] = sample_indices[right]
            sample_indices[right] = tmp
            left += 1
            right -= 1

    return left


cdef double evaluate_feature_exact(
    const bint task,
    const int n_classes,
    const double[::1, :] X,
    const double* y,
    const double* sample_weight,
    const int* sample_indices,
    const int n_samples,
    const int feature_idx,
    const bint is_categorical_feature,
    const int min_samples_leaf,
    const int min_value,
    const int max_unique_range,
    const bint is_integer_feature,
    double* out_threshold,
    int* out_left_count,
    bint* out_missing_go_left,
    int* out_status,
) noexcept nogil:
    cdef SortItem* local_sort_buffer = NULL
    cdef SortItem* local_count_sort_buffer = NULL
    cdef int* local_nan_indices = NULL
    cdef int* local_count_unique_array = NULL
    cdef CategoryStat* local_categorical_stats = NULL
    cdef int i
    cdef int idx
    cdef int n_nans = 0
    cdef int n_non_nans = 0
    cdef double x_val
    cdef double impurity = INFINITY

    out_status[0] = 0
    out_threshold[0] = 0.0
    out_left_count[0] = 0
    out_missing_go_left[0] = True

    local_sort_buffer = <SortItem*>malloc(n_samples * sizeof(SortItem))
    local_nan_indices = <int*>malloc(n_samples * sizeof(int))

    if local_sort_buffer == NULL or local_nan_indices == NULL:
        out_status[0] = 1
        if local_sort_buffer != NULL:
            free(local_sort_buffer)
        if local_nan_indices != NULL:
            free(local_nan_indices)
        return INFINITY

    if is_categorical_feature:
        local_categorical_stats = <CategoryStat*>malloc(n_samples * sizeof(CategoryStat))
        if local_categorical_stats == NULL:
            out_status[0] = 1
            free(local_sort_buffer)
            free(local_nan_indices)
            return INFINITY

    if is_integer_feature:
        local_count_unique_array = <int*>malloc(max_unique_range * sizeof(int))
        local_count_sort_buffer = <SortItem*>malloc(n_samples * sizeof(SortItem))
        if local_count_unique_array == NULL or local_count_sort_buffer == NULL:
            out_status[0] = 1
            if local_count_unique_array != NULL:
                free(local_count_unique_array)
            if local_count_sort_buffer != NULL:
                free(local_count_sort_buffer)
            if local_categorical_stats != NULL:
                free(local_categorical_stats)
            free(local_sort_buffer)
            free(local_nan_indices)
            return INFINITY

    for i in range(n_samples):
        idx = sample_indices[i]
        x_val = X[idx, feature_idx]

        if x_val != x_val:
            local_nan_indices[n_nans] = idx
            n_nans += 1
        else:
            local_sort_buffer[n_non_nans].value = x_val
            local_sort_buffer[n_non_nans].index = idx
            n_non_nans += 1

    if n_non_nans >= 2 * min_samples_leaf:
        if is_integer_feature:
            sort_pointer_array_count(
                local_sort_buffer,
                local_count_unique_array,
                local_count_sort_buffer,
                n_non_nans,
                max_unique_range,
                min_value,
            )
        else:
            sort_pointer_array(local_sort_buffer, n_non_nans)

        impurity = calculate_impurity(
            is_categorical_feature,
            n_classes,
            local_sort_buffer,
            sample_weight,
            y,
            local_nan_indices,
            local_categorical_stats,
            n_samples,
            n_nans,
            min_samples_leaf,
            out_threshold,
            out_left_count,
            out_missing_go_left,
            task,
        )

    if local_count_unique_array != NULL:
        free(local_count_unique_array)
    if local_count_sort_buffer != NULL:
        free(local_count_sort_buffer)
    if local_categorical_stats != NULL:
        free(local_categorical_stats)
    free(local_sort_buffer)
    free(local_nan_indices)

    return impurity


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
) noexcept nogil:
    cdef TreeNode* node = <TreeNode*>malloc(sizeof(TreeNode))
    cdef double min_impurity = INFINITY
    cdef double improvement
    cdef double impurity_c
    cdef double threshold_c
    cdef double linear_impurity
    cdef double* best_oblique_x = NULL
    cdef double* oblique_x = NULL
    cdef double* linear_x = NULL
    cdef double* feature_thresholds = NULL
    cdef double* feature_impurities = NULL
    cdef int* best_oblique_pair = NULL
    cdef int* oblique_pair = NULL
    cdef int* linear_pair = NULL
    cdef int* feature_left_counts = NULL
    cdef int* feature_status = NULL
    cdef bint* feature_missing_left = NULL
    cdef int n_features = X.shape[1]
    cdef int f_idx
    cdef int i
    cdef int idx
    cdef int n_nans
    cdef int n_non_nans
    cdef int left_count_c = 0
    cdef int right_count
    cdef int actual_left_count
    cdef int best_feature = -1
    cdef int best_left_count = 0
    cdef int best_n_categorical = 0
    cdef bint missing_go_left = True
    cdef bint best_missing_go_left = True
    cdef double best_threshold = 0.0
    cdef double x_val

    if node == NULL:
        with gil:
            raise MemoryError()

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
        calculate_node_value_multiclass(
            sample_weight,
            y,
            sample_indices,
            n_samples,
            n_classes,
            &node.value_multiclass,
        )
    else:
        node.value = calculate_node_value(sample_weight, y, sample_indices, n_samples)

    if n_samples == 1:
        return node

    if task == 0:
        node.impurity = calculate_node_gini(sample_indices, sample_weight, y, n_samples, n_classes)
    else:
        node.impurity = calculate_node_mse(sample_indices, sample_weight, y, n_samples)

    if (
        node.impurity == 0.0 or
        depth >= max_depth or
        n_samples < 2 * min_samples_leaf or
        n_samples < min_samples_split
    ):
        return node

    if use_oblique:
        with gil:
            oblique_x, oblique_pair = analyze(
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
            &missing_go_left,
            task,
        )

        with gil:
            linear_x, linear_pair = analyze(
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

        if linear_x != NULL and linear_pair != NULL:
            linear_impurity = calculate_impurity(
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
                task,
            )

            if linear_impurity < impurity_c:
                free_oblique_split(&oblique_x, &oblique_pair)
                best_oblique_x = linear_x
                best_oblique_pair = linear_pair
                linear_x = NULL
                linear_pair = NULL
                min_impurity = linear_impurity
            else:
                free_oblique_split(&linear_x, &linear_pair)
                best_oblique_x = oblique_x
                best_oblique_pair = oblique_pair
                oblique_x = NULL
                oblique_pair = NULL
                min_impurity = impurity_c
                best_threshold = threshold_c
                best_left_count = left_count_c
                best_missing_go_left = missing_go_left

            best_feature = -2
        else:
            free_oblique_split(&linear_x, &linear_pair)
            best_oblique_x = oblique_x
            best_oblique_pair = oblique_pair
            oblique_x = NULL
            oblique_pair = NULL
            min_impurity = impurity_c
            best_threshold = threshold_c
            best_left_count = left_count_c
            best_missing_go_left = missing_go_left
            best_feature = -2

    feature_thresholds = <double*>malloc(n_features * sizeof(double))
    feature_impurities = <double*>malloc(n_features * sizeof(double))
    feature_left_counts = <int*>malloc(n_features * sizeof(int))
    feature_status = <int*>malloc(n_features * sizeof(int))
    feature_missing_left = <bint*>malloc(n_features * sizeof(bint))

    if (
        feature_thresholds == NULL or
        feature_impurities == NULL or
        feature_left_counts == NULL or
        feature_status == NULL or
        feature_missing_left == NULL
    ):
        free(feature_thresholds)
        free(feature_impurities)
        free(feature_left_counts)
        free(feature_status)
        free(feature_missing_left)
        free_oblique_split(&best_oblique_x, &best_oblique_pair)
        with gil:
            raise MemoryError()

    for f_idx in range(n_features):
        feature_thresholds[f_idx] = 0.0
        feature_impurities[f_idx] = INFINITY
        feature_left_counts[f_idx] = 0
        feature_status[f_idx] = 0
        feature_missing_left[f_idx] = True

    for f_idx in prange(n_features, nogil=True, schedule="guided", use_threads_if=n_features > 1):
        feature_impurities[f_idx] = evaluate_feature_exact(
            task,
            n_classes,
            X,
            &y[0],
            &sample_weight[0],
            sample_indices,
            n_samples,
            f_idx,
            is_categorical[f_idx],
            min_samples_leaf,
            min_values[f_idx],
            max_unique_range,
            is_integer[f_idx],
            &feature_thresholds[f_idx],
            &feature_left_counts[f_idx],
            &feature_missing_left[f_idx],
            &feature_status[f_idx],
        )

    for f_idx in range(n_features):
        if feature_status[f_idx] != 0:
            free(feature_thresholds)
            free(feature_impurities)
            free(feature_left_counts)
            free(feature_status)
            free(feature_missing_left)
            free_oblique_split(&best_oblique_x, &best_oblique_pair)
            with gil:
                raise MemoryError()

        if feature_impurities[f_idx] < min_impurity:
            min_impurity = feature_impurities[f_idx]
            best_feature = f_idx
            best_threshold = feature_thresholds[f_idx]
            best_left_count = feature_left_counts[f_idx]
            best_missing_go_left = feature_missing_left[f_idx]
            best_n_categorical = 0
            free_oblique_split(&best_oblique_x, &best_oblique_pair)

    free(feature_thresholds)
    free(feature_impurities)
    free(feature_left_counts)
    free(feature_status)
    free(feature_missing_left)

    if best_feature >= 0 and is_categorical[best_feature]:
        n_nans = 0
        n_non_nans = 0

        for i in range(n_samples):
            idx = sample_indices[i]
            x_val = X[idx, best_feature]

            if x_val != x_val:
                nan_indices[n_nans] = idx
                n_nans += 1
            else:
                sort_buffer[n_non_nans].value = x_val
                sort_buffer[n_non_nans].index = idx
                n_non_nans += 1

        if n_non_nans >= 2 * min_samples_leaf:
            if is_integer[best_feature]:
                sort_pointer_array_count(
                    sort_buffer,
                    count_unique_array,
                    count_sort_buffer,
                    n_non_nans,
                    max_unique_range,
                    min_values[best_feature],
                )
            else:
                sort_pointer_array(sort_buffer, n_non_nans)

            impurity_c = calculate_impurity(
                True,
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
                task,
            )

            if impurity_c < INFINITY:
                best_threshold = threshold_c
                best_left_count = left_count_c
                best_missing_go_left = missing_go_left
                best_n_categorical = <int>threshold_c
                node.n_category = best_n_categorical

                if node.n_category > 0:
                    node.categories_go_left = <int*>malloc(node.n_category * sizeof(int))
                    if node.categories_go_left == NULL:
                        free_oblique_split(&best_oblique_x, &best_oblique_pair)
                        with gil:
                            raise MemoryError()

                    for i in range(node.n_category):
                        node.categories_go_left[i] = <int>categorical_stats[i].value

    improvement = node.impurity - min_impurity
    right_count = n_samples - best_left_count

    if (
        best_feature == -1 or
        improvement <= 0.0 or
        improvement < min_impurity_decrease or
        best_left_count == 0 or
        right_count == 0
    ):
        if node.categories_go_left != NULL:
            free(node.categories_go_left)
            node.categories_go_left = NULL
        node.n_category = 0
        free_oblique_split(&best_oblique_x, &best_oblique_pair)
        return node

    node.is_leaf = False
    node.feature_idx = best_feature
    node.threshold = best_threshold
    node.missing_go_left = best_missing_go_left
    node.n_category = best_n_categorical

    if best_feature == -2:
        node.x = best_oblique_x
        node.pair = best_oblique_pair
        best_oblique_x = NULL
        best_oblique_pair = NULL

    actual_left_count = partition_samples_inplace(
        X,
        sample_indices,
        n_samples,
        node.feature_idx,
        node.threshold,
        node.missing_go_left,
        node.n_pair,
        node.x,
        node.pair,
        node.n_category,
        node.categories_go_left,
    )

    right_count = n_samples - actual_left_count
    if actual_left_count == 0 or right_count == 0:
        node.is_leaf = True
        node.feature_idx = -1
        node.threshold = 0.0
        node.missing_go_left = True
        if node.x != NULL:
            free(node.x)
            node.x = NULL
        if node.pair != NULL:
            free(node.pair)
            node.pair = NULL
        if node.categories_go_left != NULL:
            free(node.categories_go_left)
            node.categories_go_left = NULL
        node.n_category = 0
        free_oblique_split(&best_oblique_x, &best_oblique_pair)
        return node

    node.left = build_tree_recursive(
        task,
        n_classes,
        X,
        y,
        depth + 1,
        max_depth,
        min_samples_leaf,
        min_samples_split,
        min_impurity_decrease,
        sort_buffer,
        sample_indices,
        nan_indices,
        actual_left_count,
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
        is_integer,
    )

    node.right = build_tree_recursive(
        task,
        n_classes,
        X,
        y,
        depth + 1,
        max_depth,
        min_samples_leaf,
        min_samples_split,
        min_impurity_decrease,
        sort_buffer,
        sample_indices + actual_left_count,
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
        is_integer,
    )

    free_oblique_split(&best_oblique_x, &best_oblique_pair)
    return node


cdef void predict(
    const TreeNode* node,
    const double[::1, :] X,
    double[:, ::1] out,
    const int* indices,
    const int n_samples,
    const int n_classes,
) noexcept nogil:
    cdef int i
    cdef int j
    cdef int idx
    cdef int left_count = 0
    cdef int right_count = 0
    cdef int* left_indices = NULL
    cdef int* right_indices = NULL

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

    left_indices = <int*>malloc(n_samples * sizeof(int))
    right_indices = <int*>malloc(n_samples * sizeof(int))

    if left_indices == NULL or right_indices == NULL:
        if left_indices != NULL:
            free(left_indices)
        if right_indices != NULL:
            free(right_indices)
        with gil:
            raise MemoryError()

    for i in range(n_samples):
        idx = indices[i]
        if sample_goes_left(
            X,
            idx,
            node.feature_idx,
            node.threshold,
            node.missing_go_left,
            node.n_pair,
            node.x,
            node.pair,
            node.n_category,
            node.categories_go_left,
        ):
            left_indices[left_count] = idx
            left_count += 1
        else:
            right_indices[right_count] = idx
            right_count += 1

    if left_count > 0:
        predict(node.left, X, out, left_indices, left_count, n_classes)
    if right_count > 0:
        predict(node.right, X, out, right_indices, right_count, n_classes)

    free(left_indices)
    free(right_indices)
