from cython.parallel cimport prange

from libc.limits cimport INT_MAX, INT_MIN
from libc.math cimport INFINITY
from libc.stdlib cimport calloc, malloc, free

from .metric cimport (
    calculate_impurity,
    calculate_node_gini,
    calculate_node_mse,
    calculate_node_value,
    calculate_node_value_multiclass,
)
from .oblique cimport analyze_both_from_pairs, prepare_candidate_pairs
from .utils cimport sort_pointer_array, sort_pointer_array_count

DEF COUNT_SORT_RANGE_FACTOR = 8
DEF MIN_PARALLEL_FEATURE_WORK = 32768


cdef inline void free_oblique_split(double** x, int** pair) noexcept nogil:
    if x[0] != NULL:
        free(x[0])
        x[0] = NULL
    if pair[0] != NULL:
        free(pair[0])
        pair[0] = NULL


cdef int _assign_node_ids(TreeNode* node, int next_node_id) noexcept nogil:
    if node == NULL:
        return next_node_id

    node.node_id = next_node_id
    next_node_id += 1

    if not node.is_leaf:
        next_node_id = _assign_node_ids(node.left, next_node_id)
        next_node_id = _assign_node_ids(node.right, next_node_id)

    return next_node_id


cdef void finalize_tree_metadata(TreeNode* node) noexcept nogil:
    _assign_node_ids(node, 0)


cdef inline bint should_use_count_sort(
    const bint is_integer_feature,
    const int n_non_nans,
    const int min_value,
    const int max_value,
    int* out_value_range,
) noexcept nogil:
    cdef long long value_range

    if not is_integer_feature or n_non_nans <= 1:
        return False

    value_range = <long long>max_value - <long long>min_value + 1
    if value_range <= 0 or value_range > INT_MAX:
        return False

    if value_range > <long long>COUNT_SORT_RANGE_FACTOR * n_non_nans:
        return False

    out_value_range[0] = <int>value_range
    return True


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
    const bint is_integer_feature,
    double* out_threshold,
    int* out_left_count,
    bint* out_missing_go_left,
    int** out_categories_go_left,
    int* out_n_categories,
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
    cdef int local_min_value = 0
    cdef int local_max_value = 0
    cdef int local_value_range = 0
    cdef int local_int_value = 0
    cdef double x_val
    cdef double impurity = INFINITY
    cdef bint local_is_integer = is_integer_feature
    cdef bint has_integer_values = False

    out_status[0] = 0
    out_threshold[0] = 0.0
    out_left_count[0] = 0
    out_missing_go_left[0] = True
    out_categories_go_left[0] = NULL
    out_n_categories[0] = 0

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

    for i in range(n_samples):
        idx = sample_indices[i]
        x_val = X[idx, feature_idx]

        if x_val != x_val:
            local_nan_indices[n_nans] = idx
            n_nans += 1
        else:
            local_sort_buffer[n_non_nans].value = x_val
            local_sort_buffer[n_non_nans].index = idx

            if local_is_integer:
                if x_val < INT_MIN or x_val > INT_MAX:
                    local_is_integer = False
                else:
                    local_int_value = <int>x_val
                    if not has_integer_values:
                        local_min_value = local_int_value
                        local_max_value = local_int_value
                        has_integer_values = True
                    else:
                        if local_int_value < local_min_value:
                            local_min_value = local_int_value
                        elif local_int_value > local_max_value:
                            local_max_value = local_int_value

            n_non_nans += 1

    if n_non_nans >= 2 * min_samples_leaf:
        if should_use_count_sort(
            local_is_integer and has_integer_values,
            n_non_nans,
            local_min_value,
            local_max_value,
            &local_value_range,
        ):
            local_count_unique_array = <int*>malloc(local_value_range * sizeof(int))
            local_count_sort_buffer = <SortItem*>malloc(n_non_nans * sizeof(SortItem))
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

            sort_pointer_array_count(
                local_sort_buffer,
                local_count_unique_array,
                local_count_sort_buffer,
                n_non_nans,
                local_value_range,
                local_min_value,
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

        if is_categorical_feature and impurity < INFINITY:
            out_n_categories[0] = <int>out_threshold[0]
            if out_n_categories[0] > 0:
                out_categories_go_left[0] = <int*>malloc(
                    out_n_categories[0] * sizeof(int)
                )
                if out_categories_go_left[0] == NULL:
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

                for i in range(out_n_categories[0]):
                    out_categories_go_left[0][i] = <int>local_categorical_stats[i].value

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
    cdef int* oblique_candidates = NULL
    cdef int** feature_categories = NULL
    cdef int* feature_category_counts = NULL
    cdef int* feature_left_counts = NULL
    cdef int* feature_status = NULL
    cdef bint* feature_missing_left = NULL
    cdef int n_features = X.shape[1]
    cdef int f_idx
    cdef int i
    cdef int left_count_c = 0
    cdef int right_count
    cdef int actual_left_count
    cdef int best_feature = -1
    cdef int best_left_count = 0
    cdef int best_n_categorical = 0
    cdef bint missing_go_left = True
    cdef bint best_missing_go_left = True
    cdef bint use_feature_threads
    cdef Py_ssize_t n_oblique_candidates = 0
    cdef double best_threshold = 0.0
    cdef long long feature_work

    if node == NULL:
        with gil:
            raise MemoryError()

    node.is_leaf = True
    node.value = 0.0
    node.node_id = 0
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
            oblique_candidates = prepare_candidate_pairs(
                task,
                n_classes,
                X.base,
                y.base,
                sample_weight.base,
                sample_indices,
                n_samples,
                n_pair,
                is_categorical,
                &n_oblique_candidates,
            )
        impurity_c = INFINITY
        linear_impurity = INFINITY

        if oblique_candidates != NULL and n_oblique_candidates > 0:
            with gil:
                oblique_x, oblique_pair, linear_x, linear_pair = analyze_both_from_pairs(
                    task,
                    n_classes,
                    X.base,
                    y.base,
                    sample_weight.base,
                    sample_indices,
                    sort_buffer,
                    n_samples,
                    n_pair,
                    oblique_candidates,
                    n_oblique_candidates,
                    rng,
                    gamma,
                    max_iter,
                    relative_change,
                )

            if oblique_x != NULL and oblique_pair != NULL:
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

        free(oblique_candidates)
        oblique_candidates = NULL

        if linear_impurity < impurity_c:
            free_oblique_split(&oblique_x, &oblique_pair)
            best_oblique_x = linear_x
            best_oblique_pair = linear_pair
            linear_x = NULL
            linear_pair = NULL
            min_impurity = linear_impurity
            best_feature = -2
        elif oblique_x != NULL and oblique_pair != NULL:
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
            free_oblique_split(&oblique_x, &oblique_pair)

    feature_thresholds = <double*>malloc(n_features * sizeof(double))
    feature_impurities = <double*>malloc(n_features * sizeof(double))
    feature_categories = <int**>calloc(n_features, sizeof(int*))
    feature_category_counts = <int*>calloc(n_features, sizeof(int))
    feature_left_counts = <int*>malloc(n_features * sizeof(int))
    feature_status = <int*>malloc(n_features * sizeof(int))
    feature_missing_left = <bint*>malloc(n_features * sizeof(bint))

    if (
        feature_thresholds == NULL or
        feature_impurities == NULL or
        feature_categories == NULL or
        feature_category_counts == NULL or
        feature_left_counts == NULL or
        feature_status == NULL or
        feature_missing_left == NULL
    ):
        free(feature_thresholds)
        free(feature_impurities)
        free(feature_categories)
        free(feature_category_counts)
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

    feature_work = <long long>n_features * <long long>n_samples
    use_feature_threads = n_features > 1 and feature_work >= MIN_PARALLEL_FEATURE_WORK

    for f_idx in prange(
        n_features,
        nogil=True,
        schedule="guided",
        use_threads_if=use_feature_threads,
    ):
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
            is_integer[f_idx],
            &feature_thresholds[f_idx],
            &feature_left_counts[f_idx],
            &feature_missing_left[f_idx],
            &feature_categories[f_idx],
            &feature_category_counts[f_idx],
            &feature_status[f_idx],
        )

    for f_idx in range(n_features):
        if feature_status[f_idx] != 0:
            for i in range(n_features):
                if feature_categories[i] != NULL:
                    free(feature_categories[i])
            free(feature_thresholds)
            free(feature_impurities)
            free(feature_categories)
            free(feature_category_counts)
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
            free_oblique_split(&best_oblique_x, &best_oblique_pair)

    if best_feature >= 0 and is_categorical[best_feature]:
        best_n_categorical = feature_category_counts[best_feature]
        node.categories_go_left = feature_categories[best_feature]
        feature_categories[best_feature] = NULL

    for f_idx in range(n_features):
        if feature_categories[f_idx] != NULL:
            free(feature_categories[f_idx])

    free(feature_thresholds)
    free(feature_impurities)
    free(feature_categories)
    free(feature_category_counts)
    free(feature_left_counts)
    free(feature_status)
    free(feature_missing_left)

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
        is_integer,
    )

    free_oblique_split(&best_oblique_x, &best_oblique_pair)
    return node


cdef inline const TreeNode* _find_leaf(
    const TreeNode* node,
    const double[::1, :] X,
    const int sample_idx,
) noexcept nogil:
    cdef const TreeNode* current = node

    while not current.is_leaf:
        if sample_goes_left(
            X,
            sample_idx,
            current.feature_idx,
            current.threshold,
            current.missing_go_left,
            current.n_pair,
            current.x,
            current.pair,
            current.n_category,
            current.categories_go_left,
        ):
            current = current.left
        else:
            current = current.right

    return current


cdef void apply(
    const TreeNode* node,
    const double[::1, :] X,
    int[::1] out,
    const int n_samples,
) noexcept nogil:
    cdef int i
    cdef const TreeNode* leaf
    for i in range(n_samples):
        leaf = _find_leaf(node, X, i)
        out[i] = leaf.node_id


cdef void predict(
    const TreeNode* node,
    const double[::1, :] X,
    double[:, ::1] out,
    const int n_samples,
    const int n_classes,
) noexcept nogil:
    cdef int i
    cdef int j
    cdef const TreeNode* leaf

    for i in range(n_samples):
        leaf = _find_leaf(node, X, i)
        if n_classes <= 2:
            out[i, 0] = leaf.value
        else:
            for j in range(n_classes):
                out[i, j] = leaf.value_multiclass[j]
