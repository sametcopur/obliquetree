from cython.parallel cimport prange

from libc.limits cimport INT_MAX, INT_MIN
from libc.math cimport INFINITY, exp, log, fabs
from libc.stdlib cimport calloc, malloc, free
from libc.string cimport memset

from .metric cimport (
    calculate_impurity,
    calculate_node_gini,
    calculate_node_mse,
    calculate_node_value,
    calculate_node_value_multiclass,
)
from .oblique cimport analyze_both_from_pairs, fill_oblique_sort_buffer, prepare_candidate_pairs
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
    const int top_k,
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
    node.leaf_coef = NULL
    node.leaf_intercept_buf = NULL
    node.leaf_n_coef = 0
    node.leaf_n_models = 0
    node.impurity = 0.0

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
                top_k,
                is_categorical,
                &n_oblique_candidates,
            )
        impurity_c = INFINITY
        linear_impurity = INFINITY

        if oblique_candidates != NULL and n_oblique_candidates > 0:
            with gil:
                analyze_both_from_pairs(
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
                    &oblique_x,
                    &oblique_pair,
                    &linear_x,
                    &linear_pair,
                )

            if oblique_x != NULL and oblique_pair != NULL:
                with gil:
                    fill_oblique_sort_buffer(
                        X.base,
                        sample_indices,
                        sort_buffer,
                        n_samples,
                        n_pair,
                        oblique_pair,
                        oblique_x,
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

            if linear_x != NULL and linear_pair != NULL:
                with gil:
                    fill_oblique_sort_buffer(
                        X.base,
                        sample_indices,
                        sort_buffer,
                        n_samples,
                        n_pair,
                        linear_pair,
                        linear_x,
                    )
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
        top_k,
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
        top_k,
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
    cdef bint use_parallel = n_samples >= 1024
    for i in prange(n_samples, nogil=True, schedule="static", use_threads_if=use_parallel):
        leaf = _find_leaf(node, X, i)
        out[i] = leaf.node_id


cdef void predict(
    const TreeNode* node,
    const double[::1, :] X,
    double[:, ::1] out,
    const int n_samples,
    const int n_classes,
    const bint linear_leaf,
    const int* numeric_features,
    const int n_numeric_features,
    const bint clip_to_unit,
) noexcept nogil:
    cdef int i, j, k
    cdef const TreeNode* leaf
    cdef double val
    cdef double x_val
    cdef bint bad_seen
    cdef double max_logit, sum_exp
    cdef int d_coef
    cdef bint use_parallel = n_samples >= 1024
    cdef bint use_linear = linear_leaf

    for i in prange(n_samples, nogil=True, schedule="static", use_threads_if=use_parallel):
        leaf = _find_leaf(node, X, i)

        if use_linear and leaf.leaf_coef != NULL and leaf.leaf_n_models > 0:
            d_coef = leaf.leaf_n_coef
            bad_seen = False
            for j in range(n_numeric_features):
                x_val = X[i, numeric_features[j]]
                if x_val != x_val or x_val >= INFINITY or x_val <= -INFINITY:
                    bad_seen = True
                    break

            if bad_seen:
                if n_classes <= 2:
                    out[i, 0] = leaf.value
                else:
                    for j in range(n_classes):
                        out[i, j] = leaf.value_multiclass[j]
            else:
                if n_classes == 1:
                    val = leaf.leaf_intercept_buf[0]
                    for j in range(d_coef):
                        val = val + leaf.leaf_coef[j] * X[i, numeric_features[j]]
                    out[i, 0] = val
                elif n_classes == 2:
                    val = leaf.leaf_intercept_buf[0]
                    for j in range(d_coef):
                        val = val + leaf.leaf_coef[j] * X[i, numeric_features[j]]
                    val = _sigmoid(val)
                    if clip_to_unit:
                        if val < 0.0:
                            val = 0.0
                        elif val > 1.0:
                            val = 1.0
                    out[i, 0] = val
                else:
                    max_logit = -INFINITY
                    for k in range(n_classes):
                        val = leaf.leaf_intercept_buf[k]
                        for j in range(d_coef):
                            val = val + leaf.leaf_coef[k * d_coef + j] * X[i, numeric_features[j]]
                        out[i, k] = val
                        if val > max_logit:
                            max_logit = val
                    sum_exp = 0.0
                    for k in range(n_classes):
                        val = exp(out[i, k] - max_logit)
                        out[i, k] = val
                        sum_exp = sum_exp + val
                    if sum_exp > 0.0:
                        for k in range(n_classes):
                            out[i, k] = out[i, k] / sum_exp
        else:
            if n_classes <= 2:
                out[i, 0] = leaf.value
            else:
                for j in range(n_classes):
                    out[i, j] = leaf.value_multiclass[j]


cdef bint _solve_cholesky(double* G, double* b, double* coef, const int d) noexcept nogil:
    """In-place Cholesky decomposition of SPD matrix G (d x d, row-major) and
    solve G @ coef = b. Lower-triangular factor overwrites G. Returns True on
    success, False if a non-positive pivot is encountered."""
    cdef int i, j, k
    cdef double sum_val
    cdef double pivot
    cdef double y_val

    for i in range(d):
        for j in range(i + 1):
            sum_val = G[i * d + j]
            for k in range(j):
                sum_val = sum_val - G[i * d + k] * G[j * d + k]
            if i == j:
                if not (sum_val > 0.0):
                    return False
                pivot = sum_val ** 0.5
                G[i * d + i] = pivot
            else:
                G[i * d + j] = sum_val / G[j * d + j]

    for i in range(d):
        sum_val = b[i]
        for k in range(i):
            sum_val = sum_val - G[i * d + k] * coef[k]
        coef[i] = sum_val / G[i * d + i]

    i = d - 1
    while i >= 0:
        sum_val = coef[i]
        for k in range(i + 1, d):
            sum_val = sum_val - G[k * d + i] * coef[k]
        coef[i] = sum_val / G[i * d + i]
        i = i - 1

    return True


cdef inline double _sigmoid(double x) noexcept nogil:
    cdef double ex
    if x >= 0.0:
        ex = exp(-x)
        return 1.0 / (1.0 + ex)
    ex = exp(x)
    return ex / (1.0 + ex)


cdef bint _centered_weighted_ols(
    const double[::1, :] X,
    const int* sample_indices,
    const int n_leaf_samples,
    const int* numeric_features,
    const int d,
    const double* w_per_sample,
    const double* z_per_sample,
    const double leaf_l2,
    double* x_mean,
    double* G,
    double* rhs,
    double* coef_out,
    double* intercept_out,
) noexcept nogil:
    cdef int i, j, k, idx
    cdef double w_sum = 0.0
    cdef double w, z_mean = 0.0, dxj, dxk, zc

    if d <= 0:
        return False

    memset(x_mean, 0, d * sizeof(double))
    memset(G, 0, d * d * sizeof(double))
    memset(rhs, 0, d * sizeof(double))

    for i in range(n_leaf_samples):
        idx = sample_indices[i]
        w = w_per_sample[i]
        w_sum = w_sum + w
        z_mean = z_mean + w * z_per_sample[i]
        for j in range(d):
            x_mean[j] = x_mean[j] + w * X[idx, numeric_features[j]]

    if w_sum <= 0.0:
        return False

    z_mean = z_mean / w_sum
    for j in range(d):
        x_mean[j] = x_mean[j] / w_sum

    for i in range(n_leaf_samples):
        idx = sample_indices[i]
        w = w_per_sample[i]
        zc = z_per_sample[i] - z_mean
        for j in range(d):
            dxj = X[idx, numeric_features[j]] - x_mean[j]
            rhs[j] = rhs[j] + w * dxj * zc
            for k in range(j + 1):
                dxk = X[idx, numeric_features[k]] - x_mean[k]
                G[j * d + k] = G[j * d + k] + w * dxj * dxk

    for j in range(d):
        for k in range(j + 1, d):
            G[j * d + k] = G[k * d + j]

    if leaf_l2 > 0.0:
        for j in range(d):
            G[j * d + j] = G[j * d + j] + leaf_l2

    if not _solve_cholesky(G, rhs, coef_out, d):
        return False

    intercept_out[0] = z_mean
    for j in range(d):
        intercept_out[0] = intercept_out[0] - coef_out[j] * x_mean[j]

    return True


cdef bint _fit_multinomial(
    const double[::1, :] X,
    const int* sample_indices,
    const int n_leaf_samples,
    const int* numeric_features,
    const int d,
    const double[::1] y,
    const double[::1] sample_weight,
    const int K,
    const double leaf_l2,
    double* coef_out,
    double* intercept_out,
) noexcept nogil:
    # K-1 reference-class parametrization: free parameters for classes 0..K-2,
    # last class fixed at logit 0. Eliminates the rank-1 redundancy of the
    # symmetric K-class form so the Hessian is non-singular at zero ridge.
    cdef int dim = d + 1
    cdef int K_free = K - 1
    cdef int total = K_free * dim
    cdef int max_iter = 50
    cdef double tol = 1e-6

    if K < 2:
        return False
    if total <= 0:
        return False

    cdef double* H = <double*>malloc(total * total * sizeof(double))
    cdef double* grad = <double*>malloc(total * sizeof(double))
    cdef double* delta = <double*>malloc(total * sizeof(double))
    cdef double* beta = <double*>malloc(total * sizeof(double))
    cdef double* p_buf = <double*>malloc(n_leaf_samples * K * sizeof(double))
    cdef double* x_aug = <double*>malloc(dim * sizeof(double))

    cdef bint alloc_ok = (H != NULL and grad != NULL and delta != NULL
                          and beta != NULL and p_buf != NULL and x_aug != NULL)
    if not alloc_ok:
        if H != NULL: free(H)
        if grad != NULL: free(grad)
        if delta != NULL: free(delta)
        if beta != NULL: free(beta)
        if p_buf != NULL: free(p_buf)
        if x_aug != NULL: free(x_aug)
        return False

    cdef int i, j, idx, k, l, a, b, it
    cdef double w, eta, max_eta, sum_exp, pk, pl, yk, coeff
    cdef double max_step
    cdef double damping
    cdef double eff_ridge
    cdef int label_int
    cdef bint cholesky_ok = True
    cdef bint converged = False
    cdef double bv

    eff_ridge = leaf_l2
    if eff_ridge < 1e-10:
        eff_ridge = 1e-10

    memset(beta, 0, total * sizeof(double))

    for it in range(max_iter):
        memset(grad, 0, total * sizeof(double))
        memset(H, 0, total * total * sizeof(double))

        for i in range(n_leaf_samples):
            idx = sample_indices[i]
            w = sample_weight[idx]
            x_aug[0] = 1.0
            for j in range(d):
                x_aug[j + 1] = X[idx, numeric_features[j]]

            # Logits: free for k in 0..K_free-1, fixed 0 for class K-1.
            max_eta = 0.0
            for k in range(K_free):
                eta = 0.0
                for j in range(dim):
                    eta = eta + beta[k * dim + j] * x_aug[j]
                p_buf[i * K + k] = eta
                if eta > max_eta:
                    max_eta = eta
            p_buf[i * K + K_free] = 0.0

            sum_exp = 0.0
            for k in range(K):
                eta = exp(p_buf[i * K + k] - max_eta)
                p_buf[i * K + k] = eta
                sum_exp = sum_exp + eta
            if sum_exp > 0.0:
                for k in range(K):
                    p_buf[i * K + k] = p_buf[i * K + k] / sum_exp

            label_int = <int>y[idx]

            for k in range(K_free):
                pk = p_buf[i * K + k]
                yk = 1.0 if k == label_int else 0.0
                coeff = w * (pk - yk)
                for a in range(dim):
                    grad[k * dim + a] = grad[k * dim + a] + coeff * x_aug[a]

            for k in range(K_free):
                pk = p_buf[i * K + k]
                for l in range(K_free):
                    pl = p_buf[i * K + l]
                    if k == l:
                        coeff = w * pk * (1.0 - pl)
                    else:
                        coeff = -w * pk * pl
                    if coeff == 0.0:
                        continue
                    for a in range(dim):
                        for b in range(dim):
                            H[(k * dim + a) * total + (l * dim + b)] = (
                                H[(k * dim + a) * total + (l * dim + b)]
                                + coeff * x_aug[a] * x_aug[b]
                            )

        # Apply ridge (with internal floor for rank-deficient X).
        for k in range(K_free):
            for a in range(1, dim):
                H[(k * dim + a) * total + (k * dim + a)] = (
                    H[(k * dim + a) * total + (k * dim + a)] + eff_ridge
                )

        if not _solve_cholesky(H, grad, delta, total):
            cholesky_ok = False
            break

        max_step = 0.0
        for j in range(total):
            if fabs(delta[j]) > max_step:
                max_step = fabs(delta[j])

        damping = 1.0
        if max_step > 5.0:
            damping = 5.0 / max_step

        max_step = 0.0
        for j in range(total):
            beta[j] = beta[j] - damping * delta[j]
            if fabs(damping * delta[j]) > max_step:
                max_step = fabs(damping * delta[j])

        if max_step < tol:
            converged = True
            break

    cdef bint finite_ok = True
    if cholesky_ok:
        for j in range(total):
            bv = beta[j]
            if bv != bv or bv >= INFINITY or bv <= -INFINITY:
                finite_ok = False
                break

    cdef bint result = cholesky_ok and finite_ok and (converged or it > 0)

    if result:
        for k in range(K_free):
            intercept_out[k] = beta[k * dim]
            for j in range(d):
                coef_out[k * d + j] = beta[k * dim + j + 1]
        # Reference class K-1 fixed at zero logit.
        intercept_out[K - 1] = 0.0
        for j in range(d):
            coef_out[(K - 1) * d + j] = 0.0

    free(H)
    free(grad)
    free(delta)
    free(beta)
    free(p_buf)
    free(x_aug)
    return result


cdef bint _fit_logistic_one(
    const double[::1, :] X,
    const int* sample_indices,
    const int n_leaf_samples,
    const int* numeric_features,
    const int d,
    const double[::1] y_target,
    const double[::1] sample_weight,
    const int target_class,
    const double leaf_l2,
    double* coef_out,
    double* intercept_out,
    double* x_mean,
    double* G,
    double* rhs,
    double* w_buf,
    double* z_buf,
    double* coef_new,
) noexcept nogil:
    cdef int i, j, idx, it
    cdef double y_mean = 0.0
    cdef double inner_ridge
    cdef double cv
    cdef double w_sum = 0.0
    cdef double w, target_i, eta, p, dp, intercept, intercept_new
    cdef double diff, max_diff
    cdef double eps = 1e-9
    cdef int max_iter = 25
    cdef double tol = 1e-6

    if n_leaf_samples < d + 1:
        return False

    for i in range(n_leaf_samples):
        idx = sample_indices[i]
        w = sample_weight[idx]
        if target_class < 0:
            target_i = y_target[idx]
        else:
            target_i = 1.0 if (<int>y_target[idx]) == target_class else 0.0
        w_sum = w_sum + w
        y_mean = y_mean + w * target_i

    if w_sum <= 0.0:
        return False
    y_mean = y_mean / w_sum

    if y_mean < eps:
        y_mean = eps
    elif y_mean > 1.0 - eps:
        y_mean = 1.0 - eps

    intercept = log(y_mean / (1.0 - y_mean))
    for j in range(d):
        coef_out[j] = 0.0

    # Classification IRLS: enforce a tiny ridge floor so the inner OLS
    # Gram is non-singular even when numeric features are correlated /
    # rank-deficient. User-supplied ridge is used unchanged when above
    # the floor.
    inner_ridge = leaf_l2
    if inner_ridge < 1e-10:
        inner_ridge = 1e-10

    for it in range(max_iter):
        for i in range(n_leaf_samples):
            idx = sample_indices[i]
            if target_class < 0:
                target_i = y_target[idx]
            else:
                target_i = 1.0 if (<int>y_target[idx]) == target_class else 0.0

            eta = intercept
            for j in range(d):
                eta = eta + coef_out[j] * X[idx, numeric_features[j]]

            p = _sigmoid(eta)
            if p < eps:
                p = eps
            elif p > 1.0 - eps:
                p = 1.0 - eps
            dp = p * (1.0 - p)

            w_buf[i] = sample_weight[idx] * dp
            z_buf[i] = eta + (target_i - p) / dp

        if not _centered_weighted_ols(
            X, sample_indices, n_leaf_samples, numeric_features, d,
            w_buf, z_buf, inner_ridge,
            x_mean, G, rhs, coef_new, &intercept_new,
        ):
            return False

        max_diff = fabs(intercept_new - intercept)
        for j in range(d):
            diff = fabs(coef_new[j] - coef_out[j])
            if diff > max_diff:
                max_diff = diff
            coef_out[j] = coef_new[j]
        intercept = intercept_new

        if max_diff < tol:
            break

    # Finite check on the last iterate (Newton can blow up on
    # near-separable data without sufficient ridge).
    if intercept != intercept or intercept >= INFINITY or intercept <= -INFINITY:
        return False
    for j in range(d):
        cv = coef_out[j]
        if cv != cv or cv >= INFINITY or cv <= -INFINITY:
            return False

    intercept_out[0] = intercept
    return True


cdef void _fit_one_leaf(
    TreeNode* leaf,
    const double[::1, :] X,
    const double[::1] y,
    const double[::1] sample_weight,
    const int* numeric_features,
    const int d,
    const int n_classes,
    const bint task,
    const double leaf_l2,
    const int* sample_indices,
    const int n_leaf_samples,
    double* x_mean,
    double* G,
    double* rhs,
    double* coef_buf,
    double* w_buf,
    double* z_buf,
    double* coef_new_buf,
) noexcept nogil:
    cdef int j, k
    cdef int n_models
    cdef double intercept_val
    cdef double w
    cdef int idx
    cdef int i

    if leaf.leaf_coef != NULL:
        free(leaf.leaf_coef)
        leaf.leaf_coef = NULL
    if leaf.leaf_intercept_buf != NULL:
        free(leaf.leaf_intercept_buf)
        leaf.leaf_intercept_buf = NULL
    leaf.leaf_n_coef = 0
    leaf.leaf_n_models = 0

    if d <= 0:
        return
    if n_leaf_samples < d + 1:
        return
    # Pure / constant leaf: y is identical (gini=0 for classification,
    # MSE=0 for regression) -> linear fit collapses to a constant anyway,
    # so skip the Newton/IRLS work and let predict use leaf.value /
    # leaf.value_multiclass.
    if leaf.impurity == 0.0:
        return

    if task == 1:
        n_models = 1
    elif n_classes == 2:
        n_models = 1
    else:
        n_models = n_classes

    leaf.leaf_coef = <double*>malloc(n_models * d * sizeof(double))
    leaf.leaf_intercept_buf = <double*>malloc(n_models * sizeof(double))
    if leaf.leaf_coef == NULL or leaf.leaf_intercept_buf == NULL:
        if leaf.leaf_coef != NULL:
            free(leaf.leaf_coef)
            leaf.leaf_coef = NULL
        if leaf.leaf_intercept_buf != NULL:
            free(leaf.leaf_intercept_buf)
            leaf.leaf_intercept_buf = NULL
        return

    if task == 1:
        # Regression: weighted OLS with z=y, w=sample_weight
        for i in range(n_leaf_samples):
            idx = sample_indices[i]
            w_buf[i] = sample_weight[idx]
            z_buf[i] = y[idx]

        if not _centered_weighted_ols(
            X, sample_indices, n_leaf_samples, numeric_features, d,
            w_buf, z_buf, leaf_l2,
            x_mean, G, rhs, coef_buf, &intercept_val,
        ):
            free(leaf.leaf_coef)
            leaf.leaf_coef = NULL
            free(leaf.leaf_intercept_buf)
            leaf.leaf_intercept_buf = NULL
            return

        for j in range(d):
            leaf.leaf_coef[j] = coef_buf[j]
        leaf.leaf_intercept_buf[0] = intercept_val

    elif n_classes == 2:
        # Binary classification: logistic IRLS
        if not _fit_logistic_one(
            X, sample_indices, n_leaf_samples, numeric_features, d,
            y, sample_weight, -1, leaf_l2,
            coef_buf, &intercept_val,
            x_mean, G, rhs, w_buf, z_buf, coef_new_buf,
        ):
            free(leaf.leaf_coef)
            leaf.leaf_coef = NULL
            free(leaf.leaf_intercept_buf)
            leaf.leaf_intercept_buf = NULL
            return

        for j in range(d):
            leaf.leaf_coef[j] = coef_buf[j]
        leaf.leaf_intercept_buf[0] = intercept_val

    else:
        # Multiclass: proper softmax regression via Newton-Raphson
        if not _fit_multinomial(
            X, sample_indices, n_leaf_samples, numeric_features, d,
            y, sample_weight, n_classes, leaf_l2,
            leaf.leaf_coef, leaf.leaf_intercept_buf,
        ):
            free(leaf.leaf_coef)
            leaf.leaf_coef = NULL
            free(leaf.leaf_intercept_buf)
            leaf.leaf_intercept_buf = NULL
            return

    leaf.leaf_n_coef = d
    leaf.leaf_n_models = n_models


cdef void _walk_and_fit(
    TreeNode* node,
    const double[::1, :] X,
    const double[::1] y,
    const double[::1] sample_weight,
    const int* numeric_features,
    const int d,
    const int n_classes,
    const bint task,
    const double leaf_l2,
    int* sample_indices,
    const int n_node_samples,
    double* x_mean,
    double* G,
    double* rhs,
    double* coef_buf,
    double* w_buf,
    double* z_buf,
    double* coef_new_buf,
) noexcept nogil:
    cdef int left_count

    if node.is_leaf:
        _fit_one_leaf(
            node,
            X,
            y,
            sample_weight,
            numeric_features,
            d,
            n_classes,
            task,
            leaf_l2,
            sample_indices,
            n_node_samples,
            x_mean,
            G,
            rhs,
            coef_buf,
            w_buf,
            z_buf,
            coef_new_buf,
        )
        return

    left_count = partition_samples_inplace(
        X,
        sample_indices,
        n_node_samples,
        node.feature_idx,
        node.threshold,
        node.missing_go_left,
        node.n_pair,
        node.x,
        node.pair,
        node.n_category,
        node.categories_go_left,
    )

    _walk_and_fit(
        node.left, X, y, sample_weight,
        numeric_features, d, n_classes, task, leaf_l2,
        sample_indices, left_count,
        x_mean, G, rhs, coef_buf, w_buf, z_buf, coef_new_buf,
    )
    _walk_and_fit(
        node.right, X, y, sample_weight,
        numeric_features, d, n_classes, task, leaf_l2,
        sample_indices + left_count, n_node_samples - left_count,
        x_mean, G, rhs, coef_buf, w_buf, z_buf, coef_new_buf,
    )


cdef void fit_linear_leaves(
    TreeNode* root,
    const double[::1, :] X,
    const double[::1] y,
    const double[::1] sample_weight,
    const int* numeric_features,
    const int n_numeric_features,
    const double leaf_l2,
    const int n_samples,
    const int n_classes,
    const bint task,
) noexcept nogil:
    cdef int* sample_indices = NULL
    cdef double* x_mean = NULL
    cdef double* G = NULL
    cdef double* rhs = NULL
    cdef double* coef_buf = NULL
    cdef double* w_buf = NULL
    cdef double* z_buf = NULL
    cdef double* coef_new_buf = NULL
    cdef int i
    cdef int d = n_numeric_features

    if root == NULL or d <= 0 or n_samples <= 0:
        return

    sample_indices = <int*>malloc(n_samples * sizeof(int))
    x_mean = <double*>malloc(d * sizeof(double))
    G = <double*>malloc(d * d * sizeof(double))
    rhs = <double*>malloc(d * sizeof(double))
    coef_buf = <double*>malloc(d * sizeof(double))
    coef_new_buf = <double*>malloc(d * sizeof(double))
    w_buf = <double*>malloc(n_samples * sizeof(double))
    z_buf = <double*>malloc(n_samples * sizeof(double))

    if (sample_indices == NULL or x_mean == NULL or G == NULL or rhs == NULL
        or coef_buf == NULL or coef_new_buf == NULL
        or w_buf == NULL or z_buf == NULL):
        if sample_indices != NULL: free(sample_indices)
        if x_mean != NULL: free(x_mean)
        if G != NULL: free(G)
        if rhs != NULL: free(rhs)
        if coef_buf != NULL: free(coef_buf)
        if coef_new_buf != NULL: free(coef_new_buf)
        if w_buf != NULL: free(w_buf)
        if z_buf != NULL: free(z_buf)
        return

    for i in range(n_samples):
        sample_indices[i] = i

    _walk_and_fit(
        root, X, y, sample_weight,
        numeric_features, d, n_classes, task, leaf_l2,
        sample_indices, n_samples,
        x_mean, G, rhs, coef_buf, w_buf, z_buf, coef_new_buf,
    )

    free(sample_indices)
    free(x_mean)
    free(G)
    free(rhs)
    free(coef_buf)
    free(coef_new_buf)
    free(w_buf)
    free(z_buf)
