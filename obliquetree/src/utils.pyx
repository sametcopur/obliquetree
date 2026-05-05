from libcpp.algorithm cimport sort as csort
from libc.string cimport memcpy, memset
from libc.math cimport floor
from libc.stdlib cimport free, malloc

cdef dict _recurse(const TreeNode* node):
    base = {
        "n_samples": node.n_samples,
        "impurity": node.impurity,
        "is_leaf": node.is_leaf
    }

    # Leaf node handling
    if node.is_leaf:
        if node.value_multiclass:
            base.update({
                "values": [node.value_multiclass[i] for i in range(node.n_classes)]
            })
        else:
            base.update({
                "value": node.value
            })
        if (node.leaf_coef != NULL and node.leaf_intercept_buf != NULL
                and node.leaf_n_coef > 0 and node.leaf_n_models > 0):
            base.update({
                "leaf_coef": [
                    node.leaf_coef[i]
                    for i in range(node.leaf_n_models * node.leaf_n_coef)
                ],
                "leaf_intercept": [
                    node.leaf_intercept_buf[i] for i in range(node.leaf_n_models)
                ],
                "leaf_n_models": node.leaf_n_models,
                "leaf_n_coef": node.leaf_n_coef,
            })
        return base

    # Add common non-leaf attributes (except left/right)
    base.update({
        "is_oblique": node.feature_idx == -2,
    })

    # Oblique split handling
    if node.feature_idx == -2:
        base.update({
            "weights": [node.x[i] for i in range(node.n_pair)],
            "features": [node.pair[i] for i in range(node.n_pair)],
            "threshold": node.threshold,
        })

        if node.value_multiclass:
            base.update({
                "values": [node.value_multiclass[i] for i in range(node.n_classes)]
            })
        else:
            base.update({
                "value": node.value
            })

    else:
        # Non-oblique split handling
        base.update({
            "feature_idx": node.feature_idx,
            "missing_go_left": bool(node.missing_go_left)
        })

        # Add value or values based on multiclass
        if node.value_multiclass:
            base.update({
                "values": [node.value_multiclass[i] for i in range(node.n_classes)]
            })
        else:
            base.update({
                "value": node.value
            })

        # Category handling
        if node.n_category == 0:
            base.update({
                "threshold": node.threshold
            })
        else:
            base.update({
                "category_left": sorted(node.categories_go_left[i] for i in range(node.n_category))
            })

    # Add left and right at the end
    base.update({
        "left": _recurse(node.left),
        "right": _recurse(node.right)
    })

    return base

# Asıl Export Fonksiyonu
cpdef dict export_tree(TreeClassifier tree):
    """
    Export both the tree structure and classifier parameters as a dictionary.
    
    Parameters:
    tree (TreeClassifier): The trained decision tree classifier.
    
    Returns:
    dict: Dictionary containing tree parameters and structure with format:
        {
            'params': dict of TreeClassifier parameters,
            'tree': nested dictionary representation of the tree structure
        }
    """
    if tree.root == NULL:
        raise ValueError("Tree is not fitted yet!")
    
    cdef int i
    numeric_features = None
    if tree.numeric_features_ != NULL and tree.n_numeric_features_ > 0:
        numeric_features = [tree.numeric_features_[i] for i in range(tree.n_numeric_features_)]

    # Export parameters
    params = {
        'max_depth': tree.max_depth,
        'min_samples_leaf': tree.min_samples_leaf,
        'min_samples_split': tree.min_samples_split,
        'min_impurity_decrease': tree.min_impurity_decrease,
        'random_state': tree.random_state,
        'n_pair': tree.n_pair,
        'top_k': tree.top_k,
        'gamma': tree.gamma,
        'max_iter': tree.max_iter,
        'relative_change': tree.relative_change,
        'use_oblique': tree.use_oblique,
        'categories': tree.categories,
        'ccp_alpha': tree.ccp_alpha,
        'task': tree.task,
        'n_classes': tree.n_classes,
        'n_features': tree.n_features,
        'cat_': tree.cat_,
        'linear_leaf': tree.linear_leaf,
        'leaf_l2': tree.leaf_l2,
        'numeric_features': numeric_features,
    }
    
    # Return combined dictionary
    return {
        'params': params,
        'tree': _recurse(tree.root) if tree.root != NULL else None,
    }

cdef TreeNode* deserialize_tree(dict tree_dict, int n_features, int n_classes) except NULL:
    """
    Deserialize a dictionary representation into a TreeNode structure.
    
    Parameters:
        tree_dict (dict): Dictionary containing the tree node data
        n_features (int): Number of features in the dataset
        n_classes (int): Number of classes for classification
        
    Returns:
        TreeNode*: Pointer to the deserialized tree node
        
    Raises:
        MemoryError: If memory allocation fails
        ValueError: If the input dictionary is invalid
    """
    if tree_dict is None:
        return NULL
    
    # Allocate memory for the node
    cdef TreeNode* node = <TreeNode*>malloc(sizeof(TreeNode))
    if not node:
        raise MemoryError("Failed to allocate memory for TreeNode")
    
    # Initialize all pointers to NULL
    node.value_multiclass = NULL
    node.x = NULL
    node.pair = NULL
    node.categories_go_left = NULL
    node.left = NULL
    node.right = NULL
    node.node_id = 0
    node.feature_idx = -1
    node.threshold = 0.0
    node.n_pair = 0
    node.n_category = 0
    node.leaf_coef = NULL
    node.leaf_intercept_buf = NULL
    node.leaf_n_coef = 0
    node.leaf_n_models = 0

    # Set basic node properties
    node.is_leaf = tree_dict['is_leaf']
    node.value = tree_dict.get('value', 0.0)
    node.impurity = tree_dict.get('impurity', 0.0)
    node.n_samples = tree_dict.get('n_samples', 0)
    node.missing_go_left = tree_dict.get('missing_go_left', 1)
    node.n_classes = n_classes

    # Handle multiclass values if present
    if 'values' in tree_dict:
        node.value_multiclass = <double*>malloc(n_classes * sizeof(double))
        if not node.value_multiclass:
            free(node)
            raise MemoryError("Failed to allocate memory for value_multiclass")
        for i in range(n_classes):
            node.value_multiclass[i] = tree_dict['values'][i]

    # Restore leaf linear model if present
    if 'leaf_coef' in tree_dict:
        leaf_coef_list = tree_dict['leaf_coef']
        leaf_intercept_obj = tree_dict.get('leaf_intercept', 0.0)
        if isinstance(leaf_intercept_obj, (list, tuple)):
            leaf_intercept_list = list(leaf_intercept_obj)
        else:
            leaf_intercept_list = [float(leaf_intercept_obj)]
        node.leaf_n_models = tree_dict.get('leaf_n_models', len(leaf_intercept_list))
        if node.leaf_n_models > 0:
            node.leaf_n_coef = tree_dict.get(
                'leaf_n_coef', len(leaf_coef_list) // node.leaf_n_models
            )
        else:
            node.leaf_n_coef = 0
        total_coef = node.leaf_n_models * node.leaf_n_coef
        if total_coef > 0 and len(leaf_coef_list) >= total_coef:
            node.leaf_coef = <double*>malloc(total_coef * sizeof(double))
            node.leaf_intercept_buf = <double*>malloc(node.leaf_n_models * sizeof(double))
            if not node.leaf_coef or not node.leaf_intercept_buf:
                if node.leaf_coef != NULL:
                    free(node.leaf_coef)
                if node.leaf_intercept_buf != NULL:
                    free(node.leaf_intercept_buf)
                if node.value_multiclass != NULL:
                    free(node.value_multiclass)
                free(node)
                raise MemoryError("Failed to allocate memory for leaf_coef")
            for i in range(total_coef):
                node.leaf_coef[i] = leaf_coef_list[i]
            for i in range(node.leaf_n_models):
                node.leaf_intercept_buf[i] = leaf_intercept_list[i]
        else:
            node.leaf_n_models = 0
            node.leaf_n_coef = 0
    
    # If not a leaf node, handle split information
    if not node.is_leaf:
        # Set split type and related properties
        node.feature_idx = tree_dict.get('feature_idx', -2)
        node.threshold = tree_dict.get('threshold', 0.0)
        
        # Handle oblique splits
        if tree_dict.get('is_oblique', False):
            node.n_pair = len(tree_dict['weights'])
            
            # Allocate and set weights
            node.x = <double*>malloc(node.n_pair * sizeof(double))
            if not node.x:
                raise MemoryError("Failed to allocate memory for weights")
            
            for i in range(node.n_pair):
                node.x[i] = tree_dict['weights'][i]
            
            # Allocate and set feature indices
            node.pair = <int*>malloc(node.n_pair * sizeof(int))
            if not node.pair:
                raise MemoryError("Failed to allocate memory for feature indices")
            
            for i in range(node.n_pair):
                node.pair[i] = tree_dict['features'][i]
        
        # Handle categorical splits
        elif 'category_left' in tree_dict:
            categories = tree_dict['category_left']
            node.n_category = len(categories)
            
            node.categories_go_left = <int*>malloc(node.n_category * sizeof(int))
            if not node.categories_go_left:
                raise MemoryError("Failed to allocate memory for categories")
            
            for i in range(node.n_category):
                node.categories_go_left[i] = categories[i]
        
        # Recursively deserialize child nodes
        node.left = deserialize_tree(tree_dict['left'], n_features, n_classes)
        node.right = deserialize_tree(tree_dict['right'], n_features, n_classes)
        
        if node.left == NULL or node.right == NULL:
            return NULL
    
    return node

cdef inline bint compare_items(const SortItem& a, const SortItem& b) noexcept nogil:
    return a.value < b.value

cdef inline void sort_pointer_array(SortItem* items, const int n_samples) noexcept nogil:
    # Wrap the raw pointer array in std::sort
    csort(items, items + n_samples, compare_items)

cdef inline bint compare_category_stats(const CategoryStat& a, const CategoryStat& b) noexcept nogil:
    # Compare mean values of CategoryStat (ascending order)
    return (a.y_sum / a.count) < (b.y_sum / b.count)

cdef inline void sort_category_stats(CategoryStat* stats, const int unique_count) noexcept nogil:
    # Sort the CategoryStat array using csort and the custom comparator
    csort(stats, stats + unique_count, compare_category_stats)


cdef inline bint compare_category_stats_multiclass(
    const CategoryStat& a, 
    const CategoryStat& b, 
    const int n_classes,
    const int* permutation_order) noexcept nogil:
    """
    Compare categories based on their class distributions after applying permutation
    """
    cdef int i
    cdef double a_prop, b_prop
    
    # Compare each class proportion in order of permutation
    for i in range(n_classes):
        if a.weight > 0 and b.weight > 0:
            a_prop = a.class_weights[permutation_order[i]] / a.weight
            b_prop = b.class_weights[permutation_order[i]] / b.weight
            
            if a_prop != b_prop:
                return a_prop > b_prop
    
    return False

# Quicksort implementasyonu
cdef inline int partition_category_stats(
    CategoryStat* stats,
    const int* permutation_order,
    int low,
    int high,
    const int n_classes) noexcept nogil:
    """
    Quicksort için partition fonksiyonu
    """
    cdef CategoryStat pivot = stats[high]
    cdef int i = low - 1
    cdef int j
    cdef CategoryStat temp
    
    for j in range(low, high):
        if not compare_category_stats_multiclass(stats[j], pivot, n_classes, permutation_order):
            i += 1
            temp = stats[i]
            stats[i] = stats[j]
            stats[j] = temp
            
    temp = stats[i + 1]
    stats[i + 1] = stats[high]
    stats[high] = temp
    return i + 1

cdef inline void quicksort_category_stats(
    CategoryStat* stats,
    const int* permutation_order,
    int low,
    int high,
    const int n_classes) noexcept nogil:
    """
    Quicksort implementasyonu
    """
    cdef int pi
    
    if low < high:
        pi = partition_category_stats(stats, permutation_order, low, high, n_classes)
        quicksort_category_stats(stats, permutation_order, low, pi - 1, n_classes)
        quicksort_category_stats(stats, permutation_order, pi + 1, high, n_classes)

cdef inline void sort_category_stats_multiclass(
    CategoryStat* stats,
    const int* permutation_order,
    const int unique_count,
    const int n_classes) noexcept nogil:
    """
    CategoryStat array'ini quicksort kullanarak sıralar
    """
    quicksort_category_stats(stats, permutation_order, 0, unique_count - 1, n_classes)

cdef void populate_indices(
    const bint missing_go_left,
    const int n_samples,
    const int left_count,
    const int right_count,
    const int n_nans,
    int* left_indices,
    int* right_indices,
    const int* best_sorted_indices,
    const int* best_nan_indices) noexcept nogil:

    cdef size_t chunk_size

    if missing_go_left:
        # 1) left_indices[0 .. (left_count - n_nans)] = best_sorted_indices[0 .. (left_count - n_nans)]
        chunk_size = (left_count - n_nans) * sizeof(int)
        memcpy(left_indices, best_sorted_indices, chunk_size)

        # 2) left_indices[(left_count - n_nans) .. left_count] = best_nan_indices[0 .. n_nans]
        chunk_size = n_nans * sizeof(int)
        memcpy(left_indices + (left_count - n_nans), best_nan_indices, chunk_size)

        # 3) right_indices[0 .. (n_samples - n_nans - (left_count - n_nans))]
        #    = best_sorted_indices[(left_count - n_nans) .. (left_count - n_nans + ...)]
        chunk_size = (n_samples - n_nans - (left_count - n_nans)) * sizeof(int)
        memcpy(right_indices, best_sorted_indices + (left_count - n_nans), chunk_size)

    else:
        # 1) left_indices[0 .. left_count] = best_sorted_indices[0 .. left_count]
        chunk_size = left_count * sizeof(int)
        memcpy(left_indices, best_sorted_indices, chunk_size)

        # 2) right_indices[0 .. (n_samples - n_nans - left_count)]
        #    = best_sorted_indices[left_count .. left_count + (...)]
        chunk_size = (n_samples - n_nans - left_count) * sizeof(int)
        memcpy(right_indices, best_sorted_indices + left_count, chunk_size)

        # 3) right_indices[(right_count - n_nans) .. right_count]
        #    = best_nan_indices[0 .. n_nans] 
        chunk_size = n_nans * sizeof(int)
        memcpy(right_indices + (right_count - n_nans), best_nan_indices, chunk_size)

cdef void sort_pointer_array_count(SortItem* items, int* count, SortItem* output, const int n_samples, const int n_unique, const int offset) noexcept nogil:
    cdef int i
    cdef int key

    memset(count, 0, n_unique * sizeof(int))

    for i in range(n_samples):
        key = <int>items[i].value - offset
        count[key] += 1

    # Step 2: Compute cumulative counts
    for i in range(1, n_unique):
        count[i] += count[i - 1]

    i = n_samples - 1
    while i >= 0:
        key = <int>items[i].value - offset
        count[key] -= 1
        output[count[key]] = items[i]
        i -= 1

    for i in range(n_samples):
        items[i] = output[i]

cdef void analyze_X(const double[::1, :] X, bint* is_integer):
    cdef int n_columns = X.shape[1]
    cdef int n_rows = X.shape[0]
    cdef int i, j
    cdef double current_val
    cdef bint is_int
    cdef bint has_non_nan
    
    for j in range(n_columns):
        is_int = True
        has_non_nan = False

        for i in range(n_rows):
            current_val = X[i, j]

            if current_val != current_val:
                continue

            has_non_nan = True
            if current_val != floor(current_val):
                is_int = False
                break

        is_integer[j] = is_int and has_non_nan



cdef int** generate_permutations(const int n_classes, int* perm_count) noexcept nogil:
    """
    Generates all possible permutations of numbers from 0 to n_classes-1.
    """
    cdef int total_perms = 1
    cdef int i, j, k
    for i in range(2, n_classes + 1):
        total_perms *= i
    
    cdef int** permutations = <int**>malloc(total_perms * sizeof(int*))
    if permutations == NULL:
        perm_count[0] = 0
        return NULL
    
    cdef int* initial = <int*>malloc(n_classes * sizeof(int))
    if initial == NULL:
        free(permutations)
        perm_count[0] = 0
        return NULL
    
    for i in range(n_classes):
        initial[i] = i
    
    permutations[0] = initial
    cdef int curr_perm = 1
    
    cdef int* working = <int*>malloc(n_classes * sizeof(int))
    cdef int* indices = <int*>malloc(n_classes * sizeof(int))
    cdef int* new_perm  = NULL
    if working == NULL or indices == NULL:
        if working != NULL:
            free(working)
        if indices != NULL:
            free(indices)
        free(initial)
        free(permutations)
        perm_count[0] = 0
        return NULL
    
    for i in range(n_classes):
        indices[i] = 0
    
    i = 1
    while i < n_classes:
        if indices[i] < i:
            memcpy(working, permutations[curr_perm - 1], n_classes * sizeof(int))
            if i % 2 == 0:
                j = 0
            else:
                j = indices[i]
            k = working[j]
            working[j] = working[i]
            working[i] = k
            
            new_perm = <int*>malloc(n_classes * sizeof(int))
            if new_perm == NULL:
                for k in range(curr_perm):
                    if permutations[k] != NULL:
                        free(permutations[k])
                free(permutations)
                free(working)
                free(indices)
                perm_count[0] = 0
                return NULL
            
            memcpy(new_perm, working, n_classes * sizeof(int))
            permutations[curr_perm] = new_perm
            curr_perm += 1
            
            indices[i] += 1
            i = 1
        else:
            indices[i] = 0
            i += 1
    
    free(working)
    free(indices)
    
    perm_count[0] = total_perms
    return permutations
