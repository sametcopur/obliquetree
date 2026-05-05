from libc.math cimport INFINITY
from libc.stdlib cimport free

cdef double calculate_node_cost(TreeNode* node) noexcept nogil:
    return node.impurity * node.n_samples  # Artık total_samples'a bölmeye gerek yok

cdef int count_leaves(TreeNode* node) noexcept nogil:
    if node.is_leaf:
        return 1
    return count_leaves(node.left) + count_leaves(node.right)

cdef WeakestLink find_weakest_link(TreeNode* node) noexcept nogil:
    cdef WeakestLink result
    result.node = NULL
    result.improvement_score = INFINITY
    result.size_diff = 0
    
    if node.is_leaf or (node.left.is_leaf and node.right.is_leaf):
        return result
        
    # Calculate improvement for this split
    cdef double node_cost = calculate_node_cost(node)
    cdef double subtree_cost = calculate_node_cost(node.left) + calculate_node_cost(node.right)
    
    cdef int n_leaves = count_leaves(node)
    cdef double improvement = (node_cost - subtree_cost) / (n_leaves - 1)
    
    # Check children recursively
    cdef WeakestLink left_weakest = find_weakest_link(node.left)
    cdef WeakestLink right_weakest = find_weakest_link(node.right)
    
    # Initialize with current node
    result.node = node
    result.improvement_score = improvement
    result.size_diff = n_leaves - 1
    
    # Compare with left child
    if left_weakest.node != NULL and left_weakest.improvement_score < result.improvement_score:
        result = left_weakest
        
    # Compare with right child
    if right_weakest.node != NULL and right_weakest.improvement_score < result.improvement_score:
        result = right_weakest
        
    return result

cdef void free_subtree(TreeNode* node) noexcept nogil:
    if node == NULL:
        return

    if node.x != NULL:
        free(node.x)
    if node.pair != NULL:
        free(node.pair)
    if node.categories_go_left != NULL:
        free(node.categories_go_left)
    if node.leaf_coef != NULL:
        free(node.leaf_coef)
    if node.leaf_intercept_buf != NULL:
        free(node.leaf_intercept_buf)
    if node.value_multiclass != NULL:
        free(node.value_multiclass)

    free_subtree(node.left)
    free_subtree(node.right)
    free(node)

cdef void prune_subtree(TreeNode* node) noexcept nogil:
    if node == NULL or node.is_leaf:
        return

    # Free children subtrees
    if node.left != NULL:
        free_subtree(node.left)
    if node.right != NULL:
        free_subtree(node.right)

    # Reset node to leaf
    node.is_leaf = True
    node.left = NULL
    node.right = NULL

    # Free split-related memory
    if node.x != NULL:
        free(node.x)
        node.x = NULL
    if node.pair != NULL:
        free(node.pair)
        node.pair = NULL
    if node.categories_go_left != NULL:
        free(node.categories_go_left)
        node.categories_go_left = NULL
    if node.leaf_coef != NULL:
        free(node.leaf_coef)
        node.leaf_coef = NULL
    if node.leaf_intercept_buf != NULL:
        free(node.leaf_intercept_buf)
        node.leaf_intercept_buf = NULL
    node.leaf_n_coef = 0
    node.leaf_n_models = 0

    node.feature_idx = -1
    node.threshold = 0.0
    node.n_pair = 0
    node.n_category = 0

cdef void prune_tree(TreeNode* root, const double alpha) noexcept nogil:
    cdef WeakestLink weakest
    
    while True:
        weakest = find_weakest_link(root)
        
        # If no pruning candidates found or improvement is good enough, stop
        if weakest.node == NULL or weakest.improvement_score > alpha:
            break
            
        # Prune the weakest link
        prune_subtree(weakest.node)
