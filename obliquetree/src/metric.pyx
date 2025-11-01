from libc.math cimport INFINITY
from libc.stdlib cimport malloc, free, calloc
from libc.string cimport memset, memcpy
from .utils cimport sort_category_stats, generate_permutations, sort_category_stats_multiclass


cdef double calculate_node_value(const double[::1] sample_weight, 
                            const double[::1] y, 
                            const int* sample_indices,
                            const int n_samples) noexcept nogil:
    cdef double weighted_sum = 0.0, weight_sum = 0.0
    cdef double w = 1.0
    cdef int i, idx
    cdef bint check_sample_weight = &sample_weight[0] is not NULL

    for i in range(n_samples):
        idx = sample_indices[i]
        w = sample_weight[idx]

        weighted_sum += y[idx] * w
        weight_sum += w

    return weighted_sum / weight_sum

cdef void calculate_node_value_multiclass(const double[::1] sample_weight,
                                           const double[::1] y,
                                           const int* sample_indices,
                                           const int n_samples,
                                           const int n_classes,
                                           double** class_probs) noexcept nogil:
    class_probs_ = <double*>malloc(n_classes * sizeof(double))
    if class_probs == NULL:
        with gil:
            raise MemoryError()
    
    cdef double w = 1.0
    cdef double weight_sum = 0.0
    cdef int i, idx, class_idx
    
    # Initialize class counts to zero
    for i in range(n_classes):
        class_probs_[i] = 0.0
        
    # Calculate weighted sum for each class
    for i in range(n_samples):
        idx = sample_indices[i]
        w = sample_weight[idx]
            
        class_idx = <int>y[idx]
        class_probs_[class_idx] += w
        weight_sum += w
    
    # Convert counts to probabilities
    if weight_sum > 0:  # Avoid division by zero
        for i in range(n_classes):
            class_probs_[i] /= weight_sum

    class_probs[0] = class_probs_
            
cdef double calculate_node_gini(const int* sample_indices, 
                                const double[::1] sample_weight, 
                                const double[::1] y, 
                                const int n_samples, 
                                int n_class = 2) noexcept nogil:
    cdef double* counts = <double*>calloc(n_class, sizeof(double))
    if not counts:
        with gil:
            raise MemoryError()

    cdef double total_weight = 0.0,  node_gini = 1.0
    cdef int i, idx
    cdef double p_k, w

    for i in range(n_samples):
        idx = sample_indices[i]
        w = sample_weight[idx]
        counts[<int>y[idx]] += w
        total_weight += w

    for i in range(n_class):
        p_k = counts[i] / total_weight
        node_gini -= p_k * p_k

    free(counts)

    return node_gini

cdef inline double calculate_gini_impurity(double first_subset_weight,
                                         double total_sample_weight,
                                         double total_weighted_score,
                                         double first_weighted_score) noexcept nogil:
    cdef double second_weighted_score = total_weighted_score - first_weighted_score
    cdef double second_subset_weight = total_sample_weight - first_subset_weight
    
    cdef double first_proportion = first_weighted_score / first_subset_weight
    cdef double first_gini = first_subset_weight * 2.0 * first_proportion * (1.0 - first_proportion)
    
    cdef double second_proportion = second_weighted_score / second_subset_weight
    cdef double second_gini = second_subset_weight * 2.0 * second_proportion * (1.0 - second_proportion)
    
    return (first_gini + second_gini) / total_sample_weight

cdef double calculate_node_mse(const int* sample_indices,
                             const double[::1] sample_weight,
                             const double[::1] y,
                             const int n_samples) noexcept nogil:
    cdef double weighted_sum = 0.0
    cdef double total_weight = 0.0
    cdef int i, idx
    cdef double w, target_value
    
    # Calculate weighted mean
    for i in range(n_samples):
        idx = sample_indices[i]
        w = sample_weight[idx]
        weighted_sum += w * y[idx]
        total_weight += w
    
    cdef double weighted_mean = weighted_sum / total_weight
    
    # Calculate MSE
    cdef double mse = 0.0
    for i in range(n_samples):
        idx = sample_indices[i]
        w = sample_weight[idx]
        target_value = y[idx]
        mse += w * (target_value - weighted_mean) * (target_value - weighted_mean)
    
    return mse / total_weight

cdef inline double calculate_mse_split(double first_subset_weight,
                                     double total_sample_weight,
                                     double total_weighted_sum,
                                     double first_weighted_sum) noexcept nogil:
    cdef double second_weighted_sum = total_weighted_sum - first_weighted_sum
    cdef double second_subset_weight = total_sample_weight - first_subset_weight
    
    # Calculate means for both splits
    cdef double first_mean = first_weighted_sum / first_subset_weight
    cdef double second_mean = second_weighted_sum / second_subset_weight
    
    # For regression trees, we want to minimize the total weighted variance
    # This is equivalent to minimizing the sum of squared errors
    return -(first_subset_weight * first_mean * first_mean + 
             second_subset_weight * second_mean * second_mean)
             
cdef double feature_impurity(
    const bint task,
    const SortItem* sort_buffer,
    const double* y,
    const double* sample_weight,  # Added sample_weight parameter
    const int n_samples,
    const int n_nans,
    const int min_samples_leaf,
    const double total_weighted_sum,
    const double total_weight,
    double* out_threshold,
    int* out_left_count,
    bint* out_missing_go_left,
    const double nan_weighted_sum,
    const double nan_weight) noexcept nogil:

    cdef int i, idx
    cdef double w = 1.0
    cdef double weighted_left_sum = 0.0, weighted_left_weight = 0.0
    cdef double weighted_left_weight_with_nan, curr_impurity
    cdef double min_impurity = INFINITY
    cdef double curr_val, next_valx
    
    for i in range(n_samples - n_nans - 1):
        # Get sample weight if available
        idx = sort_buffer[i].index
        curr_val = sort_buffer[i].value
        next_val = sort_buffer[i + 1].value 

        w = sample_weight[idx]
        
        weighted_left_sum += y[idx] * w
        weighted_left_weight += w
        
        if curr_val == next_val or weighted_left_weight < min_samples_leaf:
            continue

        if (total_weight - weighted_left_weight) < min_samples_leaf:
            break
            
        # Non-nan case
        if weighted_left_weight >= min_samples_leaf and (total_weight - weighted_left_weight) >= min_samples_leaf and n_nans != 0:

            if task == 0:
                curr_impurity = calculate_gini_impurity(weighted_left_weight, total_weight, total_weighted_sum, weighted_left_sum)
            else:
                curr_impurity = calculate_mse_split(weighted_left_weight, total_weight, total_weighted_sum, weighted_left_sum)

            if curr_impurity < min_impurity:
                min_impurity = curr_impurity
                out_threshold[0] = 0.5 * (curr_val + next_val)
                out_left_count[0] = i + 1
                out_missing_go_left[0] = False
        
        # NaN left case
        weighted_left_sum_with_nan = weighted_left_sum + nan_weighted_sum
        weighted_left_weight_with_nan = weighted_left_weight + nan_weight
        
        if weighted_left_weight_with_nan >= min_samples_leaf and (total_weight - weighted_left_weight_with_nan) >= min_samples_leaf:

            if task == 0:
                curr_impurity = calculate_gini_impurity(weighted_left_weight_with_nan, total_weight, total_weighted_sum, weighted_left_sum_with_nan)
            else:
                curr_impurity = calculate_mse_split(weighted_left_weight_with_nan, total_weight, total_weighted_sum, weighted_left_sum_with_nan)

            if curr_impurity < min_impurity:
                min_impurity = curr_impurity
                out_threshold[0] = 0.5 * (curr_val + next_val)
                out_left_count[0] = i + 1 + n_nans
                out_missing_go_left[0] = True
                
    return min_impurity

cdef double category_impurity(
    const bint task,
    const int unique_count,
    const double total_weighted_sum,
    const double total_weight,
    const double nan_weighted_sum,
    const double nan_weight,
    CategoryStat* stats,
    const int n_samples,
    const int n_nans,
    const int min_samples_leaf,
    double* out_threshold,
    int* out_left_count,
    bint* out_missing_go_left) noexcept nogil:
    
    cdef int i
    cdef double min_impurity = INFINITY
    
    cdef double weighted_left_sum = 0.0, weighted_left_weight = 0.0
    cdef int left_count = 0
    
    cdef double weighted_left_sum_with_nan, weighted_left_weight_with_nan
    cdef double curr_impurity

    # Her split için NaN'ları hem sağda hem solda dene
    for i in range(1, unique_count):
        # Önceki kategoriyi sol gruba ekle
        weighted_left_sum += stats[i-1].y_sum
        weighted_left_weight += stats[i-1].weight
        left_count += stats[i-1].count
        
        # NaN'lar sağda
        if weighted_left_weight >= min_samples_leaf and (total_weight - weighted_left_weight) >= min_samples_leaf and n_nans != 0:
            
            if task == 0:
                curr_impurity = calculate_gini_impurity(weighted_left_weight, total_weight, total_weighted_sum, weighted_left_sum)
            else:
                curr_impurity = calculate_mse_split(weighted_left_weight, total_weight, total_weighted_sum, weighted_left_sum)

            if curr_impurity < min_impurity:
                min_impurity = curr_impurity
                out_threshold[0] = i
                out_left_count[0] = left_count
                out_missing_go_left[0] = False


        # NaN'lar solda
        weighted_left_sum_with_nan = weighted_left_sum + nan_weighted_sum
        weighted_left_weight_with_nan = weighted_left_weight + nan_weight
        if weighted_left_weight_with_nan >= min_samples_leaf and (total_weight - weighted_left_weight_with_nan) >= min_samples_leaf:
            
            if task == 0:
                curr_impurity = calculate_gini_impurity(weighted_left_weight_with_nan, total_weight, total_weighted_sum, weighted_left_sum_with_nan)
            else:
                curr_impurity = calculate_mse_split(weighted_left_weight_with_nan, total_weight, total_weighted_sum, weighted_left_sum_with_nan)

            if curr_impurity < min_impurity:
                min_impurity = curr_impurity
                out_threshold[0] = i
                out_left_count[0] = left_count + n_nans
                out_missing_go_left[0] = True

    return min_impurity

cdef double find_best_split_for_feature(
    const bint task,
    const SortItem* sort_buffer,
    const double* sample_weight,  # Added sample_weight parameter
    const double* y,
    const int* nan_indices,
    const int n_samples,
    const int n_nans,
    const int min_samples_leaf,
    double* out_threshold,
    int* out_left_count,
    bint* out_missing_go_left) noexcept nogil:

    cdef int i, idx
    cdef double min_impurity
    cdef double w = 1.0  # Default weight if sample_weight is NULL
    
    # Initialize weighted sums instead of just counts
    cdef double total_weighted_sum = 0.0
    cdef double nan_weighted_sum = 0.0
    cdef double total_weight = 0.0
    cdef double nan_weight = 0.0

    # Calculate weighted sums for non-nan samples
    for i in range(n_samples - n_nans):
        idx = sort_buffer[i].index
        w = sample_weight[idx]
        total_weighted_sum += y[idx] * w
        total_weight += w
    
    # Calculate weighted sums for nan samples
    for i in range(n_nans):
        idx = nan_indices[i]
        w = sample_weight[idx]
        nan_weighted_sum += y[idx] * w
        nan_weight += w
    
    total_weighted_sum += nan_weighted_sum
    total_weight += nan_weight

    min_impurity = feature_impurity(
        task,
        sort_buffer,
        y, 
        sample_weight,
        n_samples, 
        n_nans, 
        min_samples_leaf, 
        total_weighted_sum, 
        total_weight,
        out_threshold, 
        out_left_count, 
        out_missing_go_left, 
        nan_weighted_sum, 
        nan_weight)
    
    return min_impurity

cdef double find_best_split_for_categorical(
    const bint task,
    SortItem* sort_buffer,
    const double* sample_weight,  # Parametre zaten varmış, güzel
    const double* y,
    const int* nan_indices,
    CategoryStat* stats,
    const int n_samples,
    const int n_nans,
    const int min_samples_leaf,
    double* out_threshold,
    int* out_left_count,
    bint* out_missing_go_left) noexcept nogil:
    
    cdef int i, j, idx
    cdef double current_value
    cdef double min_impurity = INFINITY
    cdef double w = 1.0  # Varsayılan ağırlık
    
    # Weighted toplamlar için değişkenler
    cdef double total_weighted_sum = 0.0
    cdef double nan_weighted_sum = 0.0
    cdef double total_weight = 0.0
    cdef double nan_weight = 0.0
    cdef CategoryStat* current_stat
    
    # NaN değerlerin weighted toplamını hesapla
    for i in range(n_nans):
        idx = nan_indices[i]
        w = sample_weight[idx]
        nan_weighted_sum += y[idx] * w
        nan_weight += w
    
    # İlk kategoriyi initialize et
    current_value = sort_buffer[0].value
    idx = sort_buffer[0].index
    stats[0].value = current_value
    w = sample_weight[idx]
    stats[0].y_sum = y[idx] * w
    stats[0].weight = w  # Yeni alan: kategori ağırlığı
    stats[0].count = 1

    cdef int unique_count = 1
    cdef double prev_value = current_value
    
    # Benzersiz kategorileri ve istatistiklerini bul (NaN'lar hariç)
    for i in range(1, n_samples - n_nans):
        idx = sort_buffer[i].index
        w = sample_weight[idx]
        current_value = sort_buffer[i].value
        if current_value != prev_value:
            current_stat = &stats[unique_count]
            current_stat.value = current_value
            current_stat.y_sum = y[idx] * w
            current_stat.weight = w
            current_stat.count = 1
            unique_count += 1
            prev_value = current_value
        else:
            current_stat = &stats[unique_count - 1]
            current_stat.y_sum += y[idx] * w
            current_stat.weight += w
            current_stat.count += 1

    # Toplam weighted sum hesapla (NaN'lar dahil)
    total_weighted_sum = nan_weighted_sum
    total_weight = nan_weight
    for i in range(unique_count):
        total_weighted_sum += stats[i].y_sum
        total_weight += stats[i].weight

    sort_category_stats(stats, unique_count)  # Y ortalamasına göre sırala
    
    # Sıralanmış kategorilere göre yeni indeks eşlemesi oluştur
    cdef int* new_indices = <int*>malloc((n_samples - n_nans) * sizeof(int))
    if new_indices == NULL:
        with gil:
            raise MemoryError()
    
    cdef int max_cat = 0
    cdef int stats_val
    for i in range(unique_count):
        stats_val = <int>stats[i].value
        if stats_val > max_cat:
            max_cat = stats_val

    # Kategori sayılarını tutacak array
    cdef int* stats_count = <int*>calloc(max_cat + 1, sizeof(int))
    if stats_count == NULL:
        with gil:
            raise MemoryError()

    # Her kategorinin başlangıç pozisyonunu hesapla
    cdef int cumsum = 0
    cdef int temp
    for i in range(unique_count):
        stats_val = <int>stats[i].value
        stats_count[stats_val] = cumsum
        cumsum += stats[i].count

    # Tek geçişte indeksleri yerleştir
    cdef int curr_val
    for j in range(n_samples - n_nans):
        curr_val = <int>sort_buffer[j].value
        new_indices[stats_count[curr_val]] = sort_buffer[j].index
        stats_count[curr_val] += 1

    # sort_buffer'ı güncelle
    for i in range(n_samples - n_nans):
        sort_buffer[i].index = new_indices[i]

    # Belleği temizle
    free(stats_count)
    free(new_indices)

    min_impurity = category_impurity(
        task,
        unique_count,
        total_weighted_sum,
        total_weight,
        nan_weighted_sum,
        nan_weight,
        stats,
        n_samples,
        n_nans,
        min_samples_leaf,
        out_threshold,
        out_left_count,
        out_missing_go_left)

    return min_impurity


cdef inline double calculate_gini_impurity_multiclass(
    const double* left_class_weights,
    const double* right_class_weights,
    const double left_total_weight,
    const double right_total_weight,
    const int n_classes,
    const double total_weight) noexcept nogil:
    
    cdef double left_gini = 1.0
    cdef double right_gini = 1.0
    cdef double p_square_sum_left = 0.0
    cdef double p_square_sum_right = 0.0
    cdef int c
    
    # Calculate left node impurity
    if left_total_weight > 0:
        for c in range(n_classes):
            if left_class_weights[c] > 0:  # Avoid unnecessary division
                p = left_class_weights[c] / left_total_weight
                p_square_sum_left += p * p
        left_gini = (1.0 - p_square_sum_left) * left_total_weight
    
    # Calculate right node impurity
    if right_total_weight > 0:
        for c in range(n_classes):
            if right_class_weights[c] > 0:  # Avoid unnecessary division
                p = right_class_weights[c] / right_total_weight
                p_square_sum_right += p * p
        right_gini = (1.0 - p_square_sum_right) * right_total_weight
    
    return (left_gini + right_gini) / total_weight

cdef double feature_impurity_multiclass(
    const SortItem* sort_buffer,
    const double* y,
    const double* sample_weight,
    const int n_samples,
    const int n_nans,
    const int min_samples_leaf,
    const int n_classes,
    const double* total_class_weights,
    const double total_weight,
    double* out_threshold,
    int* out_left_count,
    bint* out_missing_go_left,
    const double* nan_class_weights,
    const double nan_weight) noexcept nogil:

    cdef double* left_class_weights = <double*>malloc(n_classes * sizeof(double))
    cdef double* right_class_weights = <double*>malloc(n_classes * sizeof(double))
    
    if left_class_weights == NULL or right_class_weights == NULL:
        if left_class_weights != NULL:
            free(left_class_weights)
        if right_class_weights != NULL:
            free(right_class_weights)
        with gil:
            raise MemoryError()
    
    cdef int i, j, idx, class_idx
    cdef double w = 1.0
    cdef double weighted_left_weight = 0.0
    cdef double curr_val, next_val
    cdef double min_impurity = INFINITY
    cdef double curr_impurity
    cdef double weighted_left_weight_with_nan
    
    # Initialize left class weights to 0
    for i in range(n_classes):
        left_class_weights[i] = 0.0
    
    try:
        for i in range(n_samples - n_nans - 1):
            idx = sort_buffer[i].index
            curr_val = sort_buffer[i].value
            next_val = sort_buffer[i + 1].value
            w = sample_weight[idx]
            
            class_idx = <int>y[idx]
            left_class_weights[class_idx] += w
            weighted_left_weight += w
            
            if curr_val == next_val or weighted_left_weight < min_samples_leaf:
                continue
                
            if (total_weight - weighted_left_weight) < min_samples_leaf:
                break
            
            # Calculate right class weights for non-nan case
            for j in range(n_classes):
                right_class_weights[j] = total_class_weights[j] - left_class_weights[j]
            
            # Non-nan case
            if weighted_left_weight >= min_samples_leaf and (total_weight - weighted_left_weight) >= min_samples_leaf and n_nans != 0:
                curr_impurity = calculate_gini_impurity_multiclass(
                    left_class_weights,
                    right_class_weights,
                    weighted_left_weight,
                    total_weight - weighted_left_weight,
                    n_classes,
                    total_weight
                )
                
                if curr_impurity < min_impurity:
                    min_impurity = curr_impurity
                    out_threshold[0] = 0.5 * (curr_val + next_val)
                    out_left_count[0] = i + 1
                    out_missing_go_left[0] = False
                    
            # NaN left case
            weighted_left_weight_with_nan = weighted_left_weight + nan_weight
            
            if weighted_left_weight_with_nan >= min_samples_leaf and (total_weight - weighted_left_weight_with_nan) >= min_samples_leaf:
                # Calculate right class weights for nan-left case
                for j in range(n_classes):
                    left_class_weights[j] += nan_class_weights[j]  # Temporarily add nan weights
                    right_class_weights[j] = total_class_weights[j] - left_class_weights[j]
                
                curr_impurity = calculate_gini_impurity_multiclass(
                    left_class_weights,
                    right_class_weights,
                    weighted_left_weight_with_nan,
                    total_weight - weighted_left_weight_with_nan,
                    n_classes,
                    total_weight
                )
                
                # Remove nan weights from left weights for next iteration
                for j in range(n_classes):
                    left_class_weights[j] -= nan_class_weights[j]
                
                if curr_impurity < min_impurity:
                    min_impurity = curr_impurity
                    out_threshold[0] = 0.5 * (curr_val + next_val)
                    out_left_count[0] = i + 1 + n_nans
                    out_missing_go_left[0] = True
    
    finally:
        free(left_class_weights)
        free(right_class_weights)
    
    return min_impurity

cdef double find_best_split_for_feature_multiclass(
    const SortItem* sort_buffer,
    const double* sample_weight,
    const double* y,
    const int* nan_indices,
    const int n_samples,
    const int n_nans,
    const int min_samples_leaf,
    const int n_classes,
    double* out_threshold,
    int* out_left_count,
    bint* out_missing_go_left) noexcept nogil:

    cdef int i, c
    cdef double w = 1.0
    
    # Initialize class weights and totals
    cdef double* total_class_weights = <double*>malloc(n_classes * sizeof(double))
    cdef double* nan_class_weights = <double*>malloc(n_classes * sizeof(double))
    cdef double total_weight = 0.0
    cdef double nan_weight = 0.0
    
    if total_class_weights == NULL or nan_class_weights == NULL:
        if total_class_weights != NULL:
            free(total_class_weights)
        if nan_class_weights != NULL:
            free(nan_class_weights)
        with gil:
            raise MemoryError()
    
    # Initialize arrays to zero
    for c in range(n_classes):
        total_class_weights[c] = 0.0
        nan_class_weights[c] = 0.0
    
    # Calculate total weights for non-nan samples
    for i in range(n_samples - n_nans):
        idx = sort_buffer[i].index
        w = sample_weight[idx]
        class_idx = <int>y[idx]

        total_class_weights[class_idx] += w
        total_weight += w

    # Calculate total weights for nan samples
    for i in range(n_nans):
        w = sample_weight[nan_indices[i]]
        class_idx = <int>y[nan_indices[i]]

        nan_class_weights[class_idx] += w
        nan_weight += w
        total_class_weights[class_idx] += w
        total_weight += w

    # Find best split
    cdef double min_impurity = feature_impurity_multiclass(
        sort_buffer,
        y,
        sample_weight,
        n_samples,
        n_nans,
        min_samples_leaf,
        n_classes,
        total_class_weights,
        total_weight,
        out_threshold,
        out_left_count,
        out_missing_go_left,
        nan_class_weights,
        nan_weight
    )
    
    free(total_class_weights)
    free(nan_class_weights)
    
    return min_impurity

cdef double category_impurity_multiclass(
    const int unique_count,
    const double* total_class_weights,
    const double total_weight,
    const double* nan_class_weights,
    const double nan_weight,
    CategoryStat* stats,
    const int n_samples,
    const int n_nans,
    const int min_samples_leaf,
    const int n_classes,
    double* out_threshold,
    int* out_left_count,
    bint* out_missing_go_left) noexcept nogil:
    
    cdef int i, j
    cdef double min_impurity = INFINITY
    cdef double weighted_left_weight = 0.0
    cdef int left_count = 0
    cdef double curr_impurity
    cdef double weighted_left_weight_with_nan
    
    # Allocate memory for class weights
    cdef double* left_class_weights = <double*>malloc(n_classes * sizeof(double))
    cdef double* right_class_weights = <double*>malloc(n_classes * sizeof(double))
    
    if left_class_weights == NULL or right_class_weights == NULL:
        if left_class_weights != NULL:
            free(left_class_weights)
        if right_class_weights != NULL:
            free(right_class_weights)
        with gil:
            raise MemoryError()
    
    try:
        # Initialize left class weights
        for i in range(n_classes):
            left_class_weights[i] = 0.0
        
        # Try each split point
        for i in range(1, unique_count):
            # Add previous category to left group
            for j in range(n_classes):
                left_class_weights[j] += stats[i-1].class_weights[j]
            weighted_left_weight += stats[i-1].weight
            left_count += stats[i-1].count
            
            # NaN right case
            if weighted_left_weight >= min_samples_leaf and (total_weight - weighted_left_weight) >= min_samples_leaf:
                # Calculate right class weights
                for j in range(n_classes):
                    right_class_weights[j] = total_class_weights[j] - left_class_weights[j]
                
                curr_impurity = calculate_gini_impurity_multiclass(
                    left_class_weights,
                    right_class_weights,
                    weighted_left_weight,
                    total_weight - weighted_left_weight,
                    n_classes,
                    total_weight
                )
                
                if curr_impurity < min_impurity:
                    min_impurity = curr_impurity
                    out_threshold[0] = i
                    out_left_count[0] = left_count
                    out_missing_go_left[0] = False
            
            # NaN left case
            weighted_left_weight_with_nan = weighted_left_weight + nan_weight
            if weighted_left_weight_with_nan >= min_samples_leaf and (total_weight - weighted_left_weight_with_nan) >= min_samples_leaf:
                # Add nan weights to left weights temporarily
                for j in range(n_classes):
                    left_class_weights[j] += nan_class_weights[j]
                    right_class_weights[j] = total_class_weights[j] - left_class_weights[j]
                
                curr_impurity = calculate_gini_impurity_multiclass(
                    left_class_weights,
                    right_class_weights,
                    weighted_left_weight_with_nan,
                    total_weight - weighted_left_weight_with_nan,
                    n_classes,
                    total_weight
                )
                
                # Remove nan weights for next iteration
                for j in range(n_classes):
                    left_class_weights[j] -= nan_class_weights[j]
                
                if curr_impurity < min_impurity:
                    min_impurity = curr_impurity
                    out_threshold[0] = i
                    out_left_count[0] = left_count + n_nans
                    out_missing_go_left[0] = True
    
    finally:
        free(left_class_weights)
        free(right_class_weights)
    
    return min_impurity

cdef double find_best_split_for_categorical_multiclass(
    SortItem* sort_buffer,
    const double* sample_weight,
    const double* y,
    const int* nan_indices,
    CategoryStat* stats,
    const int n_samples,
    const int n_nans,
    const int min_samples_leaf,
    const int n_classes,
    double* out_threshold,
    int* out_left_count,
    bint* out_missing_go_left) noexcept nogil:

    cdef int i, j, class_idx, idx
    cdef double current_value
    cdef double min_impurity = INFINITY
    cdef double w = 1.0
    cdef double total_weight = 0.0
    cdef double nan_weight = 0.0
    cdef int unique_count = 0
    cdef double prev_value
    cdef double best_impurity = INFINITY
    
    cdef CategoryStat* current_stat = NULL
    cdef CategoryStat* best_stats = NULL

    cdef int max_cat
    cdef int stats_val
    cdef int* stats_count = NULL
    cdef int* new_indices = NULL
    cdef int** perm_orders = NULL
    cdef int perm_count = 0
    cdef int curr_val
    cdef int cumsum = 0
    cdef double out_threshold_c = 0.0
    cdef int out_left_count_c = 0
    cdef bint out_missing_go_left_c = True

    # Bellek ayrımı: total_class_weights, nan_class_weights
    cdef double* total_class_weights = <double*>malloc(n_classes * sizeof(double))
    cdef double* nan_class_weights   = <double*>malloc(n_classes * sizeof(double))
    # Artık temp_class_weights kullanmıyoruz

    if total_class_weights == NULL or nan_class_weights == NULL:
        if total_class_weights != NULL:
            free(total_class_weights)
        if nan_class_weights != NULL:
            free(nan_class_weights)
        with gil:
            raise MemoryError()
    
    try:
        memset(total_class_weights, 0, n_classes * sizeof(double))
        memset(nan_class_weights,   0, n_classes * sizeof(double))
        
        for i in range(n_nans):
            idx = nan_indices[i]
            w = sample_weight[idx]
            class_idx = <int>y[idx]
            nan_class_weights[class_idx] += w
            nan_weight += w
            total_class_weights[class_idx] += w
            total_weight += w

        if n_samples > n_nans:
            idx = sort_buffer[0].index
            current_value = sort_buffer[0].value
            current_stat = &stats[0]
            
            # Her kategori için malloc yapıyoruz
            current_stat.class_weights = <double*>malloc(n_classes * sizeof(double))
            if current_stat.class_weights == NULL:
                with gil:
                    raise MemoryError()

            memset(current_stat.class_weights, 0, n_classes * sizeof(double))
            w = sample_weight[idx]
            class_idx = <int>y[idx]
            current_stat.class_weights[class_idx] = w
            current_stat.weight = w
            current_stat.count = 1
            current_stat.value = current_value

            total_class_weights[class_idx] += w
            total_weight += w
            
            unique_count = 1
            prev_value = current_value

            # Kalan örnekler
            for i in range(1, n_samples - n_nans):
                idx = sort_buffer[i].index
                w = sample_weight[idx]
                current_value = sort_buffer[i].value
                class_idx = <int>y[idx]
                
                if current_value != prev_value:
                    # Yeni kategori için yeni malloc
                    current_stat = &stats[unique_count]
                    current_stat.value = current_value
                    current_stat.class_weights = <double*>malloc(n_classes * sizeof(double))
                    
                    if current_stat.class_weights == NULL:
                        # Mevcut tüm allocated bellekleri serbest bırak
                        for j in range(unique_count):
                            if stats[j].class_weights != NULL:
                                free(stats[j].class_weights)
                                stats[j].class_weights = NULL
                        with gil:
                            raise MemoryError()
                    
                    memset(current_stat.class_weights, 0, n_classes * sizeof(double))
                    current_stat.class_weights[class_idx] = w
                    current_stat.weight = w
                    current_stat.count = 1

                    unique_count += 1
                    prev_value = current_value
                else:
                    # Aynı kategoriye ekle
                    current_stat.class_weights[class_idx] += w
                    current_stat.weight += w
                    current_stat.count += 1
                
                total_class_weights[class_idx] += w
                total_weight += w

            max_cat = 0
            for i in range(unique_count):
                stats_val = <int>stats[i].value
                if stats_val > max_cat:
                    max_cat = stats_val

            # Kategori dizilişlerinin cumsum hesaplaması için
            stats_count = <int*>calloc(max_cat + 1, sizeof(int))
            if stats_count == NULL:
                with gil:
                    raise MemoryError()

            new_indices = <int*>calloc(n_samples - n_nans, sizeof(int))
            if new_indices == NULL:
                with gil:
                    raise MemoryError()

            best_stats = <CategoryStat*>malloc(unique_count * sizeof(CategoryStat))
            if best_stats == NULL:
                with gil:
                    raise MemoryError()
            # Initialize
            memcpy(best_stats, stats, unique_count * sizeof(CategoryStat))

            # Tüm permütasyonları üret
            perm_orders = generate_permutations(n_classes, &perm_count)

            for k in range(perm_count):
                sort_category_stats_multiclass(stats, perm_orders[k], unique_count, n_classes)

                min_impurity = category_impurity_multiclass(
                    unique_count,
                    total_class_weights,
                    total_weight,
                    nan_class_weights,
                    nan_weight,
                    stats,
                    n_samples,
                    n_nans,
                    min_samples_leaf,
                    n_classes,
                    &out_threshold_c,
                    &out_left_count_c,
                    &out_missing_go_left_c,
                )

                # Daha iyi bir split bulursak güncelle
                if min_impurity < best_impurity:
                    out_threshold[0] = out_threshold_c
                    out_left_count[0] = out_left_count_c
                    out_missing_go_left[0] = out_missing_go_left_c
                    best_impurity = min_impurity

                    memcpy(best_stats, stats, unique_count * sizeof(CategoryStat))

            cumsum = 0
            for i in range(unique_count):
                stats_val = <int>best_stats[i].value
                stats_count[stats_val] = cumsum
                cumsum += best_stats[i].count

            # Yeni sıralamaya göre indeksleri diz
            for j in range(n_samples - n_nans):
                curr_val = <int>sort_buffer[j].value
                new_indices[stats_count[curr_val]] = sort_buffer[j].index
                stats_count[curr_val] += 1

            for i in range(n_samples - n_nans):
                sort_buffer[i].index = new_indices[i]
            
            memcpy(stats, best_stats, unique_count * sizeof(CategoryStat))

    finally:

        # Fonksiyonun sonunda toplu free
        free(total_class_weights)
        free(nan_class_weights)

        if stats_count != NULL:
            free(stats_count)
        if new_indices != NULL:
            free(new_indices)
        if best_stats != NULL:
            free(best_stats)
        
        # class_weights belleklerini serbest bırak
        if stats != NULL:
            for i in range(unique_count):
                if stats[i].class_weights != NULL:
                    free(stats[i].class_weights)
                    stats[i].class_weights = NULL

        # permütasyon belleklerini serbest bırak
        if perm_orders != NULL:
            for i in range(perm_count):
                if perm_orders[i] != NULL:
                    free(perm_orders[i])
            free(perm_orders)

    return best_impurity


cdef double calculate_impurity(
                const bint is_categorical, 
                const int n_classes,
                SortItem* sort_buffer,
                const double* sample_weight,
                const double* y,
                int* nan_indices,
                CategoryStat* categorical_stats,
                const int n_samples,
                const int n_nans,
                const int min_samples_leaf,
                double *threshold_c,
                int* left_count_c,
                bint* missing_go_left,
                const bint task) noexcept nogil:

    cdef double impurity_c

    if is_categorical:
        if n_classes > 2:
            impurity_c = find_best_split_for_categorical_multiclass(
                sort_buffer,
                sample_weight,
                y,
                nan_indices,
                categorical_stats,
                n_samples,
                n_nans,
                min_samples_leaf,
                n_classes,
                threshold_c,
                left_count_c,
                missing_go_left,
                )
            

        else:
            impurity_c = find_best_split_for_categorical(
                task,
                sort_buffer,
                sample_weight,
                y,
                nan_indices,
                categorical_stats,
                n_samples,
                n_nans,
                min_samples_leaf,
                threshold_c,
                left_count_c,
                missing_go_left,
                )

    else: 
        if n_classes > 2:
            impurity_c = find_best_split_for_feature_multiclass(
                sort_buffer,
                sample_weight,
                y,
                nan_indices,
                n_samples,
                n_nans,
                min_samples_leaf,
                n_classes,
                threshold_c,
                left_count_c,
                missing_go_left,
            )

        else:
            impurity_c = find_best_split_for_feature(
                task,
                sort_buffer,
                sample_weight,
                y,
                nan_indices,
                n_samples,
                n_nans,
                min_samples_leaf,
                threshold_c,
                left_count_c,
                missing_go_left,
            )
    return impurity_c