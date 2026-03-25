from libc.math cimport INFINITY, exp, log, fabs
from libc.stdlib cimport malloc, free, calloc
from libc.string cimport memcpy, memset

import numpy as np
cimport numpy as np

from cython.parallel cimport prange
cimport openmp

from .utils cimport sort_pointer_array


# ---------------------------------------------------------------------------
# Helper inlines
# ---------------------------------------------------------------------------
cdef inline double get_y(double y, double current_class) noexcept nogil:
    return 1.0 if y == current_class else 0.0

cdef inline double sigmoid(const double x) noexcept nogil:
    return 1.0 / (1.0 + exp(-x))

# ---------------------------------------------------------------------------
# Gathered mat-vec:  work directly on global X via sample_indices + col_indices
#   X is the *original* (full_N x full_d) Fortran-order matrix.
#   We read only the rows in sample_indices[0..N) and cols in cols[0..d).
# ---------------------------------------------------------------------------
cdef inline void matvec_gather_N(
    const double* X, Py_ssize_t full_N,          # global X ptr + leading dim
    const int* row_idx, Py_ssize_t N,             # which rows
    const int* col_idx, Py_ssize_t d,             # which cols
    const double* w,                              # (d,)
    double* out,                                  # (N,)
) noexcept nogil:
    """out[i] = sum_j  X[row_idx[i], col_idx[j]] * w[j]"""
    cdef Py_ssize_t i, j
    cdef int ri
    for i in range(N):
        out[i] = 0.0
    for j in range(d):
        for i in range(N):
            out[i] += X[row_idx[i] + <Py_ssize_t>col_idx[j] * full_N] * w[j]


cdef inline void matvec_gather_T(
    const double* X, Py_ssize_t full_N,
    const int* row_idx, Py_ssize_t N,
    const int* col_idx, Py_ssize_t d,
    const double* v,                              # (N,)
    double* out,                                  # (d,)
) noexcept nogil:
    """out[j] = sum_i  X[row_idx[i], col_idx[j]] * v[i]"""
    cdef Py_ssize_t i, j
    for j in range(d):
        out[j] = 0.0
        for i in range(N):
            out[j] += X[row_idx[i] + <Py_ssize_t>col_idx[j] * full_N] * v[i]


# ---------------------------------------------------------------------------
# Loss / gradient  (pointer-based, fully nogil, gathered access)
# ---------------------------------------------------------------------------
cdef double fun_and_grad_nogil(
    const double* X, Py_ssize_t full_N,
    const int* row_idx, const int* col_idx,
    const double* y,
    const double* sample_weight,
    double* w,
    const Py_ssize_t N,
    const Py_ssize_t d,
    const double gamma,
    const double eps,
    const double total_weight,
    double* z,
    double* p,
    double* dp_dz,
    double* grad_w,
    double* dP_R1_w,
    double* dP_L1_w,
    double* tmp_y_dp_vec,
    double* dS_R_w,
    double* y_dp_vec) noexcept nogil:

    cdef Py_ssize_t i
    cdef double S_L, S_R, p_y_sum, p_1_y_sum, tmp
    cdef double P_L1, P_R1
    cdef double impurity_L, impurity_R
    cdef double loss
    cdef double factor_L, factor_R
    cdef double dImp_L, dImp_R
    cdef double denom_L, denom_R
    cdef double tmp_y_dp, gj
    cdef double one_minus_tmp, yi, sw_i
    cdef double dp_dz_i, dS_R_w_i, dP_L1_w_i, dP_R1_w_i

    # z = X[rows, cols] @ w
    matvec_gather_N(X, full_N, row_idx, N, col_idx, d, w, z)

    S_L = 0.0
    p_y_sum = 0.0
    p_1_y_sum = 0.0

    for i in range(N):
        sw_i = sample_weight[i]
        tmp = sigmoid(z[i])
        one_minus_tmp = 1.0 - tmp
        p[i] = tmp
        yi = y[i]

        S_L += tmp * sw_i
        p_y_sum += tmp * yi * sw_i
        p_1_y_sum += one_minus_tmp * yi * sw_i

        dp_dz_i = gamma * tmp * one_minus_tmp * sw_i
        dp_dz[i] = dp_dz_i
        y_dp_vec[i] = yi * dp_dz_i

    S_R = total_weight - S_L
    S_L += eps
    S_R += eps

    P_L1 = p_y_sum / S_L
    P_R1 = p_1_y_sum / S_R
    impurity_L = P_L1 * (1.0 - P_L1)
    impurity_R = P_R1 * (1.0 - P_R1)
    loss = S_L * impurity_L + S_R * impurity_R

    matvec_gather_T(X, full_N, row_idx, N, col_idx, d, dp_dz, grad_w)

    for i in range(d):
        dS_R_w[i] = -grad_w[i]

    denom_L = S_L * S_L
    denom_R = S_R * S_R
    factor_L = 1.0 - 2.0 * P_L1
    factor_R = 1.0 - 2.0 * P_R1

    matvec_gather_T(X, full_N, row_idx, N, col_idx, d, y_dp_vec, tmp_y_dp_vec)

    for i in range(d):
        tmp_y_dp = tmp_y_dp_vec[i]
        gj = grad_w[i]
        dS_R_w_i = dS_R_w[i]
        dP_L1_w_i = (tmp_y_dp * S_L - p_y_sum * gj) / denom_L
        dP_R1_w_i = (-tmp_y_dp * S_R - p_1_y_sum * dS_R_w_i) / denom_R
        dP_L1_w[i] = dP_L1_w_i
        dP_R1_w[i] = dP_R1_w_i
        dImp_L = factor_L * dP_L1_w_i
        dImp_R = factor_R * dP_R1_w_i
        grad_w[i] = (gj * impurity_L
                    + S_L * dImp_L
                    + dS_R_w_i * impurity_R
                    + S_R * dImp_R)

    return loss


cdef double fun_and_grad_linear_reg_nogil(
    const double* X, Py_ssize_t full_N,
    const int* row_idx, const int* col_idx,
    const double* y,
    const double* sample_weight,
    double* w,
    const Py_ssize_t N,
    const Py_ssize_t d,
    const double total_weight,
    double* pred,
    double* weighted_residuals,
    double* grad_w) noexcept nogil:

    cdef Py_ssize_t i
    cdef double loss = 0.0
    cdef double residual, sw_scaled

    matvec_gather_N(X, full_N, row_idx, N, col_idx, d, w, pred)

    for i in range(N):
        residual = y[i] - pred[i]
        sw_scaled = sample_weight[i] / total_weight
        weighted_residuals[i] = residual * sw_scaled
        loss += 0.5 * sw_scaled * residual * residual

    matvec_gather_T(X, full_N, row_idx, N, col_idx, d, weighted_residuals, grad_w)
    for i in range(d):
        grad_w[i] = -grad_w[i]

    return loss


cdef double fun_and_grad_multiclass_nogil(
    const double* X, Py_ssize_t full_N,
    const int* row_idx, const int* col_idx,
    const double* y,
    const double* sample_weight,
    double* w,
    const Py_ssize_t N,
    const Py_ssize_t d,
    const Py_ssize_t n_classes,
    const double gamma,
    const double eps,
    const double total_weight,
    double* z,
    double* p,
    double* dp_dz,
    double* grad_w,
    double* class_counts_L,
    double* class_counts_R,
    double* tmp_dp_vec,
    double* dS_R_w,
    double* P_k_L,
    double* P_k_R) noexcept nogil:

    cdef Py_ssize_t i, j, k
    cdef double S_L = 0.0, S_R = 0.0
    cdef double impurity_L = 0.0, impurity_R = 0.0
    cdef double loss = 0.0
    cdef double sw_i, tmp, one_minus_tmp
    cdef double denom_L, denom_R
    cdef double tmp_dp, gj, dS_R_w_i
    cdef int yi

    matvec_gather_N(X, full_N, row_idx, N, col_idx, d, w, z)

    for k in range(n_classes):
        class_counts_L[k] = 0.0
        class_counts_R[k] = 0.0

    for i in range(N):
        sw_i = sample_weight[i]
        yi = <int>y[i]

        tmp = 1.0 / (1.0 + exp(-z[i]))
        one_minus_tmp = 1.0 - tmp
        p[i] = tmp

        class_counts_L[yi] += tmp * sw_i
        class_counts_R[yi] += one_minus_tmp * sw_i
        S_L += tmp * sw_i
        S_R += one_minus_tmp * sw_i

        dp_dz[i] = gamma * tmp * one_minus_tmp * sw_i

    S_L += eps
    S_R += eps

    denom_L = S_L * S_L
    denom_R = S_R * S_R

    for k in range(n_classes):
        P_k_L[k] = class_counts_L[k] / S_L
        P_k_R[k] = class_counts_R[k] / S_R
        impurity_L += P_k_L[k] * (1.0 - P_k_L[k])
        impurity_R += P_k_R[k] * (1.0 - P_k_R[k])

    loss = S_L * impurity_L + S_R * impurity_R

    matvec_gather_T(X, full_N, row_idx, N, col_idx, d, dp_dz, grad_w)

    for i in range(d):
        dS_R_w[i] = -grad_w[i]

    for k in range(n_classes):
        for i in range(d):
            gj = grad_w[i]
            dS_R_w_i = dS_R_w[i]

            tmp_dp = (class_counts_L[k] * gj / denom_L) * (1.0 - 2.0 * P_k_L[k])
            tmp_dp += (class_counts_R[k] * dS_R_w_i / denom_R) * (1.0 - 2.0 * P_k_R[k])

            grad_w[i] += (gj * impurity_L +
                         S_L * tmp_dp +
                         dS_R_w_i * impurity_R +
                         S_R * (-tmp_dp))

    return loss


cdef double fun_and_grad_reg_nogil(
    const double* X, Py_ssize_t full_N,
    const int* row_idx, const int* col_idx,
    const double* y,
    const double* sample_weight,
    double* w,
    const Py_ssize_t N,
    const Py_ssize_t d,
    const double gamma,
    const double eps,
    const double total_weight,
    double* z,
    double* p,
    double* dp_dz,
    double* grad_w,
    double* temp_vec1,
    double* temp_vec2,
    double* temp_grad) noexcept nogil:

    cdef Py_ssize_t i, j
    cdef double loss = 0.0
    cdef double S_L = 0.0, M_L = 0.0
    cdef double S_R = 0.0, M_R = 0.0
    cdef double mL, mR
    cdef double p_val, dp_val, sw_i
    cdef double diffL, diffR
    cdef double dS_L_w, dM_L_w, d_mL_w, d_mR_w

    for i in range(d):
        grad_w[i] = 0.0
        temp_grad[i] = 0.0

    matvec_gather_N(X, full_N, row_idx, N, col_idx, d, w, z)

    for i in range(N):
        sw_i = sample_weight[i]
        p_val = sigmoid(gamma * z[i])
        p[i] = p_val

        dp_val = gamma * p_val * (1.0 - p_val)
        dp_dz[i] = dp_val

        S_L += sw_i * p_val
        M_L += sw_i * p_val * y[i]

        S_R += sw_i * (1.0 - p_val)
        M_R += sw_i * (1.0 - p_val) * y[i]

    S_L += eps
    S_R += eps

    mL = M_L / S_L
    mR = M_R / S_R

    for i in range(N):
        sw_i = sample_weight[i]
        diffL = y[i] - mL
        diffR = y[i] - mR

        loss += sw_i * p[i] * (diffL * diffL)
        loss += sw_i * (1.0 - p[i]) * (diffR * diffR)
        temp_vec1[i] = dp_dz[i] * sw_i * (diffL * diffL - diffR * diffR)

    loss /= total_weight

    matvec_gather_T(X, full_N, row_idx, N, col_idx, d, temp_vec1, grad_w)

    for i in range(d):
        temp_grad[i] = 0.0

    for i in range(N):
        temp_vec2[i] = dp_dz[i] * sample_weight[i]

    matvec_gather_T(X, full_N, row_idx, N, col_idx, d, temp_vec2, temp_grad)

    for j in range(d):
        dS_L_w = temp_grad[j]
        dM_L_w = temp_grad[j]
        d_mL_w = (S_L * dM_L_w - M_L * dS_L_w) / (S_L * S_L)
        d_mR_w = (S_R * (-dM_L_w) - M_R * (-dS_L_w)) / (S_R * S_R)

        for i in range(N):
            sw_i = sample_weight[i]
            temp_vec1[i] = -2.0 * sw_i * (
                p[i] * (y[i] - mL) * d_mL_w +
                (1.0 - p[i]) * (y[i] - mR) * d_mR_w
            )

        for i in range(d):
            temp_vec2[i] = 0.0
        matvec_gather_T(X, full_N, row_idx, N, col_idx, d, temp_vec1, temp_vec2)
        for i in range(d):
            grad_w[i] += temp_vec2[i]

    for i in range(d):
        grad_w[i] /= total_weight

    return loss


cdef double fun_and_grad_binary_linear_nogil(
    const double current_class,
    const double* X, Py_ssize_t full_N,
    const int* row_idx, const int* col_idx,
    const double* y,
    const double* sample_weight,
    double* w,
    const Py_ssize_t N,
    const Py_ssize_t d,
    const double total_weight,
    double* pred,
    double* weighted_residuals,
    double* grad_w) noexcept nogil:

    cdef Py_ssize_t i
    cdef double loss = 0.0
    cdef double prob, sw_scaled, y_i

    matvec_gather_N(X, full_N, row_idx, N, col_idx, d, w, pred)

    for i in range(N):
        y_i = get_y(y[i], current_class)
        prob = sigmoid(pred[i])
        sw_scaled = sample_weight[i] / total_weight

        if y_i:
            loss -= sw_scaled * log(prob + 1e-15)
        else:
            loss -= sw_scaled * log(1.0 - prob + 1e-15)

        weighted_residuals[i] = sw_scaled * (prob - y_i)

    matvec_gather_T(X, full_N, row_idx, N, col_idx, d, weighted_residuals, grad_w)

    return loss


# ---------------------------------------------------------------------------
# Unified loss/grad dispatcher
# ---------------------------------------------------------------------------
cdef double eval_loss_grad(
    const bint task_,
    const int n_classes,
    const double current_class,
    const bint linear,
    const double* X, Py_ssize_t full_N,
    const int* row_idx, const int* col_idx,
    const double* y,
    const double* sample_weight,
    double* w,
    const Py_ssize_t N,
    const Py_ssize_t d,
    const double gamma,
    const double eps,
    const double total_weight,
    double* grad_w,
    double* buf_z, double* buf_p, double* buf_dp_dz,
    double* buf1, double* buf2, double* buf3,
    double* buf4, double* buf5,
    double* buf6, double* buf7) noexcept nogil:

    if task_ == 0:
        if linear:
            return fun_and_grad_binary_linear_nogil(
                current_class, X, full_N, row_idx, col_idx,
                y, sample_weight, w,
                N, d, total_weight, buf1, buf2, grad_w)
        else:
            if n_classes > 2:
                return fun_and_grad_multiclass_nogil(
                    X, full_N, row_idx, col_idx,
                    y, sample_weight, w,
                    N, d, n_classes, gamma, eps, total_weight,
                    buf_z, buf_p, buf_dp_dz, grad_w,
                    buf1, buf2, buf3, buf4, buf5, buf6)
            else:
                return fun_and_grad_nogil(
                    X, full_N, row_idx, col_idx,
                    y, sample_weight, w,
                    N, d, gamma, eps, total_weight,
                    buf_z, buf_p, buf_dp_dz, grad_w,
                    buf1, buf2, buf3, buf4, buf5)
    else:
        if linear:
            return fun_and_grad_linear_reg_nogil(
                X, full_N, row_idx, col_idx,
                y, sample_weight, w,
                N, d, total_weight, buf1, buf2, grad_w)
        else:
            return fun_and_grad_reg_nogil(
                X, full_N, row_idx, col_idx,
                y, sample_weight, w,
                N, d, gamma, eps, total_weight,
                buf_z, buf_p, buf_dp_dz, grad_w,
                buf1, buf2, buf3)


# ---------------------------------------------------------------------------
# Pure Cython L-BFGS  (nogil, no scipy)
# ---------------------------------------------------------------------------
cdef inline double dot(const double* a, const double* b,
                       Py_ssize_t n) noexcept nogil:
    cdef Py_ssize_t i
    cdef double s = 0.0
    for i in range(n):
        s += a[i] * b[i]
    return s

cdef inline double inf_norm(const double* a, Py_ssize_t n) noexcept nogil:
    cdef Py_ssize_t i
    cdef double mx = 0.0, v
    for i in range(n):
        v = fabs(a[i])
        if v > mx:
            mx = v
    return mx


cdef double lbfgs_minimize_nogil(
    const bint task_,
    const int n_classes,
    const double current_class,
    const bint linear,
    double* x,
    const double* X_data, Py_ssize_t full_N,
    const int* row_idx, const int* col_idx,
    const double* y_data,
    const double* sample_weight,
    const double total_weight,
    const Py_ssize_t N,
    const Py_ssize_t d,
    const double gamma,
    const double eps,
    const int m,
    const int maxiter,
    const double relative_change,
    const double pgtol,
    const int maxls,
    double* grad,
    double* grad_new,
    double* direction,
    double* x_new,
    double* s_hist,
    double* y_hist,
    double* rho,
    double* alpha_buf,
    double* buf_z,
    double* buf_p,
    double* buf_dp_dz,
    double* buf1, double* buf2, double* buf3,
    double* buf4, double* buf5,
    double* buf6, double* buf7,
) noexcept nogil:
    cdef Py_ssize_t i, j, k, idx
    cdef int n_stored = 0
    cdef int head = 0
    cdef double f_val, f_new, f_old
    cdef double step, dg, ys, yy, beta_val, gamma_k
    cdef double c1 = 1e-4
    cdef int ls
    cdef bint ls_ok

    f_val = eval_loss_grad(
        task_, n_classes, current_class, linear,
        X_data, full_N, row_idx, col_idx,
        y_data, sample_weight, x,
        N, d, gamma, eps, total_weight,
        grad,
        buf_z, buf_p, buf_dp_dz,
        buf1, buf2, buf3, buf4, buf5, buf6, buf7)

    if inf_norm(grad, d) < pgtol:
        return f_val

    f_old = INFINITY

    for k in range(maxiter):
        # Two-loop recursion
        for i in range(d):
            direction[i] = -grad[i]

        j = (head - 1 + m) % m
        for i in range(n_stored):
            alpha_buf[j] = rho[j] * dot(&s_hist[j * d], direction, d)
            for idx in range(d):
                direction[idx] -= alpha_buf[j] * y_hist[j * d + idx]
            j = (j - 1 + m) % m

        if n_stored > 0:
            j = (head - 1 + m) % m
            yy = dot(&y_hist[j * d], &y_hist[j * d], d)
            if yy > 0:
                gamma_k = dot(&s_hist[j * d], &y_hist[j * d], d) / yy
            else:
                gamma_k = 1.0
            for i in range(d):
                direction[i] *= gamma_k

        j = (head - n_stored + m) % m
        for i in range(n_stored):
            beta_val = rho[j] * dot(&y_hist[j * d], direction, d)
            for idx in range(d):
                direction[idx] += (alpha_buf[j] - beta_val) * s_hist[j * d + idx]
            j = (j + 1) % m

        # Backtracking line search (Armijo)
        dg = dot(grad, direction, d)
        if dg >= 0:
            for i in range(d):
                direction[i] = -grad[i]
            dg = dot(grad, direction, d)

        step = 1.0
        ls_ok = 0
        for ls in range(maxls):
            for i in range(d):
                x_new[i] = x[i] + step * direction[i]

            f_new = eval_loss_grad(
                task_, n_classes, current_class, linear,
                X_data, full_N, row_idx, col_idx,
                y_data, sample_weight, x_new,
                N, d, gamma, eps, total_weight,
                grad_new,
                buf_z, buf_p, buf_dp_dz,
                buf1, buf2, buf3, buf4, buf5, buf6, buf7)

            if f_new <= f_val + c1 * step * dg:
                ls_ok = 1
                break
            step *= 0.5

        if not ls_ok:
            n_stored = 0
            head = 0
            continue

        # Update history
        ys = 0.0
        for i in range(d):
            s_hist[head * d + i] = x_new[i] - x[i]
            y_hist[head * d + i] = grad_new[i] - grad[i]
            ys += s_hist[head * d + i] * y_hist[head * d + i]

        if ys > 1e-30:
            rho[head] = 1.0 / ys
            head = (head + 1) % m
            if n_stored < m:
                n_stored += 1

        for i in range(d):
            x[i] = x_new[i]
            grad[i] = grad_new[i]

        f_val = f_new

        if inf_norm(grad, d) < pgtol:
            break
        if f_old != INFINITY and f_old > 0:
            if (f_old - f_val) / f_old <= relative_change:
                break
        f_old = f_val

    return f_val


# ---------------------------------------------------------------------------
# Per-thread workspace
# ---------------------------------------------------------------------------
cdef struct ThreadWorkspace:
    double* grad
    double* grad_new
    double* direction
    double* x_new
    double* s_hist
    double* y_hist
    double* rho
    double* alpha_buf
    double* buf_z
    double* buf_p
    double* buf_dp_dz
    double* buf1
    double* buf2
    double* buf3
    double* buf4
    double* buf5
    double* buf6
    double* buf7
    double* x_work


cdef bint _check_ws(ThreadWorkspace* w, bint task_, bint linear,
                    int n_classes) noexcept nogil:
    if (w.grad == NULL or w.grad_new == NULL or w.direction == NULL or
        w.x_new == NULL or w.s_hist == NULL or w.y_hist == NULL or
        w.rho == NULL or w.alpha_buf == NULL or w.x_work == NULL or
        w.buf_z == NULL or w.buf_p == NULL or w.buf_dp_dz == NULL):
        return False
    if task_ == 0:
        if linear:
            return w.buf1 != NULL and w.buf2 != NULL
        elif n_classes > 2:
            return (w.buf1 != NULL and w.buf2 != NULL and w.buf3 != NULL and
                    w.buf4 != NULL and w.buf5 != NULL and w.buf6 != NULL)
        else:
            return (w.buf1 != NULL and w.buf2 != NULL and w.buf3 != NULL and
                    w.buf4 != NULL and w.buf5 != NULL)
    else:
        if linear:
            return w.buf1 != NULL and w.buf2 != NULL
        else:
            return w.buf1 != NULL and w.buf2 != NULL and w.buf3 != NULL


cdef ThreadWorkspace* alloc_workspaces(
    int n_threads, Py_ssize_t N, Py_ssize_t d,
    int lbfgs_m, int n_classes, bint task_, bint linear) noexcept:
    cdef ThreadWorkspace* ws = <ThreadWorkspace*>calloc(
        n_threads, sizeof(ThreadWorkspace))
    if ws == NULL:
        return NULL
    cdef int t

    for t in range(n_threads):
        ws[t].grad      = <double*>malloc(d * sizeof(double))
        ws[t].grad_new  = <double*>malloc(d * sizeof(double))
        ws[t].direction = <double*>malloc(d * sizeof(double))
        ws[t].x_new     = <double*>malloc(d * sizeof(double))
        ws[t].s_hist    = <double*>calloc(lbfgs_m * d, sizeof(double))
        ws[t].y_hist    = <double*>calloc(lbfgs_m * d, sizeof(double))
        ws[t].rho       = <double*>calloc(lbfgs_m, sizeof(double))
        ws[t].alpha_buf = <double*>malloc(lbfgs_m * sizeof(double))
        ws[t].x_work    = <double*>malloc(d * sizeof(double))

        ws[t].buf_z     = <double*>malloc(N * sizeof(double))
        ws[t].buf_p     = <double*>malloc(N * sizeof(double))
        ws[t].buf_dp_dz = <double*>malloc(N * sizeof(double))

        if task_ == 0:
            if linear:
                ws[t].buf1 = <double*>malloc(N * sizeof(double))
                ws[t].buf2 = <double*>malloc(N * sizeof(double))
                ws[t].buf3 = NULL
                ws[t].buf4 = NULL
                ws[t].buf5 = NULL
                ws[t].buf6 = NULL
                ws[t].buf7 = NULL
            elif n_classes > 2:
                ws[t].buf1 = <double*>malloc(n_classes * sizeof(double))
                ws[t].buf2 = <double*>malloc(n_classes * sizeof(double))
                ws[t].buf3 = <double*>malloc(d * sizeof(double))
                ws[t].buf4 = <double*>malloc(d * sizeof(double))
                ws[t].buf5 = <double*>calloc(n_classes, sizeof(double))
                ws[t].buf6 = <double*>calloc(n_classes, sizeof(double))
                ws[t].buf7 = NULL
            else:
                ws[t].buf1 = <double*>malloc(d * sizeof(double))
                ws[t].buf2 = <double*>malloc(d * sizeof(double))
                ws[t].buf3 = <double*>malloc(d * sizeof(double))
                ws[t].buf4 = <double*>malloc(d * sizeof(double))
                ws[t].buf5 = <double*>malloc(N * sizeof(double))
                ws[t].buf6 = NULL
                ws[t].buf7 = NULL
        else:
            if linear:
                ws[t].buf1 = <double*>malloc(N * sizeof(double))
                ws[t].buf2 = <double*>malloc(N * sizeof(double))
                ws[t].buf3 = NULL
                ws[t].buf4 = NULL
                ws[t].buf5 = NULL
                ws[t].buf6 = NULL
                ws[t].buf7 = NULL
            else:
                ws[t].buf1 = <double*>malloc(N * sizeof(double))
                ws[t].buf2 = <double*>malloc(N * sizeof(double))
                ws[t].buf3 = <double*>malloc(d * sizeof(double))
                ws[t].buf4 = NULL
                ws[t].buf5 = NULL
                ws[t].buf6 = NULL
                ws[t].buf7 = NULL

        if not _check_ws(&ws[t], task_, linear, n_classes):
            free_workspaces(ws, n_threads)
            return NULL

    return ws


cdef void free_workspaces(ThreadWorkspace* ws, int n_threads) noexcept:
    cdef int t
    for t in range(n_threads):
        free(ws[t].grad)
        free(ws[t].grad_new)
        free(ws[t].direction)
        free(ws[t].x_new)
        free(ws[t].s_hist)
        free(ws[t].y_hist)
        free(ws[t].rho)
        free(ws[t].alpha_buf)
        free(ws[t].x_work)
        free(ws[t].buf_z)
        free(ws[t].buf_p)
        free(ws[t].buf_dp_dz)
        free(ws[t].buf1)
        free(ws[t].buf2)
        free(ws[t].buf3)
        free(ws[t].buf4)
        free(ws[t].buf5)
        free(ws[t].buf6)
        free(ws[t].buf7)
    free(ws)


# ---------------------------------------------------------------------------
# C-level pair generation  (no Python combinations)
# ---------------------------------------------------------------------------
cdef Py_ssize_t count_pairs(Py_ssize_t n, int n_pair) noexcept nogil:
    """C(n, n_pair) for n_pair = 2 or 3."""
    if n_pair == 2:
        return n * (n - 1) // 2
    elif n_pair == 3:
        return n * (n - 1) * (n - 2) // 6
    return 0

cdef void generate_pairs_2(const int* features, Py_ssize_t n_feat,
                           int* out) noexcept nogil:
    cdef Py_ssize_t idx = 0, i, j
    for i in range(n_feat):
        for j in range(i + 1, n_feat):
            out[idx * 2]     = features[i]
            out[idx * 2 + 1] = features[j]
            idx += 1

cdef void generate_pairs_3(const int* features, Py_ssize_t n_feat,
                           int* out) noexcept nogil:
    cdef Py_ssize_t idx = 0, i, j, k
    for i in range(n_feat):
        for j in range(i + 1, n_feat):
            for k in range(j + 1, n_feat):
                out[idx * 3]     = features[i]
                out[idx * 3 + 1] = features[j]
                out[idx * 3 + 2] = features[k]
                idx += 1


# ---------------------------------------------------------------------------
# analyze()  —  parallel oblique split search
# ---------------------------------------------------------------------------
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
                const bint* is_categorical,
                object rng,
                const double gamma,
                const int maxiter,
                const double relative_change,
              ) noexcept:

    # --- Subsample y and sample_weight (small); keep X global ---
    cdef np.ndarray[int, ndim=1] sample_dx = np.frombuffer(
        <bytes>(<char*>sample_indices)[:n_samples * sizeof(int)], dtype=np.int32)
    cdef np.ndarray[double, ndim=1] y_sub = y[sample_dx].copy()
    cdef np.ndarray[double, ndim=1] sw_sub = sample_weight[sample_dx].copy()

    cdef double sum_sample_weight = sw_sub.sum()
    cdef Py_ssize_t N = n_samples
    cdef Py_ssize_t d = X.shape[1]
    cdef Py_ssize_t full_N = X.shape[0]
    cdef Py_ssize_t i

    cdef int* best_pair = NULL
    cdef double* best_x = NULL
    cdef double best_fx = INFINITY

    # --- Build usable-feature list in C ---
    cdef int* usable_features = <int*>malloc(d * sizeof(int))
    if usable_features == NULL:
        return NULL, NULL
    cdef Py_ssize_t n_usable = 0

    if is_categorical:
        for i in range(d):
            if not is_categorical[i]:
                usable_features[n_usable] = <int>i
                n_usable += 1
    else:
        for i in range(d):
            usable_features[i] = <int>i
        n_usable = d

    if n_usable < n_pair:
        free(usable_features)
        return best_x, best_pair

    cdef Py_ssize_t n_feature_pairs = count_pairs(n_usable, n_pair)
    if n_feature_pairs == 0:
        free(usable_features)
        return best_x, best_pair

    # --- Generate pair indices in C ---
    cdef int* pair_indices = <int*>malloc(n_feature_pairs * n_pair * sizeof(int))
    if pair_indices == NULL:
        free(usable_features)
        return NULL, NULL

    if n_pair == 2:
        generate_pairs_2(usable_features, n_usable, pair_indices)
    elif n_pair == 3:
        generate_pairs_3(usable_features, n_usable, pair_indices)
    free(usable_features)

    best_pair = <int*>malloc(n_pair * sizeof(int))
    best_x = <double*>malloc(n_pair * sizeof(double))
    if best_pair == NULL or best_x == NULL:
        free(best_pair)
        free(best_x)
        free(pair_indices)
        return NULL, NULL

    # --- Ensure X is Fortran-order for gathered access ---
    if not X.flags['F_CONTIGUOUS']:
        X = np.asfortranarray(X)
    cdef double* X_ptr = <double*>np.PyArray_DATA(X)

    # --- Pre-generate all random x0 values ---
    cdef np.ndarray[double, ndim=2] all_x0 = rng.standard_normal(
        (n_feature_pairs, n_pair)).astype(np.float64)
    cdef double* all_x0_ptr = <double*>np.PyArray_DATA(all_x0)

    cdef double* y_ptr = <double*>np.PyArray_DATA(y_sub)
    cdef double* sw_ptr = <double*>np.PyArray_DATA(sw_sub)

    # --- Allocate per-thread workspaces ---
    cdef int n_threads = openmp.omp_get_max_threads()
    cdef int lbfgs_m = 10
    cdef ThreadWorkspace* workspaces = alloc_workspaces(
        n_threads, N, n_pair, lbfgs_m, n_classes, task, linear)

    if workspaces == NULL:
        free(pair_indices)
        free(best_pair)
        free(best_x)
        return NULL, NULL

    # --- Results arrays ---
    cdef double* pair_losses = <double*>malloc(n_feature_pairs * sizeof(double))
    cdef double* pair_weights = <double*>malloc(
        n_feature_pairs * n_pair * sizeof(double))

    if pair_losses == NULL or pair_weights == NULL:
        free(pair_indices)
        free(pair_losses)
        free(pair_weights)
        free(best_pair)
        free(best_x)
        free_workspaces(workspaces, n_threads)
        return NULL, NULL

    for i in range(n_feature_pairs):
        pair_losses[i] = INFINITY

    # --- Parallel feature-pair optimization ---
    # Linear multiclass: class loop runs inside per-pair, so pair columns
    # are gathered once and warm-start carries across classes.
    cdef Py_ssize_t pair_idx
    cdef int tid, ci
    cdef double f_val
    cdef ThreadWorkspace* w
    cdef int multi_range = 1
    if linear and n_classes > 2:
        multi_range = n_classes

    for pair_idx in prange(n_feature_pairs, nogil=True, schedule="dynamic"):
        tid = openmp.omp_get_thread_num()
        w = &workspaces[tid]

        # Copy x0
        memcpy(w.x_work, &all_x0_ptr[pair_idx * n_pair],
               n_pair * sizeof(double))

        # Run optimization — class loop inside for warm-start
        f_val = INFINITY
        for ci in range(multi_range):
            memset(w.s_hist, 0, lbfgs_m * n_pair * sizeof(double))
            memset(w.y_hist, 0, lbfgs_m * n_pair * sizeof(double))
            memset(w.rho, 0, lbfgs_m * sizeof(double))

            f_val = lbfgs_minimize_nogil(
                task, n_classes, <double>ci, linear,
                w.x_work,
                X_ptr, full_N,
                sample_indices, &pair_indices[pair_idx * n_pair],
                y_ptr, sw_ptr, sum_sample_weight,
                N, n_pair,
                gamma, 1e-6, lbfgs_m, maxiter, relative_change,
                1e-5, 20,
                w.grad, w.grad_new, w.direction, w.x_new,
                w.s_hist, w.y_hist, w.rho, w.alpha_buf,
                w.buf_z, w.buf_p, w.buf_dp_dz,
                w.buf1, w.buf2, w.buf3,
                w.buf4, w.buf5, w.buf6, w.buf7)

        pair_losses[pair_idx] = f_val
        memcpy(&pair_weights[pair_idx * n_pair], w.x_work,
               n_pair * sizeof(double))

    # --- Find best result ---
    cdef Py_ssize_t best_job = 0
    for i in range(n_feature_pairs):
        if pair_losses[i] < best_fx:
            best_fx = pair_losses[i]
            best_job = i

    # Normalize weights
    cdef double max_abs = 0.0
    for i in range(n_pair):
        if fabs(pair_weights[best_job * n_pair + i]) > max_abs:
            max_abs = fabs(pair_weights[best_job * n_pair + i])
    if max_abs == 0.0:
        max_abs = 1.0

    for i in range(n_pair):
        best_pair[i] = pair_indices[best_job * n_pair + i]
        best_x[i] = pair_weights[best_job * n_pair + i] / max_abs

    # Compute best_values and sort  (use gathered dot on global X)
    cdef double bv
    with nogil:
        for i in range(N):
            bv = 0.0
            for ci in range(n_pair):
                bv += X_ptr[sample_indices[i] + <Py_ssize_t>best_pair[ci] * full_N] * best_x[ci]
            sort_buffer[i].value = bv
            sort_buffer[i].index = sample_indices[i]
        sort_pointer_array(sort_buffer, N)

    # Cleanup
    free(pair_indices)
    free(pair_losses)
    free(pair_weights)
    free_workspaces(workspaces, n_threads)

    return best_x, best_pair
