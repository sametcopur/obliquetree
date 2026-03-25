from libc.math cimport INFINITY, exp, log, fabs, sqrt
from libc.stdlib cimport malloc, free, calloc
from libc.string cimport memcpy, memset

import numpy as np
cimport numpy as np

from cython.parallel cimport prange
cimport openmp

from .utils cimport sort_pointer_array

DEF MIN_SCREEN_PAIRS = 64
DEF MIN_SCREEN_SURVIVORS = 4
DEF SCREEN_KEEP_DIVISOR = 5
DEF MAX_SCREEN_SAMPLES = 2048


# ---------------------------------------------------------------------------
# Helper inlines
# ---------------------------------------------------------------------------
cdef inline double get_y(double y, double current_class) noexcept nogil:
    return 1.0 if y == current_class else 0.0

cdef inline double sigmoid(const double x) noexcept nogil:
    return 1.0 / (1.0 + exp(-x))

# ---------------------------------------------------------------------------
# Contiguous mat-vec  (Fortran col-major, lda = N)
# Unrolled for d=2/3, generic fallback otherwise.
# ---------------------------------------------------------------------------
cdef inline void matvec_N(const double* X, const double* w,
                          double* out, Py_ssize_t N, Py_ssize_t d,
                          Py_ssize_t lda) noexcept nogil:
    cdef Py_ssize_t i, j
    cdef double w0, w1, w2
    cdef const double* c0
    cdef const double* c1
    cdef const double* c2
    if d == 2:
        w0 = w[0]; w1 = w[1]
        c0 = X; c1 = X + lda
        for i in range(N):
            out[i] = c0[i] * w0 + c1[i] * w1
    elif d == 3:
        w0 = w[0]; w1 = w[1]; w2 = w[2]
        c0 = X; c1 = X + lda; c2 = X + 2 * lda
        for i in range(N):
            out[i] = c0[i] * w0 + c1[i] * w1 + c2[i] * w2
    else:
        for i in range(N):
            out[i] = 0.0
        for j in range(d):
            for i in range(N):
                out[i] += X[i + j * lda] * w[j]

cdef inline void matvec_T(const double* X, const double* v,
                          double* out, Py_ssize_t N, Py_ssize_t d,
                          Py_ssize_t lda) noexcept nogil:
    cdef Py_ssize_t i, j
    cdef double s0, s1, s2
    cdef const double* c0
    cdef const double* c1
    cdef const double* c2
    if d == 2:
        s0 = 0.0; s1 = 0.0
        c0 = X; c1 = X + lda
        for i in range(N):
            s0 += c0[i] * v[i]
            s1 += c1[i] * v[i]
        out[0] = s0; out[1] = s1
    elif d == 3:
        s0 = 0.0; s1 = 0.0; s2 = 0.0
        c0 = X; c1 = X + lda; c2 = X + 2 * lda
        for i in range(N):
            s0 += c0[i] * v[i]
            s1 += c1[i] * v[i]
            s2 += c2[i] * v[i]
        out[0] = s0; out[1] = s1; out[2] = s2
    else:
        for j in range(d):
            out[j] = 0.0
            for i in range(N):
                out[j] += X[i + j * lda] * v[i]

# ---------------------------------------------------------------------------
# Gather: fill contiguous X_pair from global X.  Unrolled for d=2/3.
# ---------------------------------------------------------------------------
cdef inline void gather_X_pair(
    const double* X_global, Py_ssize_t full_N,
    const int* row_idx, Py_ssize_t N,
    const int* col_idx, Py_ssize_t d,
    double* X_pair,
) noexcept nogil:
    cdef Py_ssize_t i, j
    cdef const double* src0
    cdef const double* src1
    cdef const double* src2
    if d == 2:
        src0 = X_global + <Py_ssize_t>col_idx[0] * full_N
        src1 = X_global + <Py_ssize_t>col_idx[1] * full_N
        for i in range(N):
            X_pair[i]     = src0[row_idx[i]]
            X_pair[i + N] = src1[row_idx[i]]
    elif d == 3:
        src0 = X_global + <Py_ssize_t>col_idx[0] * full_N
        src1 = X_global + <Py_ssize_t>col_idx[1] * full_N
        src2 = X_global + <Py_ssize_t>col_idx[2] * full_N
        for i in range(N):
            X_pair[i]         = src0[row_idx[i]]
            X_pair[i + N]     = src1[row_idx[i]]
            X_pair[i + 2 * N] = src2[row_idx[i]]
    else:
        for j in range(d):
            for i in range(N):
                X_pair[i + j * N] = X_global[row_idx[i] + <Py_ssize_t>col_idx[j] * full_N]


# ---------------------------------------------------------------------------
# Loss / gradient  (contiguous X_pair, fully nogil)
# ---------------------------------------------------------------------------
cdef double fun_and_grad_nogil(
    const double* X, const double* y, const double* sample_weight,
    double* w,
    const Py_ssize_t N, const Py_ssize_t d,
    const double gamma, const double eps, const double total_weight,
    double* z, double* p, double* dp_dz, double* grad_w,
    double* dP_R1_w, double* dP_L1_w, double* tmp_y_dp_vec,
    double* dS_R_w, double* y_dp_vec) noexcept nogil:

    cdef Py_ssize_t i, lda = N
    cdef double S_L, S_R, p_y_sum, p_1_y_sum, tmp
    cdef double P_L1, P_R1, impurity_L, impurity_R, loss
    cdef double factor_L, factor_R, dImp_L, dImp_R
    cdef double denom_L, denom_R, tmp_y_dp, gj
    cdef double one_minus_tmp, yi, sw_i
    cdef double dp_dz_i, dS_R_w_i, dP_L1_w_i, dP_R1_w_i

    matvec_N(X, w, z, N, d, lda)

    S_L = 0.0; p_y_sum = 0.0; p_1_y_sum = 0.0
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
    S_L += eps; S_R += eps

    P_L1 = p_y_sum / S_L
    P_R1 = p_1_y_sum / S_R
    impurity_L = P_L1 * (1.0 - P_L1)
    impurity_R = P_R1 * (1.0 - P_R1)
    loss = S_L * impurity_L + S_R * impurity_R

    matvec_T(X, dp_dz, grad_w, N, d, lda)
    for i in range(d):
        dS_R_w[i] = -grad_w[i]

    denom_L = S_L * S_L; denom_R = S_R * S_R
    factor_L = 1.0 - 2.0 * P_L1; factor_R = 1.0 - 2.0 * P_R1

    matvec_T(X, y_dp_vec, tmp_y_dp_vec, N, d, lda)
    for i in range(d):
        tmp_y_dp = tmp_y_dp_vec[i]
        gj = grad_w[i]
        dS_R_w_i = dS_R_w[i]
        dP_L1_w_i = (tmp_y_dp * S_L - p_y_sum * gj) / denom_L
        dP_R1_w_i = (-tmp_y_dp * S_R - p_1_y_sum * dS_R_w_i) / denom_R
        dImp_L = factor_L * dP_L1_w_i
        dImp_R = factor_R * dP_R1_w_i
        grad_w[i] = (gj * impurity_L + S_L * dImp_L
                    + dS_R_w_i * impurity_R + S_R * dImp_R)
    return loss


cdef double fun_and_grad_linear_reg_nogil(
    const double* X, const double* y, const double* sample_weight,
    double* w,
    const Py_ssize_t N, const Py_ssize_t d, const double total_weight,
    double* pred, double* weighted_residuals, double* grad_w) noexcept nogil:

    cdef Py_ssize_t i, lda = N
    cdef double loss = 0.0, residual, sw_scaled

    matvec_N(X, w, pred, N, d, lda)
    for i in range(N):
        residual = y[i] - pred[i]
        sw_scaled = sample_weight[i] / total_weight
        weighted_residuals[i] = residual * sw_scaled
        loss += 0.5 * sw_scaled * residual * residual

    matvec_T(X, weighted_residuals, grad_w, N, d, lda)
    for i in range(d):
        grad_w[i] = -grad_w[i]
    return loss


cdef double fun_and_grad_multiclass_nogil(
    const double* X, const double* y, const double* sample_weight,
    double* w,
    const Py_ssize_t N, const Py_ssize_t d, const Py_ssize_t n_classes,
    const double gamma, const double eps, const double total_weight,
    double* z, double* p, double* dp_dz, double* grad_w,
    double* class_counts_L, double* class_counts_R,
    double* tmp_dp_vec, double* dS_R_w,
    double* P_k_L, double* P_k_R) noexcept nogil:
    """
    Multiclass weighted Gini oblique split loss + gradient.

    L = S_L * Gini_L + S_R * Gini_R
    where Gini_X = sum_k P_X,k * (1 - P_X,k)

    The multiclass gradient collapses to:
      grad = X^T(dp_dz * 2 * (P_R[y] - P_L[y])) + dS_L * (Gini_R - Gini_L)
    where dS_L = X^T(dp_dz).
    """
    cdef Py_ssize_t i, k, j, lda = N
    cdef double S_L = 0.0, S_R = 0.0
    cdef double impurity_L = 0.0, impurity_R = 0.0, loss
    cdef double sw_i, tmp, one_minus_tmp
    cdef double scale_i
    cdef int yi

    matvec_N(X, w, z, N, d, lda)

    for k in range(n_classes):
        class_counts_L[k] = 0.0; class_counts_R[k] = 0.0
    for i in range(N):
        sw_i = sample_weight[i]; yi = <int>y[i]
        tmp = 1.0 / (1.0 + exp(-z[i]))
        one_minus_tmp = 1.0 - tmp
        p[i] = tmp
        class_counts_L[yi] += tmp * sw_i
        class_counts_R[yi] += one_minus_tmp * sw_i
        S_L += tmp * sw_i; S_R += one_minus_tmp * sw_i
        dp_dz[i] = gamma * tmp * one_minus_tmp * sw_i

    S_L += eps; S_R += eps
    for k in range(n_classes):
        P_k_L[k] = class_counts_L[k] / S_L
        P_k_R[k] = class_counts_R[k] / S_R
        impurity_L += P_k_L[k] * (1.0 - P_k_L[k])
        impurity_R += P_k_R[k] * (1.0 - P_k_R[k])
    loss = S_L * impurity_L + S_R * impurity_R

    # dS_L/dw = X^T(dp_dz)
    matvec_T(X, dp_dz, tmp_dp_vec, N, d, lda)
    # Build the per-sample class contribution into z and project once.
    for i in range(N):
        yi = <int>y[i]
        scale_i = 2.0 * (P_k_R[yi] - P_k_L[yi])
        z[i] = dp_dz[i] * scale_i
    matvec_T(X, z, dS_R_w, N, d, lda)
    for j in range(d):
        grad_w[j] = dS_R_w[j] + tmp_dp_vec[j] * (impurity_R - impurity_L)

    return loss


cdef double fun_and_grad_reg_nogil(
    const double* X, const double* y, const double* sample_weight,
    double* w,
    const Py_ssize_t N, const Py_ssize_t d,
    const double gamma, const double eps, const double total_weight,
    double* z, double* p, double* dp_dz, double* grad_w,
    double* temp_vec1, double* temp_vec2, double* temp_grad) noexcept nogil:

    cdef Py_ssize_t i, j, lda = N
    cdef double loss = 0.0
    cdef double S_L = 0.0, M_L = 0.0, S_R = 0.0, M_R = 0.0
    cdef double mL, mR, p_val, dp_val, sw_i, diffL, diffR
    cdef double dS_L_w, dM_L_w, d_mL_w, d_mR_w

    for i in range(d):
        grad_w[i] = 0.0; temp_grad[i] = 0.0

    matvec_N(X, w, z, N, d, lda)
    for i in range(N):
        sw_i = sample_weight[i]
        p_val = sigmoid(gamma * z[i]); p[i] = p_val
        dp_val = gamma * p_val * (1.0 - p_val); dp_dz[i] = dp_val
        S_L += sw_i * p_val; M_L += sw_i * p_val * y[i]
        S_R += sw_i * (1.0 - p_val); M_R += sw_i * (1.0 - p_val) * y[i]

    S_L += eps; S_R += eps
    mL = M_L / S_L; mR = M_R / S_R

    for i in range(N):
        sw_i = sample_weight[i]
        diffL = y[i] - mL; diffR = y[i] - mR
        loss += sw_i * p[i] * (diffL * diffL)
        loss += sw_i * (1.0 - p[i]) * (diffR * diffR)
        temp_vec1[i] = dp_dz[i] * sw_i * (diffL * diffL - diffR * diffR)
    loss /= total_weight

    matvec_T(X, temp_vec1, grad_w, N, d, lda)
    for i in range(d):
        temp_grad[i] = 0.0
    for i in range(N):
        temp_vec2[i] = dp_dz[i] * sample_weight[i]
    matvec_T(X, temp_vec2, temp_grad, N, d, lda)
    for i in range(N):
        temp_vec1[i] = temp_vec2[i] * y[i]
    matvec_T(X, temp_vec1, temp_vec2, N, d, lda)

    for j in range(d):
        dS_L_w = temp_grad[j]
        dM_L_w = temp_vec2[j]
        d_mL_w = (S_L * dM_L_w - M_L * dS_L_w) / (S_L * S_L)
        d_mR_w = (S_R * (-dM_L_w) - M_R * (-dS_L_w)) / (S_R * S_R)
        for i in range(N):
            sw_i = sample_weight[i]
            temp_vec1[i] = -2.0 * sw_i * (
                p[i] * (y[i] - mL) * d_mL_w +
                (1.0 - p[i]) * (y[i] - mR) * d_mR_w)
        for i in range(d):
            temp_vec2[i] = 0.0
        matvec_T(X, temp_vec1, temp_vec2, N, d, lda)
        for i in range(d):
            grad_w[i] += temp_vec2[i]

    for i in range(d):
        grad_w[i] /= total_weight
    return loss


cdef double fun_and_grad_binary_linear_nogil(
    const double current_class,
    const double* X, const double* y, const double* sample_weight,
    double* w,
    const Py_ssize_t N, const Py_ssize_t d, const double total_weight,
    double* pred, double* weighted_residuals, double* grad_w) noexcept nogil:

    cdef Py_ssize_t i, lda = N
    cdef double loss = 0.0, prob, sw_scaled, y_i

    matvec_N(X, w, pred, N, d, lda)
    for i in range(N):
        y_i = get_y(y[i], current_class)
        prob = sigmoid(pred[i])
        sw_scaled = sample_weight[i] / total_weight
        if y_i:
            loss -= sw_scaled * log(prob + 1e-15)
        else:
            loss -= sw_scaled * log(1.0 - prob + 1e-15)
        weighted_residuals[i] = sw_scaled * (prob - y_i)

    matvec_T(X, weighted_residuals, grad_w, N, d, lda)
    return loss


# ---------------------------------------------------------------------------
# Unified dispatcher
# ---------------------------------------------------------------------------
cdef double eval_loss_grad(
    const bint task_, const int n_classes,
    const double current_class, const bint linear,
    const double* X, const double* y, const double* sample_weight,
    double* w,
    const Py_ssize_t N, const Py_ssize_t d,
    const double gamma, const double eps, const double total_weight,
    double* grad_w,
    double* buf_z, double* buf_p, double* buf_dp_dz,
    double* buf1, double* buf2, double* buf3,
    double* buf4, double* buf5, double* buf6, double* buf7,
) noexcept nogil:
    if task_ == 0:
        if linear:
            return fun_and_grad_binary_linear_nogil(
                current_class, X, y, sample_weight, w,
                N, d, total_weight, buf1, buf2, grad_w)
        elif n_classes > 2:
            return fun_and_grad_multiclass_nogil(
                X, y, sample_weight, w,
                N, d, n_classes, gamma, eps, total_weight,
                buf_z, buf_p, buf_dp_dz, grad_w,
                buf1, buf2, buf3, buf4, buf5, buf6)
        else:
            return fun_and_grad_nogil(
                X, y, sample_weight, w,
                N, d, gamma, eps, total_weight,
                buf_z, buf_p, buf_dp_dz, grad_w,
                buf1, buf2, buf3, buf4, buf5)
    else:
        if linear:
            return fun_and_grad_linear_reg_nogil(
                X, y, sample_weight, w,
                N, d, total_weight, buf1, buf2, grad_w)
        else:
            return fun_and_grad_reg_nogil(
                X, y, sample_weight, w,
                N, d, gamma, eps, total_weight,
                buf_z, buf_p, buf_dp_dz, grad_w,
                buf1, buf2, buf3)


# ---------------------------------------------------------------------------
# L-BFGS (nogil)
# ---------------------------------------------------------------------------
cdef inline double dot(const double* a, const double* b,
                       Py_ssize_t n) noexcept nogil:
    if n == 2:
        return a[0] * b[0] + a[1] * b[1]
    elif n == 3:
        return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
    cdef Py_ssize_t i
    cdef double s = 0.0
    for i in range(n):
        s += a[i] * b[i]
    return s

cdef inline double inf_norm(const double* a, Py_ssize_t n) noexcept nogil:
    cdef Py_ssize_t i
    cdef double mx = 0.0, v
    if n == 2:
        mx = fabs(a[0])
        v = fabs(a[1])
        return mx if mx >= v else v
    elif n == 3:
        mx = fabs(a[0])
        v = fabs(a[1])
        if v > mx: mx = v
        v = fabs(a[2])
        if v > mx: mx = v
        return mx
    for i in range(n):
        v = fabs(a[i])
        if v > mx:
            mx = v
    return mx


cdef double proxy_screen_nogil(
    const bint task_, const int n_classes,
    const double current_class, const bint linear,
    double* x,
    const double* X_pair,
    const double* y_data, const double* sample_weight,
    const double total_weight,
    const Py_ssize_t N, const Py_ssize_t d,
    const double gamma, const double eps, const double pgtol,
    double* grad, double* grad_new, double* x_new,
    double* buf_z, double* buf_p, double* buf_dp_dz,
    double* buf1, double* buf2, double* buf3,
    double* buf4, double* buf5, double* buf6, double* buf7,
) noexcept nogil:
    cdef Py_ssize_t i
    cdef int attempt
    cdef double f_best, f_trial, grad_norm, step
    cdef double step_scale

    f_best = eval_loss_grad(
        task_, n_classes, current_class, linear,
        X_pair, y_data, sample_weight, x,
        N, d, gamma, eps, total_weight, grad,
        buf_z, buf_p, buf_dp_dz,
        buf1, buf2, buf3, buf4, buf5, buf6, buf7)

    grad_norm = inf_norm(grad, d)
    if grad_norm <= pgtol:
        return f_best

    step = 1.0 / grad_norm
    for attempt in range(2):
        step_scale = 1.0 if attempt == 0 else 0.25
        for i in range(d):
            x_new[i] = x[i] - step_scale * step * grad[i]

        f_trial = eval_loss_grad(
            task_, n_classes, current_class, linear,
            X_pair, y_data, sample_weight, x_new,
            N, d, gamma, eps, total_weight, grad_new,
            buf_z, buf_p, buf_dp_dz,
            buf1, buf2, buf3, buf4, buf5, buf6, buf7)

        if f_trial < f_best:
            memcpy(x, x_new, d * sizeof(double))
            memcpy(grad, grad_new, d * sizeof(double))
            f_best = f_trial
            break

    return f_best


cdef double lbfgs_minimize_nogil(
    const bint task_, const int n_classes,
    const double current_class, const bint linear,
    double* x,
    const double* X_pair,       # contiguous (N x d) Fortran
    const double* y_data, const double* sample_weight,
    const double total_weight,
    const Py_ssize_t N, const Py_ssize_t d,
    const double gamma, const double eps,
    const int m, const int maxiter,
    const double relative_change, const double pgtol, const int maxls,
    # workspaces
    double* grad, double* grad_new, double* direction, double* x_new,
    double* s_hist, double* y_hist, double* rho, double* alpha_buf,
    double* buf_z, double* buf_p, double* buf_dp_dz,
    double* buf1, double* buf2, double* buf3,
    double* buf4, double* buf5, double* buf6, double* buf7,
) noexcept nogil:
    cdef Py_ssize_t i, j, k, idx
    cdef int n_stored = 0, head = 0
    cdef double f_val, f_new, f_old
    cdef double step, dg, ys, yy, beta_val, gamma_k
    cdef double c1 = 1e-4
    cdef int ls
    cdef bint ls_ok

    f_val = eval_loss_grad(
        task_, n_classes, current_class, linear,
        X_pair, y_data, sample_weight, x,
        N, d, gamma, eps, total_weight, grad,
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

        # Line search (Armijo)
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
                X_pair, y_data, sample_weight, x_new,
                N, d, gamma, eps, total_weight, grad_new,
                buf_z, buf_p, buf_dp_dz,
                buf1, buf2, buf3, buf4, buf5, buf6, buf7)
            if f_new <= f_val + c1 * step * dg:
                ls_ok = 1
                break
            step *= 0.5

        if not ls_ok:
            n_stored = 0; head = 0
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
            x[i] = x_new[i]; grad[i] = grad_new[i]
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
    double* X_pair       # (N * n_pair) contiguous gather target
    double* pair_best_weights  # best weights for current pair across ci loop
    # thread-local best tracking
    double  best_loss
    double* best_weights  # (n_pair,)
    Py_ssize_t best_pi    # pair index of best


cdef bint _check_ws(ThreadWorkspace* w, bint task_, bint linear,
                    int n_classes) noexcept nogil:
    if (w.grad == NULL or w.grad_new == NULL or w.direction == NULL or
        w.x_new == NULL or w.s_hist == NULL or w.y_hist == NULL or
        w.rho == NULL or w.alpha_buf == NULL or w.x_work == NULL or
        w.X_pair == NULL or w.pair_best_weights == NULL or w.best_weights == NULL or
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


cdef ThreadWorkspace* alloc_workspaces_shared(
    int n_threads, Py_ssize_t N, Py_ssize_t d,
    int lbfgs_m, int n_classes, bint task_) noexcept:
    cdef ThreadWorkspace* ws = <ThreadWorkspace*>calloc(
        n_threads, sizeof(ThreadWorkspace))
    cdef Py_ssize_t buf12_size = N
    cdef Py_ssize_t buf5_size = N
    cdef Py_ssize_t d_size = d if d > 0 else 1
    cdef Py_ssize_t cls_size = n_classes if n_classes > 0 else 1
    cdef int t

    if ws == NULL:
        return NULL

    if d_size > buf12_size:
        buf12_size = d_size
    if cls_size > buf12_size:
        buf12_size = cls_size
    if cls_size > buf5_size:
        buf5_size = cls_size

    for t in range(n_threads):
        ws[t].grad      = <double*>malloc(d_size * sizeof(double))
        ws[t].grad_new  = <double*>malloc(d_size * sizeof(double))
        ws[t].direction = <double*>malloc(d_size * sizeof(double))
        ws[t].x_new     = <double*>malloc(d_size * sizeof(double))
        ws[t].s_hist    = <double*>calloc(lbfgs_m * d_size, sizeof(double))
        ws[t].y_hist    = <double*>calloc(lbfgs_m * d_size, sizeof(double))
        ws[t].rho       = <double*>calloc(lbfgs_m, sizeof(double))
        ws[t].alpha_buf = <double*>malloc(lbfgs_m * sizeof(double))
        ws[t].x_work    = <double*>malloc(d_size * sizeof(double))
        ws[t].X_pair    = <double*>malloc(N * d_size * sizeof(double))
        ws[t].pair_best_weights = <double*>malloc(d_size * sizeof(double))
        ws[t].best_weights = <double*>malloc(d_size * sizeof(double))
        ws[t].best_loss = INFINITY
        ws[t].best_pi   = 0

        ws[t].buf_z     = <double*>malloc(N * sizeof(double))
        ws[t].buf_p     = <double*>malloc(N * sizeof(double))
        ws[t].buf_dp_dz = <double*>malloc(N * sizeof(double))

        if task_ == 0:
            ws[t].buf1 = <double*>malloc(buf12_size * sizeof(double))
            ws[t].buf2 = <double*>malloc(buf12_size * sizeof(double))
            ws[t].buf3 = <double*>malloc(d_size * sizeof(double))
            ws[t].buf4 = <double*>malloc(d_size * sizeof(double))
            ws[t].buf5 = <double*>malloc(buf5_size * sizeof(double))
            ws[t].buf6 = <double*>malloc(cls_size * sizeof(double))
            ws[t].buf7 = NULL
        else:
            ws[t].buf1 = <double*>malloc(N * sizeof(double))
            ws[t].buf2 = <double*>malloc(N * sizeof(double))
            ws[t].buf3 = <double*>malloc(d_size * sizeof(double))
            ws[t].buf4 = NULL; ws[t].buf5 = NULL
            ws[t].buf6 = NULL; ws[t].buf7 = NULL

        if not _check_ws(&ws[t], task_, False, n_classes):
            free_workspaces(ws, n_threads)
            return NULL
    return ws


cdef void free_workspaces(ThreadWorkspace* ws, int n_threads) noexcept:
    cdef int t
    for t in range(n_threads):
        free(ws[t].grad);      free(ws[t].grad_new)
        free(ws[t].direction); free(ws[t].x_new)
        free(ws[t].s_hist);    free(ws[t].y_hist)
        free(ws[t].rho);       free(ws[t].alpha_buf)
        free(ws[t].x_work);    free(ws[t].X_pair)
        free(ws[t].pair_best_weights); free(ws[t].best_weights)
        free(ws[t].buf_z);     free(ws[t].buf_p);  free(ws[t].buf_dp_dz)
        free(ws[t].buf1);      free(ws[t].buf2);   free(ws[t].buf3)
        free(ws[t].buf4);      free(ws[t].buf5)
        free(ws[t].buf6);      free(ws[t].buf7)
    free(ws)


# ---------------------------------------------------------------------------
# C-level pair generation
# ---------------------------------------------------------------------------
cdef Py_ssize_t count_pairs(Py_ssize_t n, int n_pair) noexcept nogil:
    """C(n, n_pair) — combinations count."""
    if n < n_pair:
        return 0
    if n_pair == 2:
        return n * (n - 1) // 2
    elif n_pair == 3:
        return n * (n - 1) * (n - 2) // 6
    # General: C(n, k) via iterative multiply
    cdef Py_ssize_t result = 1
    cdef Py_ssize_t i
    cdef int k = n_pair
    if k > n - k:
        k = <int>(n - k)
    for i in range(k):
        result = result * (n - i) // (i + 1)
    return result

# ---------------------------------------------------------------------------
# Deterministic per-pair x0 init from seed + pair index  (no Python RNG)
# ---------------------------------------------------------------------------
cdef inline void init_x0_deterministic(
    double* x, Py_ssize_t d, unsigned long long seed,
    Py_ssize_t pair_idx) noexcept nogil:
    """Simple hash-based init producing values in roughly [-1, 1]."""
    cdef unsigned long long h
    cdef Py_ssize_t j
    for j in range(d):
        h = seed ^ (<unsigned long long>(pair_idx * 31 + j) * 6364136223846793005ULL + 1442695040888963407ULL)
        h = (h ^ (h >> 30)) * 0xbf58476d1ce4e5b9ULL
        h = (h ^ (h >> 27)) * 0x94d049bb133111ebULL
        h = h ^ (h >> 31)
        # Map to [-1, 1]
        x[j] = (<double>(h & <unsigned long long>0xFFFFFFFF) / 2147483648.0) - 1.0


cdef void generate_pairs_2(const int* features, Py_ssize_t n_feat,
                           int* out) noexcept nogil:
    cdef Py_ssize_t idx = 0, i, j
    for i in range(n_feat):
        for j in range(i + 1, n_feat):
            out[idx * 2] = features[i]
            out[idx * 2 + 1] = features[j]
            idx += 1

cdef void generate_pairs_3(const int* features, Py_ssize_t n_feat,
                           int* out) noexcept nogil:
    cdef Py_ssize_t idx = 0, i, j, k
    for i in range(n_feat):
        for j in range(i + 1, n_feat):
            for k in range(j + 1, n_feat):
                out[idx * 3] = features[i]
                out[idx * 3 + 1] = features[j]
                out[idx * 3 + 2] = features[k]
                idx += 1


cdef void generate_combinations(
    const int* features, Py_ssize_t n_feat, int n_pair,
    int* out,
) noexcept nogil:
    """General C(n_feat, n_pair) generator via stack-based iteration."""
    cdef int* stack = <int*>malloc(n_pair * sizeof(int))
    if stack == NULL:
        return
    cdef Py_ssize_t idx = 0
    cdef int depth = 0, start
    cdef Py_ssize_t j

    stack[0] = 0
    while depth >= 0:
        if depth == n_pair:
            for j in range(n_pair):
                out[idx * n_pair + j] = features[stack[j]]
            idx += 1
            depth -= 1
            if depth >= 0:
                stack[depth] += 1
        elif stack[depth] <= <int>n_feat - (n_pair - depth):
            if depth + 1 < n_pair:
                stack[depth + 1] = stack[depth] + 1
            depth += 1
        else:
            depth -= 1
            if depth >= 0:
                stack[depth] += 1
    free(stack)


# ---------------------------------------------------------------------------
# Feature scoring: weighted abs correlation with y  (nogil, O(N) per feature)
# ---------------------------------------------------------------------------
cdef void score_features(
    const bint task,
    const int n_classes,
    const double* X, Py_ssize_t full_N,
    const int* row_idx, Py_ssize_t N,
    const int* features, Py_ssize_t n_feat,
    const double* y_full, const double* sw_full, double total_sw,
    double* scores,  # output (n_feat,)
) noexcept nogil:
    """Cheap feature ranking for top-k oblique search."""
    cdef Py_ssize_t f, i, cls_idx
    cdef int col, yi_int
    cdef double mean_y, mean_x, cov, var_x, var_y
    cdef double xi, yi, wi
    cdef double between_var, denom, cls_mean, diff
    cdef const double* col_ptr
    cdef double* class_weight = NULL
    cdef double* class_sum = NULL
    cdef bint use_multiclass_score = task == 0 and n_classes > 2

    if total_sw <= 0.0:
        for f in range(n_feat):
            scores[f] = 0.0
        return

    mean_y = 0.0
    var_y = 0.0
    if not use_multiclass_score:
        for i in range(N):
            mean_y += sw_full[row_idx[i]] * y_full[row_idx[i]]
        mean_y /= total_sw

        for i in range(N):
            wi = sw_full[row_idx[i]]
            yi = y_full[row_idx[i]] - mean_y
            var_y += wi * yi * yi
    else:
        class_weight = <double*>malloc(n_classes * sizeof(double))
        class_sum = <double*>malloc(n_classes * sizeof(double))
        if class_weight == NULL or class_sum == NULL:
            free(class_weight)
            free(class_sum)
            use_multiclass_score = False

            for i in range(N):
                mean_y += sw_full[row_idx[i]] * y_full[row_idx[i]]
            mean_y /= total_sw

            for i in range(N):
                wi = sw_full[row_idx[i]]
                yi = y_full[row_idx[i]] - mean_y
                var_y += wi * yi * yi

    for f in range(n_feat):
        col = features[f]
        col_ptr = X + <Py_ssize_t>col * full_N

        mean_x = 0.0
        for i in range(N):
            mean_x += sw_full[row_idx[i]] * col_ptr[row_idx[i]]
        mean_x /= total_sw

        cov = 0.0; var_x = 0.0
        if use_multiclass_score:
            memset(class_weight, 0, n_classes * sizeof(double))
            memset(class_sum, 0, n_classes * sizeof(double))

            for i in range(N):
                xi = col_ptr[row_idx[i]] - mean_x
                wi = sw_full[row_idx[i]]
                yi_int = <int>y_full[row_idx[i]]
                var_x += wi * xi * xi
                class_weight[yi_int] += wi
                class_sum[yi_int] += wi * col_ptr[row_idx[i]]

            if var_x <= 0.0:
                scores[f] = 0.0
                continue

            between_var = 0.0
            for cls_idx in range(n_classes):
                if class_weight[cls_idx] <= 0.0:
                    continue
                cls_mean = class_sum[cls_idx] / class_weight[cls_idx]
                diff = cls_mean - mean_x
                between_var += class_weight[cls_idx] * diff * diff
            scores[f] = between_var / var_x
            continue

        for i in range(N):
            xi = col_ptr[row_idx[i]] - mean_x
            yi = y_full[row_idx[i]] - mean_y
            wi = sw_full[row_idx[i]]
            cov += wi * xi * yi
            var_x += wi * xi * xi

        if var_x > 0.0 and var_y > 0.0:
            denom = sqrt(var_x * var_y)
            if denom > 0.0:
                scores[f] = fabs(cov) / denom
            else:
                scores[f] = 0.0
        else:
            scores[f] = 0.0

    free(class_weight)
    free(class_sum)

# ---------------------------------------------------------------------------
# Simple top-k selection via partial insertion sort  (nogil, O(n * k))
# ---------------------------------------------------------------------------
cdef void topk_indices(
    const double* scores, Py_ssize_t n,
    int* out, Py_ssize_t k,          # output: indices of top-k
    const int* features,              # map score index -> feature id
) noexcept nogil:
    """Select k indices with largest scores.  O(n*k), fine for n < ~100."""
    cdef Py_ssize_t i, j, pos
    cdef double s
    # init with -INF
    cdef double* vals = <double*>malloc(k * sizeof(double))
    if vals == NULL:
        for i in range(k):
            out[i] = features[i]
        return
    for i in range(k):
        vals[i] = -INFINITY
        out[i] = features[0]

    for i in range(n):
        s = scores[i]
        if s > vals[k - 1]:
            # insert
            pos = k - 1
            while pos > 0 and s > vals[pos - 1]:
                vals[pos] = vals[pos - 1]
                out[pos] = out[pos - 1]
                pos -= 1
            vals[pos] = s
            out[pos] = features[i]
    free(vals)


# ---------------------------------------------------------------------------
# Projection helper
# ---------------------------------------------------------------------------
cdef void fill_oblique_sort_buffer(
                np.ndarray[double, ndim=2] X,
                const int* sample_indices,
                SortItem* sort_buffer,
                const int n_samples,
                const int n_pair,
                const int* pair_idx,
                const double* x) noexcept:
    cdef Py_ssize_t N = n_samples
    cdef Py_ssize_t full_N = X.shape[0]
    cdef Py_ssize_t i, fi
    cdef double bv
    cdef double* X_ptr

    if not X.flags['F_CONTIGUOUS']:
        X = np.asfortranarray(X)
    X_ptr = <double*>np.PyArray_DATA(X)

    with nogil:
        for i in range(N):
            bv = 0.0
            for fi in range(n_pair):
                bv += X_ptr[sample_indices[i] + <Py_ssize_t>pair_idx[fi] * full_N] * x[fi]
            sort_buffer[i].value = bv
            sort_buffer[i].index = sample_indices[i]
        sort_pointer_array(sort_buffer, N)


cdef inline void fill_oblique_sort_buffer_ptr(
                const double* X_ptr,
                const Py_ssize_t full_N,
                const int* sample_indices,
                SortItem* sort_buffer,
                const int n_samples,
                const int n_pair,
                const int* pair_idx,
                const double* x) noexcept nogil:
    cdef Py_ssize_t N = n_samples
    cdef Py_ssize_t i, fi
    cdef double bv

    for i in range(N):
        bv = 0.0
        for fi in range(n_pair):
            bv += X_ptr[sample_indices[i] + <Py_ssize_t>pair_idx[fi] * full_N] * x[fi]
        sort_buffer[i].value = bv
        sort_buffer[i].index = sample_indices[i]
    sort_pointer_array(sort_buffer, N)


# ---------------------------------------------------------------------------
# analyze()
# ---------------------------------------------------------------------------
cdef int* prepare_candidate_pairs(
                const bint task,
                const int n_classes,
                np.ndarray[double, ndim=2] X,
                np.ndarray[double, ndim=1] y,
                np.ndarray[double, ndim=1] sample_weight,
                const int* sample_indices,
                const int n_samples,
                const int n_pair,
                const bint* is_categorical,
                Py_ssize_t* out_n_fp,
              ) noexcept:
    cdef Py_ssize_t N = n_samples
    cdef Py_ssize_t d = X.shape[1]
    cdef Py_ssize_t full_N = X.shape[0]
    cdef Py_ssize_t i
    cdef Py_ssize_t n_usable = 0
    cdef Py_ssize_t k
    cdef Py_ssize_t n_fp
    cdef double sum_sw = 0.0
    cdef int* usable = NULL
    cdef int* selected = NULL
    cdef int* topk_buf = NULL
    cdef int* pair_idx = NULL
    cdef double* f_scores = NULL
    cdef double* X_ptr
    cdef double* y_ptr
    cdef double* sw_ptr

    out_n_fp[0] = 0

    usable = <int*>malloc(d * sizeof(int))
    if usable == NULL:
        return NULL

    if is_categorical:
        for i in range(d):
            if not is_categorical[i]:
                usable[n_usable] = <int>i
                n_usable += 1
    else:
        for i in range(d):
            usable[i] = <int>i
        n_usable = d

    if n_usable < n_pair:
        free(usable)
        return NULL

    if not X.flags['F_CONTIGUOUS']:
        X = np.asfortranarray(X)
    X_ptr = <double*>np.PyArray_DATA(X)
    y_ptr = <double*>np.PyArray_DATA(y)
    sw_ptr = <double*>np.PyArray_DATA(sample_weight)

    for i in range(N):
        sum_sw += sw_ptr[sample_indices[i]]

    k = <Py_ssize_t>sqrt(<double>n_usable)
    if k < 2 * n_pair:
        k = 2 * n_pair
    if k > n_usable:
        k = n_usable

    selected = usable
    if k < n_usable:
        f_scores = <double*>malloc(n_usable * sizeof(double))
        topk_buf = <int*>malloc(k * sizeof(int))
        if f_scores == NULL or topk_buf == NULL:
            free(f_scores)
            free(topk_buf)
            free(usable)
            return NULL

        score_features(task, n_classes, X_ptr, full_N, sample_indices, N,
                       usable, n_usable, y_ptr, sw_ptr, sum_sw, f_scores)
        topk_indices(f_scores, n_usable, topk_buf, k, usable)
        free(f_scores)
        selected = topk_buf

    n_fp = count_pairs(k, n_pair)
    if n_fp > 0:
        pair_idx = <int*>malloc(n_fp * n_pair * sizeof(int))
        if pair_idx != NULL:
            if n_pair == 2:
                generate_pairs_2(selected, k, pair_idx)
            elif n_pair == 3:
                generate_pairs_3(selected, k, pair_idx)
            else:
                generate_combinations(selected, k, n_pair, pair_idx)
            out_n_fp[0] = n_fp

    if selected != usable:
        free(selected)
    free(usable)
    return pair_idx


cdef tuple[double*, int*] _analyze_from_pairs_compact(
                const bint task,
                const int n_classes,
                const bint linear,
                const double* X_ptr,
                const Py_ssize_t full_N,
                np.ndarray[double, ndim=1] y_sub,
                np.ndarray[double, ndim=1] sw_sub,
                const double sum_sw,
                const int* sample_indices,
                SortItem* sort_buffer,
                const int n_samples,
                const int n_pair,
                const int* pair_idx,
                const Py_ssize_t n_fp,
                const unsigned long long rng_seed,
                ThreadWorkspace* ws,
                const int n_threads,
                const int lbfgs_m,
                const bint write_sort_buffer,
                const double gamma,
                const int maxiter,
                const double relative_change,
              ) noexcept:
    cdef Py_ssize_t N = n_samples
    cdef Py_ssize_t i
    cdef int* best_pair = NULL
    cdef double* best_x = NULL
    cdef double best_fx = INFINITY
    cdef double* y_ptr
    cdef double* sw_ptr
    cdef const int* screen_sample_indices = sample_indices
    cdef double* y_screen_ptr
    cdef double* sw_screen_ptr
    cdef Py_ssize_t screen_N = N
    cdef double screen_sum_sw
    cdef int t
    cdef Py_ssize_t pi, fi
    cdef int tid, ci
    cdef double f_val
    cdef double pair_best_f
    cdef ThreadWorkspace* w
    cdef int multi_range = 1
    cdef Py_ssize_t n_survivors, surv_target
    cdef double* screen_losses = NULL
    cdef double* screen_weights = NULL
    cdef Py_ssize_t* surv_idx = NULL
    cdef bint do_screen
    cdef Py_ssize_t j_s
    cdef double loss_j
    cdef Py_ssize_t si, real_pi
    cdef Py_ssize_t best_job = 0
    cdef double max_abs = 0.0
    cdef np.ndarray[int, ndim=1] screen_sample_idx_arr
    cdef np.ndarray[double, ndim=1] y_screen_arr
    cdef np.ndarray[double, ndim=1] sw_screen_arr
    cdef int* screen_sample_idx_ptr = NULL
    cdef double* y_screen_arr_ptr = NULL
    cdef double* sw_screen_arr_ptr = NULL
    cdef Py_ssize_t screen_src_i

    if pair_idx == NULL or n_fp <= 0:
        return NULL, NULL

    best_pair = <int*>malloc(n_pair * sizeof(int))
    best_x = <double*>malloc(n_pair * sizeof(double))
    if best_pair == NULL or best_x == NULL:
        free(best_pair); free(best_x)
        return NULL, NULL

    y_ptr = <double*>np.PyArray_DATA(y_sub)
    sw_ptr = <double*>np.PyArray_DATA(sw_sub)
    y_screen_ptr = y_ptr
    sw_screen_ptr = sw_ptr
    screen_sum_sw = sum_sw

    if ws == NULL:
        free(best_pair); free(best_x)
        return NULL, NULL

    if linear and n_classes > 2:
        multi_range = n_classes

    # ===================================================================
    # Phase 3: Cheap proxy screening on all pairs
    # ===================================================================
    # Skip screening when top-k has already reduced the pair set enough.
    do_screen = n_fp >= MIN_SCREEN_PAIRS

    if do_screen and N > MAX_SCREEN_SAMPLES:
        screen_N = MAX_SCREEN_SAMPLES
        screen_sample_idx_arr = np.empty(screen_N, dtype=np.int32)
        y_screen_arr = np.empty(screen_N, dtype=np.float64)
        sw_screen_arr = np.empty(screen_N, dtype=np.float64)
        screen_sample_idx_ptr = <int*>np.PyArray_DATA(screen_sample_idx_arr)
        y_screen_arr_ptr = <double*>np.PyArray_DATA(y_screen_arr)
        sw_screen_arr_ptr = <double*>np.PyArray_DATA(sw_screen_arr)
        screen_sum_sw = 0.0
        for i in range(screen_N):
            screen_src_i = ((2 * i + 1) * N) // (2 * screen_N)
            if screen_src_i >= N:
                screen_src_i = N - 1
            screen_sample_idx_ptr[i] = sample_indices[screen_src_i]
            y_screen_arr_ptr[i] = y_ptr[screen_src_i]
            sw_screen_arr_ptr[i] = sw_ptr[screen_src_i]
            screen_sum_sw += sw_screen_arr_ptr[i]
        screen_sample_indices = screen_sample_idx_ptr
        y_screen_ptr = y_screen_arr_ptr
        sw_screen_ptr = sw_screen_arr_ptr

    if do_screen:
        screen_losses = <double*>malloc(n_fp * sizeof(double))
        screen_weights = <double*>malloc(n_fp * n_pair * sizeof(double))
        if screen_losses == NULL or screen_weights == NULL:
            free(screen_losses)
            free(screen_weights)
            screen_losses = NULL
            screen_weights = NULL
            do_screen = False

    if do_screen:
        for pi in prange(n_fp, nogil=True, schedule="dynamic"):
            tid = openmp.omp_get_thread_num()
            w = &ws[tid]

            gather_X_pair(X_ptr, full_N, screen_sample_indices, screen_N,
                          &pair_idx[pi * n_pair], n_pair, w.X_pair)
            init_x0_deterministic(w.x_work, n_pair, rng_seed, pi)
            memcpy(w.pair_best_weights, w.x_work, n_pair * sizeof(double))

            pair_best_f = INFINITY
            for ci in range(multi_range):
                memset(w.s_hist, 0, lbfgs_m * n_pair * sizeof(double))
                memset(w.y_hist, 0, lbfgs_m * n_pair * sizeof(double))
                memset(w.rho, 0, lbfgs_m * sizeof(double))
                f_val = proxy_screen_nogil(
                    task, n_classes, <double>ci, linear,
                    w.x_work, w.X_pair, y_screen_ptr, sw_screen_ptr, screen_sum_sw,
                    screen_N, n_pair, gamma, 1e-6, 1e-5,
                    w.grad, w.grad_new, w.x_new,
                    w.buf_z, w.buf_p, w.buf_dp_dz,
                    w.buf1, w.buf2, w.buf3,
                    w.buf4, w.buf5, w.buf6, w.buf7)
                if f_val < pair_best_f:
                    pair_best_f = f_val
                    memcpy(w.pair_best_weights, w.x_work, n_pair * sizeof(double))
            screen_losses[pi] = pair_best_f
            memcpy(&screen_weights[pi * n_pair], w.pair_best_weights, n_pair * sizeof(double))

        # Keep only a small survivor set when screening is actually worthwhile.
        surv_target = n_fp // SCREEN_KEEP_DIVISOR
        if surv_target < MIN_SCREEN_SURVIVORS:
            surv_target = MIN_SCREEN_SURVIVORS
        if surv_target > n_fp:
            surv_target = n_fp

        # Find threshold: partial sort via finding the surv_target-th smallest
        # Simple approach: collect all losses and pick threshold
        surv_idx = <Py_ssize_t*>malloc(n_fp * sizeof(Py_ssize_t))
        if surv_idx == NULL:
            free(screen_losses)
            free(screen_weights)
            screen_losses = NULL
            screen_weights = NULL
            do_screen = False

    if do_screen:
        n_survivors = 0
        for i in range(n_fp):
            # Insert into sorted surv_idx
            loss_j = screen_losses[i]
            if n_survivors < surv_target:
                # Find insert position
                j_s = n_survivors
                while j_s > 0 and screen_losses[surv_idx[j_s - 1]] > loss_j:
                    surv_idx[j_s] = surv_idx[j_s - 1]
                    j_s -= 1
                surv_idx[j_s] = i
                n_survivors += 1
            elif loss_j < screen_losses[surv_idx[n_survivors - 1]]:
                # Replace worst survivor
                j_s = n_survivors - 1
                while j_s > 0 and screen_losses[surv_idx[j_s - 1]] > loss_j:
                    surv_idx[j_s] = surv_idx[j_s - 1]
                    j_s -= 1
                surv_idx[j_s] = i
        free(screen_losses)
    else:
        free(screen_weights)
        screen_weights = NULL
        # No screening: all pairs are survivors
        n_survivors = n_fp
        surv_idx = <Py_ssize_t*>malloc(n_fp * sizeof(Py_ssize_t))
        if surv_idx != NULL:
            for i in range(n_fp):
                surv_idx[i] = i
        else:
            free(best_pair); free(best_x)
            free_workspaces(ws, n_threads)
            return NULL, NULL

    # ===================================================================
    # Phase 4: Full optimization on survivors only
    # ===================================================================
    for t in range(n_threads):
        ws[t].best_loss = INFINITY
        ws[t].best_pi = 0

    for si in prange(n_survivors, nogil=True, schedule="dynamic"):
        tid = openmp.omp_get_thread_num()
        w = &ws[tid]
        real_pi = surv_idx[si]

        gather_X_pair(X_ptr, full_N, sample_indices, N,
                      &pair_idx[real_pi * n_pair], n_pair, w.X_pair)
        if do_screen:
            memcpy(w.x_work, &screen_weights[real_pi * n_pair], n_pair * sizeof(double))
        else:
            init_x0_deterministic(w.x_work, n_pair, rng_seed, real_pi)
        memcpy(w.pair_best_weights, w.x_work, n_pair * sizeof(double))

        pair_best_f = INFINITY
        for ci in range(multi_range):
            memset(w.s_hist, 0, lbfgs_m * n_pair * sizeof(double))
            memset(w.y_hist, 0, lbfgs_m * n_pair * sizeof(double))
            memset(w.rho, 0, lbfgs_m * sizeof(double))
            f_val = lbfgs_minimize_nogil(
                task, n_classes, <double>ci, linear,
                w.x_work, w.X_pair, y_ptr, sw_ptr, sum_sw,
                N, n_pair, gamma, 1e-6, lbfgs_m,
                maxiter, relative_change, 1e-5, 20,
                w.grad, w.grad_new, w.direction, w.x_new,
                w.s_hist, w.y_hist, w.rho, w.alpha_buf,
                w.buf_z, w.buf_p, w.buf_dp_dz,
                w.buf1, w.buf2, w.buf3,
                w.buf4, w.buf5, w.buf6, w.buf7)
            if f_val < pair_best_f:
                pair_best_f = f_val
                memcpy(w.pair_best_weights, w.x_work, n_pair * sizeof(double))

        if pair_best_f < w.best_loss:
            w.best_loss = pair_best_f
            w.best_pi = real_pi
            memcpy(w.best_weights, w.pair_best_weights, n_pair * sizeof(double))

    free(surv_idx)
    free(screen_weights)

    # --- Final reduction across threads ---
    for t in range(n_threads):
        if ws[t].best_loss < best_fx:
            best_fx = ws[t].best_loss
            best_job = ws[t].best_pi
            memcpy(best_x, ws[t].best_weights, n_pair * sizeof(double))

    # Normalize weights
    for i in range(n_pair):
        if fabs(best_x[i]) > max_abs:
            max_abs = fabs(best_x[i])
    if max_abs == 0.0:
        max_abs = 1.0
    for i in range(n_pair):
        best_pair[i] = pair_idx[best_job * n_pair + i]
        best_x[i] = best_x[i] / max_abs

    if write_sort_buffer:
        with nogil:
            fill_oblique_sort_buffer_ptr(
                X_ptr,
                full_N,
                sample_indices,
                sort_buffer,
                n_samples,
                n_pair,
                best_pair,
                best_x,
            )
    return best_x, best_pair


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
    cdef Py_ssize_t n_fp = 0
    cdef int* pair_idx = NULL
    cdef double* best_x = NULL
    cdef int* best_pair = NULL
    cdef np.ndarray[int, ndim=1] sample_dx
    cdef np.ndarray[double, ndim=1] y_sub
    cdef np.ndarray[double, ndim=1] sw_sub
    cdef double sum_sw
    cdef double* X_ptr
    cdef Py_ssize_t full_N
    cdef unsigned long long rng_seed
    cdef int n_threads
    cdef int lbfgs_m = 10
    cdef ThreadWorkspace* ws = NULL
    pair_idx = prepare_candidate_pairs(
        task,
        n_classes,
        X,
        y,
        sample_weight,
        sample_indices,
        n_samples,
        n_pair,
        is_categorical,
        &n_fp,
    )

    if pair_idx == NULL or n_fp == 0:
        return NULL, NULL

    sample_dx = np.frombuffer(
        <bytes>(<char*>sample_indices)[:n_samples * sizeof(int)], dtype=np.int32)
    y_sub = y[sample_dx].copy()
    sw_sub = sample_weight[sample_dx].copy()
    sum_sw = sw_sub.sum()
    if not X.flags['F_CONTIGUOUS']:
        X = np.asfortranarray(X)
    X_ptr = <double*>np.PyArray_DATA(X)
    full_N = X.shape[0]
    rng_seed = <unsigned long long>rng.integers(0, 2**63)
    n_threads = openmp.omp_get_max_threads()
    ws = alloc_workspaces_shared(n_threads, n_samples, n_pair, lbfgs_m, n_classes, task)
    if ws == NULL:
        free(pair_idx)
        return NULL, NULL

    best_x, best_pair = _analyze_from_pairs_compact(
        task,
        n_classes,
        linear,
        X_ptr,
        full_N,
        y_sub,
        sw_sub,
        sum_sw,
        sample_indices,
        sort_buffer,
        n_samples,
        n_pair,
        pair_idx,
        n_fp,
        rng_seed,
        ws,
        n_threads,
        lbfgs_m,
        True,
        gamma,
        maxiter,
        relative_change,
    )
    free_workspaces(ws, n_threads)
    free(pair_idx)
    return best_x, best_pair


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
                const double relative_change,
              ) noexcept:
    cdef np.ndarray[int, ndim=1] sample_dx
    cdef np.ndarray[double, ndim=1] y_sub
    cdef np.ndarray[double, ndim=1] sw_sub
    cdef double sum_sw
    cdef double* X_ptr
    cdef Py_ssize_t full_N
    cdef unsigned long long rng_seed
    cdef int n_threads
    cdef int lbfgs_m = 10
    cdef ThreadWorkspace* ws = NULL
    cdef double* best_x = NULL
    cdef int* best_pair = NULL

    sample_dx = np.frombuffer(
        <bytes>(<char*>sample_indices)[:n_samples * sizeof(int)], dtype=np.int32)
    y_sub = y[sample_dx].copy()
    sw_sub = sample_weight[sample_dx].copy()
    sum_sw = sw_sub.sum()
    if not X.flags['F_CONTIGUOUS']:
        X = np.asfortranarray(X)
    X_ptr = <double*>np.PyArray_DATA(X)
    full_N = X.shape[0]
    rng_seed = <unsigned long long>rng.integers(0, 2**63)
    n_threads = openmp.omp_get_max_threads()
    ws = alloc_workspaces_shared(n_threads, n_samples, n_pair, lbfgs_m, n_classes, task)
    if ws == NULL:
        return NULL, NULL

    best_x, best_pair = _analyze_from_pairs_compact(
        task,
        n_classes,
        linear,
        X_ptr,
        full_N,
        y_sub,
        sw_sub,
        sum_sw,
        sample_indices,
        sort_buffer,
        n_samples,
        n_pair,
        pair_idx,
        n_fp,
        rng_seed,
        ws,
        n_threads,
        lbfgs_m,
        True,
        gamma,
        maxiter,
        relative_change,
    )
    free_workspaces(ws, n_threads)
    return best_x, best_pair


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
                int** out_linear_pair,
              ) noexcept:
    cdef np.ndarray[int, ndim=1] sample_dx
    cdef np.ndarray[double, ndim=1] y_sub
    cdef np.ndarray[double, ndim=1] sw_sub
    cdef double sum_sw
    cdef double* X_ptr
    cdef Py_ssize_t full_N
    cdef unsigned long long rng_seed
    cdef int n_threads
    cdef int lbfgs_m = 10
    cdef ThreadWorkspace* ws = NULL
    cdef double* oblique_x = NULL
    cdef int* oblique_pair = NULL
    cdef double* linear_x = NULL
    cdef int* linear_pair = NULL

    out_oblique_x[0] = NULL
    out_oblique_pair[0] = NULL
    out_linear_x[0] = NULL
    out_linear_pair[0] = NULL

    sample_dx = np.frombuffer(
        <bytes>(<char*>sample_indices)[:n_samples * sizeof(int)], dtype=np.int32)
    y_sub = y[sample_dx].copy()
    sw_sub = sample_weight[sample_dx].copy()
    sum_sw = sw_sub.sum()
    if not X.flags['F_CONTIGUOUS']:
        X = np.asfortranarray(X)
    X_ptr = <double*>np.PyArray_DATA(X)
    full_N = X.shape[0]
    rng_seed = <unsigned long long>rng.integers(0, 2**63)
    n_threads = openmp.omp_get_max_threads()
    ws = alloc_workspaces_shared(n_threads, n_samples, n_pair, lbfgs_m, n_classes, task)
    if ws == NULL:
        return

    oblique_x, oblique_pair = _analyze_from_pairs_compact(
        task,
        n_classes,
        False,
        X_ptr,
        full_N,
        y_sub,
        sw_sub,
        sum_sw,
        sample_indices,
        sort_buffer,
        n_samples,
        n_pair,
        pair_idx,
        n_fp,
        rng_seed,
        ws,
        n_threads,
        lbfgs_m,
        False,
        gamma,
        maxiter,
        relative_change,
    )
    linear_x, linear_pair = _analyze_from_pairs_compact(
        task,
        n_classes,
        True,
        X_ptr,
        full_N,
        y_sub,
        sw_sub,
        sum_sw,
        sample_indices,
        sort_buffer,
        n_samples,
        n_pair,
        pair_idx,
        n_fp,
        rng_seed ^ <unsigned long long>0x9e3779b97f4a7c15ULL,
        ws,
        n_threads,
        lbfgs_m,
        False,
        gamma,
        maxiter,
        relative_change,
    )
    free_workspaces(ws, n_threads)
    out_oblique_x[0] = oblique_x
    out_oblique_pair[0] = oblique_pair
    out_linear_x[0] = linear_x
    out_linear_pair[0] = linear_pair
