#pragma once

#include <cstddef>

namespace forward_analysis {

// OpenCL C source of the element-wise and row-wise decoder kernels, compiled at runtime by OpenClRuntime.
// Matrices are column-major doubles (element (r, c) of an R-row matrix at c * R + r), the layout Eigen uses,
// so host matrices upload without reordering. Each kernel reproduces the arithmetic order of the matching
// Eigen expression in AutoregressiveCausalDecoder / MatrixOps / AdamMoment.
struct DecoderKernelSource {
    // Work-items per work-group in the reduction kernels (layer norm, softmax, gemm_reduce, column gradients); a
    // power of two, passed as -DROW_GROUP.
    static constexpr std::size_t kRowGroup = 128;

    static const char* text() noexcept {
        return R"CLC(
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#pragma OPENCL FP_CONTRACT OFF

// Row kernels run one work-group of ROW_GROUP work-items per row (ROW_GROUP is a power of two passed at build
// time); each work-item strides over the row's columns and the partial results meet in a local-memory tree.
double group_sum(__local double* scratch, const double value) {
    const int l = get_local_id(0);
    scratch[l] = value;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int span = ROW_GROUP / 2; span > 0; span >>= 1) {
        if (l < span) {
            scratch[l] += scratch[l + span];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    const double total = scratch[0];
    barrier(CLK_LOCAL_MEM_FENCE);
    return total;
}

double group_max(__local double* scratch, const double value) {
    const int l = get_local_id(0);
    scratch[l] = value;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int span = ROW_GROUP / 2; span > 0; span >>= 1) {
        if (l < span) {
            scratch[l] = fmax(scratch[l], scratch[l + span]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    const double top = scratch[0];
    barrier(CLK_LOCAL_MEM_FENCE);
    return top;
}

// C (M x N) = alpha op(A) op(B) + beta C, column-major with leading dimensions lda / ldb / ldc; op transposes when
// the flag is set; with beta = 0, C is not read. One work-item per element of C, accumulating over K in order.
__kernel void gemm_direct(const int M, const int N, const int K, const int trans_a, const int trans_b, const double alpha,
                          __global const double* A, const int lda, __global const double* B, const int ldb,
                          const double beta, __global double* C, const int ldc) {
    const int i = get_global_id(0);
    const int j = get_global_id(1);
    if (i >= M || j >= N) {
        return;
    }
    double acc = 0.0;
    for (int k = 0; k < K; ++k) {
        const double a = trans_a != 0 ? A[i * lda + k] : A[k * lda + i];
        const double b = trans_b != 0 ? B[k * ldb + j] : B[j * ldb + k];
        acc += a * b;
    }
    const int index = j * ldc + i;
    C[index] = beta == 0.0 ? alpha * acc : alpha * acc + beta * C[index];
}

// The same product for outputs with few entries: one ROW_GROUP work-group per element of C, its work-items striding
// over K and meeting in a local-memory tree.
__kernel __attribute__((reqd_work_group_size(ROW_GROUP, 1, 1)))
void gemm_reduce(const int M, const int N, const int K, const int trans_a, const int trans_b, const double alpha,
                 __global const double* A, const int lda, __global const double* B, const int ldb, const double beta,
                 __global double* C, const int ldc) {
    __local double scratch[ROW_GROUP];
    const int element = get_group_id(0);
    const int i = element % M;
    const int j = element / M;
    const int l = get_local_id(0);
    double part = 0.0;
    for (int k = l; k < K; k += ROW_GROUP) {
        const double a = trans_a != 0 ? A[i * lda + k] : A[k * lda + i];
        const double b = trans_b != 0 ? B[k * ldb + j] : B[j * ldb + k];
        part += a * b;
    }
    const double acc = group_sum(scratch, part);
    if (l == 0) {
        const int index = j * ldc + i;
        C[index] = beta == 0.0 ? alpha * acc : alpha * acc + beta * C[index];
    }
}

)CLC"
               R"CLC(// u (S x d): u[t] = E[prev_t] * m1[t] + w2 * (E2[prev2_t] * m2[t]) + F[field] + P[t]
__kernel void build_step_inputs(const int S, const int d, const int V, const int nF, const int nP, const int field,
                                const double w2, __global const int* prev, __global const int* prev2,
                                __global const double* E, __global const double* E2, __global const double* F,
                                __global const double* P, __global const double* m1, __global const double* m2,
                                __global double* u) {
    const int t = get_global_id(0);
    const int j = get_global_id(1);
    if (t >= S || j >= d) {
        return;
    }
    const int i = j * S + t;
    u[i] = ((E[j * V + prev[t]] * m1[i] + w2 * (E2[j * V + prev2[t]] * m2[i])) + F[j * nF + field]) + P[j * nP + t];
}

// Per row r: inv_std[r] = 1 / sqrt(mean((X_r - mean_r)^2) + eps), Y_r = inv_std[r] * (X_r - mean_r)
__kernel __attribute__((reqd_work_group_size(ROW_GROUP, 1, 1)))
void layer_norm_rows(const int R, const int C, const double eps, __global const double* X, __global double* Y,
                     __global double* inv_std) {
    __local double scratch[ROW_GROUP];
    const int r = get_group_id(0);
    const int l = get_local_id(0);
    double part = 0.0;
    for (int c = l; c < C; c += ROW_GROUP) {
        part += X[c * R + r];
    }
    const double mean = group_sum(scratch, part) / (double)C;
    part = 0.0;
    for (int c = l; c < C; c += ROW_GROUP) {
        const double z = X[c * R + r] - mean;
        part += z * z;
    }
    const double var = group_sum(scratch, part) / (double)C;
    const double s = 1.0 / sqrt(var + eps);
    if (l == 0) {
        inv_std[r] = s;
    }
    for (int c = l; c < C; c += ROW_GROUP) {
        Y[c * R + r] = s * (X[c * R + r] - mean);
    }
}

// Row-wise softmax in place: exp(max(x - rowmax, -60)) / row sum
__kernel __attribute__((reqd_work_group_size(ROW_GROUP, 1, 1)))
void softmax_rows(const int R, const int C, __global double* X) {
    __local double scratch[ROW_GROUP];
    const int r = get_group_id(0);
    const int l = get_local_id(0);
    double part = -INFINITY;
    for (int c = l; c < C; c += ROW_GROUP) {
        part = fmax(part, X[c * R + r]);
    }
    const double top = group_max(scratch, part);
    part = 0.0;
    for (int c = l; c < C; c += ROW_GROUP) {
        const double e = exp(fmax(X[c * R + r] - top, -60.0));
        X[c * R + r] = e;
        part += e;
    }
    const double sum = group_sum(scratch, part);
    for (int c = l; c < C; c += ROW_GROUP) {
        X[c * R + r] /= sum;
    }
}

)CLC"
               R"CLC(// softmax_rows for few columns (attention over the M latents): one work-item per row.
__kernel void softmax_rows_narrow(const int R, const int C, __global double* X) {
    const int r = get_global_id(0);
    if (r >= R) {
        return;
    }
    double top = X[r];
    for (int c = 1; c < C; ++c) {
        top = fmax(top, X[c * R + r]);
    }
    double sum = 0.0;
    for (int c = 0; c < C; ++c) {
        const double e = exp(fmax(X[c * R + r] - top, -60.0));
        X[c * R + r] = e;
        sum += e;
    }
    for (int c = 0; c < C; ++c) {
        X[c * R + r] /= sum;
    }
}

// Inverted-dropout masks for one training pass, bit-identical to DeterministicRng::uniform() draws taken in the
// host order: the token mask row by row (S x d), then the second-token mask (S x d), then the hidden mask (S x h).
// Draw k (0-based) of the pass is splitmix64 of state + (k + 1) * golden; a draw below p zeroes the entry,
// otherwise it becomes keep = 1 / (1 - p). Masks are stored column-major like every other matrix. The 64-bit
// state arrives as its low and high 32-bit halves.
__kernel void dropout_masks(const int S, const int d, const int h, const uint state_low, const uint state_high,
                            const double p, const double keep, __global double* m1, __global double* m2,
                            __global double* mh) {
    const int k = get_global_id(0);
    const int token_draws = S * d;
    if (k >= 2 * token_draws + S * h) {
        return;
    }
    const ulong state = ((ulong)state_high << 32) | (ulong)state_low;
    ulong z = state + (ulong)(k + 1) * 0x9E3779B97F4A7C15UL;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
    z = z ^ (z >> 31);
    const double unit = (double)(z >> 11) * (1.0 / 9007199254740992.0);
    const double value = unit < p ? 0.0 : keep;
    if (k < token_draws) {
        m1[(k % d) * S + k / d] = value;
    } else if (k < 2 * token_draws) {
        const int q = k - token_draws;
        m2[(q % d) * S + q / d] = value;
    } else {
        const int q = k - 2 * token_draws;
        mh[(q % h) * S + q / h] = value;
    }
}

// X (R x C, holding A V) becomes (U + A V) + (W_aux d_hat)^T in every row; W_aux is C x dim.
__kernel void add_inputs_and_conditioning(const int R, const int C, const int dim, __global double* X,
                                          __global const double* U, __global const double* W,
                                          __global const double* descriptor) {
    const int r = get_global_id(0);
    const int c = get_global_id(1);
    if (r >= R || c >= C) {
        return;
    }
    double conditioning = 0.0;
    for (int k = 0; k < dim; ++k) {
        conditioning += W[k * C + c] * descriptor[k];
    }
    const int i = c * R + r;
    X[i] = (U[i] + X[i]) + conditioning;
}

// H (R x C, holding x W1^T) becomes tanh(H + b1) in place; dropped = H * mask
__kernel void bias_tanh_mask(const int R, const int C, __global double* H, __global const double* b,
                             __global const double* mask, __global double* dropped) {
    const int r = get_global_id(0);
    const int c = get_global_id(1);
    if (r < R && c < C) {
        const int i = c * R + r;
        const double h = tanh(H[i] + b[c]);
        H[i] = h;
        dropped[i] = h * mask[i];
    }
}

// Per row r of X (R x C, holding h W_out^T): P = softmax(X + b) with exp(max(x - rowmax, -60)),
// loss[loss_offset + r] = -log(max(1e-12, P[r, target_r])); with write_gradient, X then holds
// dLoss/dlogits = (P - onehot(target)) / R instead of P.
__kernel __attribute__((reqd_work_group_size(ROW_GROUP, 1, 1)))
void softmax_loss_rows(const int R, const int C, __global double* X, __global const double* b,
                       __global const int* targets, __global double* loss, const int loss_offset,
                       const int write_gradient) {
    __local double scratch[ROW_GROUP];
    const int r = get_group_id(0);
    const int l = get_local_id(0);
    const int target = targets[r];
    double part = -INFINITY;
    for (int c = l; c < C; c += ROW_GROUP) {
        part = fmax(part, X[c * R + r] + b[c]);
    }
    const double top = group_max(scratch, part);
    part = 0.0;
    for (int c = l; c < C; c += ROW_GROUP) {
        const double e = exp(fmax((X[c * R + r] + b[c]) - top, -60.0));
        X[c * R + r] = e;
        part += e;
    }
    const double sum = group_sum(scratch, part);
    for (int c = l; c < C; c += ROW_GROUP) {
        const double p = X[c * R + r] / sum;
        if (c == target) {
            loss[loss_offset + r] = -log(fmax(1e-12, p));
        }
        X[c * R + r] = write_gradient != 0 ? (c == target ? p - 1.0 : p) / (double)R : p;
    }
}

// out (C x dim) = (column sums of dX (R x C)) d_hat^T, the W_aux gradient.
__kernel void descriptor_gradient(const int R, const int C, const int dim, __global const double* dX,
                                  __global const double* descriptor, __global double* out) {
    const int c = get_global_id(0);
    const int k = get_global_id(1);
    if (c >= C || k >= dim) {
        return;
    }
    double s = 0.0;
    for (int r = 0; r < R; ++r) {
        s += dX[c * R + r];
    }
    out[k * C + c] = s * descriptor[k];
}

// out[c * stride + offset] = sum_r X[r, c]
__kernel void column_sums(const int R, const int C, __global const double* X, const int stride, const int offset,
                          __global double* out) {
    const int c = get_global_id(0);
    if (c >= C) {
        return;
    }
    double s = 0.0;
    for (int r = 0; r < R; ++r) {
        s += X[c * R + r];
    }
    out[c * stride + offset] = s;
}

)CLC"
               R"CLC(// One ROW_GROUP work-group per column c of the R x C hidden layer: dpre = (dh * mask) * (1 - h^2), and
// bias_grad[c] = sum_r dpre[r, c] (the b1 gradient).
__kernel __attribute__((reqd_work_group_size(ROW_GROUP, 1, 1)))
void tanh_backward_columns(const int R, const int C, __global const double* dh, __global const double* mask,
                           __global const double* h, __global double* dpre, __global double* bias_grad) {
    __local double scratch[ROW_GROUP];
    const int c = get_group_id(0);
    const int l = get_local_id(0);
    double part = 0.0;
    for (int r = l; r < R; r += ROW_GROUP) {
        const int i = c * R + r;
        const double v = (dh[i] * mask[i]) * (1.0 - h[i] * h[i]);
        dpre[i] = v;
        part += v;
    }
    const double s = group_sum(scratch, part);
    if (l == 0) {
        bias_grad[c] = s;
    }
}

// Per row: dX = [base +] inv_std * ((dY - mean(dY)) - mean(dY * Y) * Y)
__kernel __attribute__((reqd_work_group_size(ROW_GROUP, 1, 1)))
void layer_norm_backward(const int R, const int C, __global const double* Y, __global const double* inv_std,
                         __global const double* dY, const int has_base, __global const double* base,
                         __global double* dX) {
    __local double scratch[ROW_GROUP];
    const int r = get_group_id(0);
    const int l = get_local_id(0);
    double part_grad = 0.0;
    double part_dot = 0.0;
    for (int c = l; c < C; c += ROW_GROUP) {
        const int i = c * R + r;
        part_grad += dY[i];
        part_dot += dY[i] * Y[i];
    }
    const double mean_grad = group_sum(scratch, part_grad) / (double)C;
    const double mean_grad_dot = group_sum(scratch, part_dot) / (double)C;
    const double s = inv_std[r];
    for (int c = l; c < C; c += ROW_GROUP) {
        const int i = c * R + r;
        const double v = s * ((dY[i] - mean_grad) - mean_grad_dot * Y[i]);
        dX[i] = has_base != 0 ? base[i] + v : v;
    }
}

// Per row: dS = A * (dA - sum_c dA * A)
__kernel void softmax_backward_rows(const int R, const int C, __global const double* A, __global const double* dA,
                                    __global double* dS) {
    const int r = get_global_id(0);
    if (r >= R) {
        return;
    }
    double dot = 0.0;
    for (int c = 0; c < C; ++c) {
        dot += dA[c * R + r] * A[c * R + r];
    }
    for (int c = 0; c < C; ++c) {
        const int i = c * R + r;
        dS[i] = A[i] * (dA[i] - dot);
    }
}

// Work-item (t, j): when step t is the first occurrence of token prev_t, gE[prev_t, j] = sum over the steps t' >= t
// with prev_t' = prev_t (in step order) of dU[t', j] * m1[t', j]; likewise gE2[prev2_t, j] with
// w2 * (dU * m2). Work-item (0, j) also writes gF[field, j] = sum_t dU[t, j] (F has nF rows). Rows of tokens
// absent from the pass keep the zeros they were cleared to.
__kernel void embedding_gradients(const int S, const int d, const int V, const double w2, const int nF, const int field,
                                  __global const int* prev, __global const int* prev2, __global const double* dU,
                                  __global const double* m1, __global const double* m2, __global double* gE,
                                  __global double* gE2, __global double* gF) {
    const int t = get_global_id(0);
    const int j = get_global_id(1);
    if (t >= S || j >= d) {
        return;
    }
    const int token = prev[t];
    const int token2 = prev2[t];
    bool first = true;
    bool first2 = true;
    for (int q = 0; q < t; ++q) {
        first = first && prev[q] != token;
        first2 = first2 && prev2[q] != token2;
    }
    if (first) {
        double s = 0.0;
        for (int q = t; q < S; ++q) {
            if (prev[q] == token) {
                s += dU[j * S + q] * m1[j * S + q];
            }
        }
        gE[j * V + token] = s;
    }
    if (first2) {
        double s = 0.0;
        for (int q = t; q < S; ++q) {
            if (prev2[q] == token2) {
                s += w2 * (dU[j * S + q] * m2[j * S + q]);
            }
        }
        gE2[j * V + token2] = s;
    }
    if (t == 0) {
        double field_sum = 0.0;
        for (int q = 0; q < S; ++q) {
            field_sum += dU[j * S + q];
        }
        gF[j * nF + field] = field_sum;
    }
}

// Adam over every packed parameter: m = b1 m + (1 - b1) g, v = b2 v + (1 - b2) g^2,
// p -= lr (m / c1) / (sqrt(v / c2) + eps), with the bias corrections applied as multiplications by inv_c1 = 1 / c1
// and inv_c2 = 1 / c2 (one division per entry instead of three; double-precision division is slow on GPUs);
// entries in [decay_begin, decay_end) are then multiplied by decay_factor (decoupled weight decay on W_out).
__kernel void adam_update(const int n, const double lr, const double inv_c1, const double inv_c2, const double b1,
                          const double b2, const double eps, const int decay_begin, const int decay_end,
                          const double decay_factor, __global double* p, __global const double* g,
                          __global double* m, __global double* v) {
    const int i = get_global_id(0);
    if (i >= n) {
        return;
    }
    const double gi = g[i];
    const double mi = b1 * m[i] + (1.0 - b1) * gi;
    const double vi = b2 * v[i] + (1.0 - b2) * (gi * gi);
    m[i] = mi;
    v[i] = vi;
    double pi = p[i] - (lr * (mi * inv_c1)) / (sqrt(vi * inv_c2) + eps);
    if (i >= decay_begin && i < decay_end) {
        pi *= decay_factor;
    }
    p[i] = pi;
}
)CLC";
    }
};

}  // namespace forward_analysis
