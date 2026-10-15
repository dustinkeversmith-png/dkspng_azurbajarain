#pragma once

#if FORWARD_ANALYSIS_WITH_OPENCL

#include "forward_analysis/compute/decoder_kernel_source.hpp"
#include "forward_analysis/compute/opencl_runtime.hpp"

#include <clblast.h>

#include <CL/cl.h>

#include <cstddef>
#include <stdexcept>
#include <string>

namespace forward_analysis {

// Double-precision GEMM on the runtime's queue, column-major like Eigen:
// C (m x n) = alpha op(A) op(B) + beta C, where op transposes when requested and ld* are the stored row counts.
// Three implementations, chosen by shape:
//   m n <= kReduceMaximumOutputs                          gemm_reduce (one work-group per output element)
//   k <= kDirectMaximumDepth and m n k <= kDirectMaximumWork  gemm_direct (one work-item per output element)
//   otherwise                                              CLBlast
// Measured on an RTX 4070: CLBlast's kernels leave most of the device idle when C has few entries or the inner
// dimension is short (about 20 us for 41 x 8 x 256 against 9 us for gemm_reduce, 18 us for 41 x 256 x 8 against
// 11 us for gemm_direct), and every CLBlast call also costs 15-20 us of host time.
class DeviceGemm {
public:
    static constexpr std::size_t kReduceMaximumOutputs = 1024;
    static constexpr std::size_t kDirectMaximumDepth = 64;
    static constexpr std::size_t kDirectMaximumWork = std::size_t{1} << 20;

    static void multiply(const OpenClRuntime& runtime,
                         bool transpose_a,
                         bool transpose_b,
                         std::size_t m,
                         std::size_t n,
                         std::size_t k,
                         double alpha,
                         cl_mem a,
                         std::size_t lda,
                         cl_mem b,
                         std::size_t ldb,
                         double beta,
                         cl_mem c,
                         std::size_t ldc) {
        const int trans_a = transpose_a ? 1 : 0;
        const int trans_b = transpose_b ? 1 : 0;
        if (m * n <= kReduceMaximumOutputs) {
            runtime.launch_groups("gemm_reduce", m * n, 1, DecoderKernelSource::kRowGroup, 1, m, n, k, trans_a, trans_b,
                                  alpha, a, lda, b, ldb, beta, c, ldc);
            return;
        }
        if (k <= kDirectMaximumDepth && m * n * k <= kDirectMaximumWork) {
            runtime.launch("gemm_direct", m, n, m, n, k, trans_a, trans_b, alpha, a, lda, b, ldb, beta, c, ldc);
            return;
        }
        cl_command_queue queue = runtime.queue();
        const clblast::StatusCode status = clblast::Gemm<double>(
            clblast::Layout::kColMajor, transpose_a ? clblast::Transpose::kYes : clblast::Transpose::kNo,
            transpose_b ? clblast::Transpose::kYes : clblast::Transpose::kNo, m, n, k, alpha, a, 0, lda, b, 0, ldb, beta, c,
            0, ldc, &queue, nullptr);
        if (status != clblast::StatusCode::kSuccess) {
            throw std::runtime_error("CLBlast Gemm failed with status " + std::to_string(static_cast<int>(status)));
        }
    }
};

}  // namespace forward_analysis

#endif
