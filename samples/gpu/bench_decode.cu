// bench_decode.cu — thí nghiệm cốt lõi: vì sao decode chậm còn prefill nhanh.
//
// Cùng MỘT phép nhân ma trận (N×K), chỉ đổi M = số token xử lý cùng lúc:
//   M = 1     -> đúng là decode (sinh 1 token)
//   M = 512   -> đúng là prefill (nạp prompt 512 token)
//
// Quan sát cần rút ra: thời gian gần như KHÔNG đổi từ M=1 đến M≈32-64.
// Nghĩa là 63 token đầu tiên gần như MIỄN PHÍ. Đó là toàn bộ cơ sở của
// batching, speculative decoding và continuous batching.
//
//   nvcc -O3 -arch=sm_87 bench_decode.cu -o bench_decode -lcublas && ./bench_decode
//
// Tham khảo: Pope et al., "Efficiently Scaling Transformer Inference",
// MLSys 2023 (§3 phân tích memory- vs compute-bound của decode).

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>

#define CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA error: %s (%s:%d)\n", cudaGetErrorString(e), __FILE__, __LINE__); \
    exit(1); } } while (0)

#define CUB(x) do { cublasStatus_t s = (x); if (s != CUBLAS_STATUS_SUCCESS) { \
    fprintf(stderr, "cuBLAS error %d (%s:%d)\n", (int)s, __FILE__, __LINE__); exit(1); } } while (0)

template <typename F>
static double bench(F f, int iters, int warmup = 5) {
    cudaEvent_t s, e;
    cudaEventCreate(&s); cudaEventCreate(&e);
    for (int i = 0; i < warmup; i++) f();
    CHECK(cudaDeviceSynchronize());
    cudaEventRecord(s);
    for (int i = 0; i < iters; i++) f();
    cudaEventRecord(e);
    cudaEventSynchronize(e);
    float ms; cudaEventElapsedTime(&ms, s, e);
    cudaEventDestroy(s); cudaEventDestroy(e);
    return ms / iters;
}

int main(int argc, char** argv) {
    int N = (argc > 1) ? atoi(argv[1]) : 4096;   // ~ d_model của Llama-8B
    int K = (argc > 2) ? atoi(argv[2]) : 4096;

    cudaDeviceProp p; CHECK(cudaGetDeviceProperties(&p, 0));
    cublasHandle_t h; CUB(cublasCreate(&h));
    CUB(cublasSetMathMode(h, CUBLAS_TENSOR_OP_MATH));

    printf("=========================================================================\n");
    printf(" %s   |  GEMM  M x %d x %d  (đổi M = số token xử lý cùng lúc)\n", p.name, N, K);
    printf("=========================================================================\n");

    const int MMAX = 4096;
    __half *dA, *dB, *dC;
    CHECK(cudaMalloc(&dA, (size_t)MMAX * K * sizeof(__half)));
    CHECK(cudaMalloc(&dB, (size_t)K * N * sizeof(__half)));
    CHECK(cudaMalloc(&dC, (size_t)MMAX * N * sizeof(__half)));
    CHECK(cudaMemset(dA, 0x11, (size_t)MMAX * K * sizeof(__half)));
    CHECK(cudaMemset(dB, 0x11, (size_t)K * N * sizeof(__half)));

    float alpha = 1.f, beta = 0.f;

    printf("\n[FP16 tensor core]\n");
    printf("%6s %10s %10s %10s %9s   %s\n", "M", "ms", "TFLOP/s", "GB/s", "AI", "đây là gì");
    printf("-------------------------------------------------------------------------\n");

    double peak_tf = 0, ms_at_1 = 0;
    int Ms[] = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096};

    for (int mi = 0; mi < (int)(sizeof(Ms) / sizeof(int)); mi++) {
        int M = Ms[mi];
        int iters = (M >= 1024) ? 10 : 50;
        double ms = bench([&] {
            cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
                         &alpha, dB, CUDA_R_16F, N, dA, CUDA_R_16F, K,
                         &beta,  dC, CUDA_R_16F, N,
                         CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        }, iters);

        double flops = 2.0 * M * N * K;
        double bytes = ((double)M * K + (double)K * N + (double)M * N) * 2.0;
        double tf = flops / (ms * 1e-3) / 1e12;
        double gb = bytes / (ms * 1e-3) / 1e9;
        if (tf > peak_tf) peak_tf = tf;
        if (M == 1) ms_at_1 = ms;

        const char* what = (M == 1) ? "DECODE (1 token)"
                         : (M <= 64) ? "vẫn ~= giá của 1 token!"
                         : (M >= 512) ? "PREFILL" : "chuyển tiếp";
        printf("%6d %10.3f %10.2f %10.1f %9.1f   %s\n", M, ms, tf, gb, flops / bytes, what);
    }

    // ---- INT8 ----
    printf("\n[INT8 tensor core]  (cùng phép toán, weights + activations 8-bit)\n");
    int8_t *iA, *iB; int32_t* iC;
    CHECK(cudaMalloc(&iA, (size_t)MMAX * K));
    CHECK(cudaMalloc(&iB, (size_t)K * N));
    CHECK(cudaMalloc(&iC, (size_t)MMAX * N * sizeof(int32_t)));
    CHECK(cudaMemset(iA, 1, (size_t)MMAX * K));
    CHECK(cudaMemset(iB, 1, (size_t)K * N));
    int32_t ia = 1, ib = 0;

    printf("%6s %10s %10s %10s   %s\n", "M", "ms", "TOPS", "GB/s", "");
    printf("-------------------------------------------------------------------------\n");
    double peak_tops = 0;
    int Mi[] = {1, 8, 64, 512, 4096};
    for (int mi = 0; mi < 5; mi++) {
        int M = Mi[mi];
        cublasStatus_t st = cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K,
                                         &ia, iB, CUDA_R_8I, K, iA, CUDA_R_8I, K,
                                         &ib, iC, CUDA_R_32I, N,
                                         CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
        if (st != CUBLAS_STATUS_SUCCESS) {
            printf("%6d   cuBLAS không hỗ trợ cấu hình INT8 này (status %d)\n", M, (int)st);
            continue;
        }
        double ms = bench([&] {
            cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K,
                         &ia, iB, CUDA_R_8I, K, iA, CUDA_R_8I, K,
                         &ib, iC, CUDA_R_32I, N,
                         CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
        }, M >= 1024 ? 10 : 50);
        double ops = 2.0 * M * N * K;
        double bytes = (double)M * K + (double)K * N + (double)M * N * 4.0;
        double tops = ops / (ms * 1e-3) / 1e12;
        if (tops > peak_tops) peak_tops = tops;
        printf("%6d %10.3f %10.2f %10.1f\n", M, ms, tops, bytes / (ms * 1e-3) / 1e9);
    }

    printf("\n=========================================================================\n");
    printf(" ĐỌC BẢNG TRÊN\n");
    printf("   FP16 đỉnh (M lớn)     %6.2f TFLOP/s\n", peak_tf);
    if (peak_tops > 0) printf("   INT8 đỉnh (M lớn)     %6.2f TOPS\n", peak_tops);
    printf("   M=1 (decode) đạt      %6.2f TFLOP/s = %.1f%% của đỉnh\n",
           2.0 * 1 * N * K / (ms_at_1 * 1e-3) / 1e12,
           100.0 * (2.0 * N * K / (ms_at_1 * 1e-3) / 1e12) / peak_tf);
    printf("\n   Ở M=1, tensor core NHÀN RỖI gần như hoàn toàn. GPU chỉ đang\n");
    printf("   chờ đọc ma trận B (%.1f MB) từ DRAM. Đó chính là LLM decode.\n",
           (double)K * N * 2 / 1e6);
    printf("\n   Vì thời gian gần như không đổi tới M~32-64, xử lý 32 token một lúc\n");
    printf("   gần như miễn phí -> đây là lý do speculative decoding hoạt động,\n");
    printf("   và vì sao server dùng continuous batching.\n");
    printf("=========================================================================\n");

    cublasDestroy(h);
    return 0;
}
