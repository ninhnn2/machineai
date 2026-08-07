// bench_roofline.cu — dựng đường roofline THỰC NGHIỆM của board.
//
// Hai phần:
//   A. STREAM-like: đo băng thông DRAM đạt được (read / copy / scale / triad)
//   B. AI sweep   : cùng một kernel, tăng dần số FLOP làm trên mỗi byte đọc,
//                   quan sát máy đi từ memory-bound sang compute-bound.
//
// Phần B là thí nghiệm quan trọng: nó vẽ ra đường roofline mà không cần tin
// bất kỳ con số marketing nào. Điểm gãy của đường cong CHÍNH LÀ machine balance.
//
//   nvcc -O3 -arch=sm_87 bench_roofline.cu -o bench_roofline && ./bench_roofline
//
// Tham chiếu lý thuyết: Williams, Waterman, Patterson, "Roofline: An Insightful
// Visual Performance Model for Multicore Architectures", CACM 52(4), 2009.

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA error: %s (%s:%d)\n", cudaGetErrorString(e), __FILE__, __LINE__); \
    exit(1); } } while (0)

// ---------------------------------------------------------------- STREAM

__global__ void k_read(const float* __restrict__ a, float* out, size_t n) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    float s = 0.f;
    for (; i < n; i += stride) s += a[i];
    if (s == 1.2345678e30f) out[0] = s;   // chặn compiler loại bỏ vòng lặp
}

__global__ void k_copy(const float* __restrict__ a, float* __restrict__ b, size_t n) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (; i < n; i += stride) b[i] = a[i];
}

__global__ void k_scale(const float* __restrict__ a, float* __restrict__ b, size_t n, float q) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (; i < n; i += stride) b[i] = q * a[i];
}

__global__ void k_triad(const float* __restrict__ a, const float* __restrict__ b,
                        float* __restrict__ c, size_t n, float q) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (; i < n; i += stride) c[i] = a[i] + q * b[i];
}

// ---- phiên bản VECTOR HOÁ (float4) ----
// Load scalar 4 byte KHÔNG bão hoà được DRAM controller: mỗi thread chỉ có 1
// yêu cầu bộ nhớ đang bay, không đủ để che độ trễ. float4 = 16 byte/lệnh cho
// 4x số byte đang bay với cùng số thread. Đây là khác biệt 2-4x trên Tegra.
// (CUDA Best Practices Guide §9.2.1 "Coalesced Access"; Luitjens, "CUDA Pro Tip:
//  Increase Performance with Vectorized Memory Access", NVIDIA Developer Blog.)

__global__ void k_read4(const float4* __restrict__ a, float* out, size_t n4) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    float4 s = make_float4(0.f, 0.f, 0.f, 0.f);
    for (; i < n4; i += stride) {
        float4 v = a[i];
        s.x += v.x; s.y += v.y; s.z += v.z; s.w += v.w;
    }
    if (s.x + s.y + s.z + s.w == 1.2345678e30f) out[0] = s.x;
}

__global__ void k_copy4(const float4* __restrict__ a, float4* __restrict__ b, size_t n4) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (; i < n4; i += stride) b[i] = a[i];
}

__global__ void k_triad4(const float4* __restrict__ a, const float4* __restrict__ b,
                         float4* __restrict__ c, size_t n4, float q) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (; i < n4; i += stride) {
        float4 x = a[i], y = b[i];
        c[i] = make_float4(x.x + q * y.x, x.y + q * y.y, x.z + q * y.z, x.w + q * y.w);
    }
}

// Đọc int8 — đúng pattern đọc weights đã quantize trong LLM decode.
__global__ void k_read_i8(const char* __restrict__ a, int* out, size_t n) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    int s = 0;
    for (; i < n / 4; i += stride) {          // đọc 4 byte/lần cho coalescing tốt
        int v = reinterpret_cast<const int*>(a)[i];
        s += (v & 0xff) + ((v >> 8) & 0xff) + ((v >> 16) & 0xff) + ((v >> 24) & 0xff);
    }
    if (s == 0x7fffffff) out[0] = s;
}

// ---------------------------------------------------------------- AI sweep
//
// Mỗi phần tử: đọc 4 byte, ghi 4 byte = 8 byte.
// FLOP: NF vòng × 4 FMA × 2 flop = 8·NF, cộng 3 phép cộng cuối.
// => arithmetic intensity ≈ 8·NF / 8 = NF FLOP/byte.  Tiện: AI = NF.
//
// 4 accumulator độc lập để không bị nghẽn bởi độ trễ phụ thuộc của FMA
// (nếu dùng 1 chuỗi phụ thuộc, ta đo độ trễ chứ không đo thông lượng).
template <int NF>
__global__ void k_ai(const float* __restrict__ in, float* __restrict__ out, size_t n, float a) {
    size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = idx; i < n; i += stride) {
        float x = in[i];
        float r0 = x, r1 = x * 1.01f, r2 = x * 1.02f, r3 = x * 1.03f;
#pragma unroll
        for (int k = 0; k < NF; k++) {
            r0 = fmaf(r0, a, x); r1 = fmaf(r1, a, x);
            r2 = fmaf(r2, a, x); r3 = fmaf(r3, a, x);
        }
        out[i] = r0 + r1 + r2 + r3;
    }
}

// ---------------------------------------------------------------- tiện ích

struct Timer {
    cudaEvent_t s, e;
    Timer()  { cudaEventCreate(&s); cudaEventCreate(&e); }
    ~Timer() { cudaEventDestroy(s); cudaEventDestroy(e); }
    void start() { cudaEventRecord(s); }
    float stop() { cudaEventRecord(e); cudaEventSynchronize(e);
                   float ms; cudaEventElapsedTime(&ms, s, e); return ms; }
};

static int g_blocks = 0, g_threads = 256;

template <typename F>
static float bench(F f, int iters = 30, int warmup = 5) {
    Timer t;
    for (int i = 0; i < warmup; i++) f();
    CHECK(cudaDeviceSynchronize());
    t.start();
    for (int i = 0; i < iters; i++) f();
    float ms = t.stop();
    CHECK(cudaGetLastError());
    return ms / iters;
}

double g_peak_bw = 0.0, g_peak_flops = 0.0;

int main(int argc, char** argv) {
    size_t MB = (argc > 1) ? atoll(argv[1]) : 256;
    size_t n = MB * 1024 * 1024 / sizeof(float);
    size_t bytes = n * sizeof(float);

    cudaDeviceProp p;
    CHECK(cudaGetDeviceProperties(&p, 0));
    g_blocks = p.multiProcessorCount * 32;

    printf("=========================================================================\n");
    printf(" %s\n", p.name);
    printf(" SM %d.%d | %d SM | clock %.0f MHz | L2 %.1f MB | RAM %.2f GB\n",
           p.major, p.minor, p.multiProcessorCount, p.clockRate / 1000.0,
           p.l2CacheSize / 1048576.0, p.totalGlobalMem / 1e9);
    printf(" mem clock %.0f MHz | bus %d bit | peak lý thuyết %.1f GB/s\n",
           p.memoryClockRate / 1000.0, p.memoryBusWidth,
           2.0 * p.memoryClockRate * 1000.0 * (p.memoryBusWidth / 8) / 1e9);
    printf(" buffer %zu MB (>> L2, nên số đo là DRAM thật)\n", MB);
    printf("=========================================================================\n");

    float *a, *b, *c;
    CHECK(cudaMalloc(&a, bytes));
    CHECK(cudaMalloc(&b, bytes));
    CHECK(cudaMalloc(&c, bytes));
    CHECK(cudaMemset(a, 1, bytes));
    CHECK(cudaMemset(b, 2, bytes));

    // ---- A. băng thông ----
    printf("\n[A] BĂNG THÔNG DRAM  (STREAM-like)\n");
    printf("%-28s %10s %10s   %s\n", "phép đo", "ms", "GB/s", "ghi chú");
    printf("-------------------------------------------------------------------------\n");

    struct { const char* name; double bytes_moved; float ms; const char* note; } r[8];
    int ri = 0;

    r[ri].ms = bench([&]{ k_read<<<g_blocks,g_threads>>>(a, c, n); });
    r[ri].name = "read (1R)"; r[ri].bytes_moved = bytes;
    r[ri].note = "pattern của LLM decode"; ri++;

    r[ri].ms = bench([&]{ k_copy<<<g_blocks,g_threads>>>(a, b, n); });
    r[ri].name = "copy (1R+1W)"; r[ri].bytes_moved = 2.0 * bytes;
    r[ri].note = "sát peak DRAM nhất"; ri++;

    r[ri].ms = bench([&]{ k_scale<<<g_blocks,g_threads>>>(a, b, n, 3.0f); });
    r[ri].name = "scale (1R+1W)"; r[ri].bytes_moved = 2.0 * bytes; r[ri].note = ""; ri++;

    r[ri].ms = bench([&]{ k_triad<<<g_blocks,g_threads>>>(a, b, c, n, 3.0f); });
    r[ri].name = "triad (2R+1W)"; r[ri].bytes_moved = 3.0 * bytes; r[ri].note = ""; ri++;

    r[ri].ms = bench([&]{ k_read_i8<<<g_blocks,g_threads>>>((const char*)a, (int*)c, bytes); });
    r[ri].name = "int8 read"; r[ri].bytes_moved = bytes;
    r[ri].note = "đọc weights INT8/INT4"; ri++;

    size_t n4 = n / 4;
    r[ri].ms = bench([&]{ k_read4<<<g_blocks,g_threads>>>((const float4*)a, c, n4); });
    r[ri].name = "read  VECTOR float4"; r[ri].bytes_moved = bytes;
    r[ri].note = "<< so cái này với read scalar"; ri++;

    r[ri].ms = bench([&]{ k_copy4<<<g_blocks,g_threads>>>((const float4*)a, (float4*)b, n4); });
    r[ri].name = "copy  VECTOR float4"; r[ri].bytes_moved = 2.0 * bytes;
    r[ri].note = "thường là số cao nhất"; ri++;

    r[ri].ms = bench([&]{ k_triad4<<<g_blocks,g_threads>>>((const float4*)a, (const float4*)b,
                                                           (float4*)c, n4, 3.0f); });
    r[ri].name = "triad VECTOR float4"; r[ri].bytes_moved = 3.0 * bytes;
    r[ri].note = ""; ri++;

    for (int i = 0; i < ri; i++) {
        double gbs = r[i].bytes_moved / (r[i].ms * 1e-3) / 1e9;
        if (gbs > g_peak_bw) g_peak_bw = gbs;
        printf("%-28s %10.3f %10.1f   %s\n", r[i].name, r[i].ms, gbs, r[i].note);
    }

    // ---- B. quét arithmetic intensity ----
    printf("\n[B] QUÉT ARITHMETIC INTENSITY  (cùng kernel, tăng FLOP/byte)\n");
    printf("%6s %10s %10s %12s   %s\n", "AI", "ms", "GB/s", "GFLOP/s", "chế độ");
    printf("-------------------------------------------------------------------------\n");

    // n nhỏ hơn cho AI cao, tránh chờ quá lâu
    size_t n2 = n / 4;
    double res_ai[16], res_bw[16], res_fl[16];
    int nres = 0;

    auto run = [&](int NF, float ms) {
        double B = 2.0 * n2 * sizeof(float);          // 1 đọc + 1 ghi
        double F = (double)n2 * (8.0 * NF + 3.0);
        double gbs = B / (ms * 1e-3) / 1e9;
        double gfl = F / (ms * 1e-3) / 1e9;
        if (gfl > g_peak_flops) g_peak_flops = gfl;
        res_ai[nres] = F / B; res_bw[nres] = gbs; res_fl[nres] = gfl; nres++;
        printf("%6.0f %10.3f %10.1f %12.1f   %s\n", F / B, ms, gbs, gfl,
               gbs > 0.8 * g_peak_bw ? "MEMORY-bound (chạm trần BW)"
                                     : (gfl > 0.8 * g_peak_flops ? "COMPUTE-bound" : "chuyển tiếp"));
    };

#define SWEEP(NF) run(NF, bench([&]{ k_ai<NF><<<g_blocks,g_threads>>>(a, b, n2, 1.000001f); }, 20))
    SWEEP(1); SWEEP(2); SWEEP(4); SWEEP(8); SWEEP(16);
    SWEEP(32); SWEEP(64); SWEEP(128); SWEEP(256);
#undef SWEEP

    // ---- kết luận ----
    double balance = g_peak_flops * 1e9 / (g_peak_bw * 1e9);
    printf("\n=========================================================================\n");
    printf(" KẾT QUẢ ĐO ĐƯỢC\n");
    printf("   băng thông đỉnh      %8.1f GB/s   <- dùng số này cho roofline.py\n", g_peak_bw);
    printf("   FP32 đỉnh            %8.1f GFLOP/s\n", g_peak_flops);
    printf("   MACHINE BALANCE      %8.1f FLOP/byte\n", balance);
    printf("\n   LƯU Ý: %.0f là balance của CUDA core FP32. LLM chạy trên TENSOR CORE,\n", balance);
    printf("   nhanh hơn ~17x FP32 -> balance thực tế cho LLM là ~%.0f FLOP/byte.\n",
           balance * 17);
    printf("   => LLM decode FP16 có AI = 8, INT8 = 16, INT4 = 32\n");
    printf("      -> memory-bound %.0fx / %.0fx / %.0fx so với tensor core.\n",
           balance * 17 / 8, balance * 17 / 16, balance * 17 / 32);
    printf("      -> tối ưu compute là VÔ ÍCH; phải giảm bytes đọc.\n");
    printf("      (chạy ./bench_decode để đo peak tensor core thật)\n");
    printf("\n   TRẦN DECODE trên board này:  tok/s = %.0f / (model_GB)\n", g_peak_bw);
    printf("     Llama-3.2-1B Q4 (0.7 GB) -> %5.1f tok/s\n", g_peak_bw / 0.7);
    printf("     Llama-3.2-3B Q4 (1.9 GB) -> %5.1f tok/s\n", g_peak_bw / 1.9);
    printf("     Llama-3.1-8B Q4 (4.8 GB) -> %5.1f tok/s\n", g_peak_bw / 4.8);
    printf("=========================================================================\n");

    cudaFree(a); cudaFree(b); cudaFree(c);
    return 0;
}
