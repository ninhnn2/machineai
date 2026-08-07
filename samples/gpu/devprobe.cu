// Dump mọi thông số kiến trúc GPU mà việc tối ưu thực sự phụ thuộc vào, cộng với
// vài phép đo trực tiếp. Dùng cái này thay vì tra datasheet: datasheet nói về
// dòng sản phẩm, còn bạn tối ưu cho đúng một con chip.
//
//   nvcc -O3 -arch=sm_87 devprobe.cu -o devprobe && ./devprobe
//
// Đọc kèm: docs/11-toi-uu-nvidia.md

#include <cstdio>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){ \
  printf("CUDA %s @%d\n", cudaGetErrorString(e), __LINE__); return 1; } } while(0)

// Kernel dùng để đo occupancy thật theo số thanh ghi. __launch_bounds__ ép
// compiler giới hạn thanh ghi; bỏ đi thì nó tự chọn và occupancy có thể tụt.
__global__ void k_probe(float *o, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  float a = i * 1.0f, b = a + 1, c = a + 2, d = a + 3;
  for (int k = 0; k < 8; k++) { a = fmaf(a,b,c); b = fmaf(b,c,d); c = fmaf(c,d,a); d = fmaf(d,a,b); }
  if (i < n) o[i] = a + b + c + d;
}

int main() {
  int dev = 0;
  cudaDeviceProp p;
  CK(cudaGetDeviceProperties(&p, dev));

  int rt = 0, drv = 0;
  cudaRuntimeGetVersion(&rt);
  cudaDriverGetVersion(&drv);

  printf("========================================================================\n");
  printf(" %s   (compute capability %d.%d = sm_%d%d)\n",
         p.name, p.major, p.minor, p.major, p.minor);
  printf(" CUDA runtime %d.%d | driver %d.%d\n",
         rt/1000, (rt%1000)/10, drv/1000, (drv%1000)/10);
  printf("========================================================================\n");

  printf("\n[ THỰC THI ]\n");
  printf("  SM (multiprocessor)          %d\n", p.multiProcessorCount);
  printf("  warp size                    %d\n", p.warpSize);
  printf("  max thread / block           %d\n", p.maxThreadsPerBlock);
  printf("  max thread / SM              %d   -> %d warp/SM tối đa\n",
         p.maxThreadsPerMultiProcessor, p.maxThreadsPerMultiProcessor / p.warpSize);
  printf("  max block / SM               %d\n", p.maxBlocksPerMultiProcessor);
  printf("  thanh ghi / block            %d\n", p.regsPerBlock);
  printf("  thanh ghi / SM               %d\n", p.regsPerMultiprocessor);
  printf("  clock GPU                    %.0f MHz\n", p.clockRate / 1000.0);
  printf("  grid tối đa                  %d x %d x %d\n",
         p.maxGridSize[0], p.maxGridSize[1], p.maxGridSize[2]);

  printf("\n[ BỘ NHỚ ]\n");
  printf("  global                       %.2f GB\n", p.totalGlobalMem / 1e9);
  printf("  L2 cache                     %.2f MB\n", p.l2CacheSize / 1048576.0);
  printf("  shared / block (mặc định)    %zu KB\n", p.sharedMemPerBlock / 1024);
  printf("  shared / block (tối đa opt-in) %zu KB\n", p.sharedMemPerBlockOptin / 1024);
  printf("  shared / SM                  %zu KB\n", p.sharedMemPerMultiprocessor / 1024);
  printf("  constant                     %zu KB\n", p.totalConstMem / 1024);
  printf("  bus rộng                     %d bit\n", p.memoryBusWidth);
  printf("  memory clock (báo cáo)       %.0f MHz  <-- KHÔNG tin trên Tegra\n",
         p.memoryClockRate / 1000.0);

  printf("\n[ TÍNH CHẤT QUYẾT ĐỊNH CÁCH VIẾT KERNEL ]\n");
  printf("  unified addressing           %s\n", p.unifiedAddressing ? "CÓ" : "không");
  printf("  managed memory               %s\n", p.managedMemory ? "CÓ" : "không");
  printf("  bộ nhớ dùng chung với host   %s   <-- Jetson: KHÔNG cần memcpy H2D\n",
         p.integrated ? "CÓ (iGPU)" : "không (dGPU)");
  printf("  map host memory              %s\n", p.canMapHostMemory ? "CÓ" : "không");
  printf("  concurrent kernels           %s\n", p.concurrentKernels ? "CÓ" : "không");
  printf("  async engine                 %d\n", p.asyncEngineCount);
  printf("  cooperative launch           %s\n", p.cooperativeLaunch ? "CÓ" : "không");

  printf("\n[ ĐỈNH LÝ THUYẾT (suy từ thông số trên) ]\n");
  // Ampere GA10x: 128 FP32 lane/SM. GA100 cũng 128 nhưng khác cách chia.
  int fp32_per_sm = (p.major == 8) ? 128 : (p.major == 7 ? 64 : 64);
  double ghz = p.clockRate / 1e6;
  double fp32_tflops = p.multiProcessorCount * fp32_per_sm * 2.0 * ghz / 1000.0;
  printf("  CUDA core (ước %d/SM)       %d\n", fp32_per_sm,
         p.multiProcessorCount * fp32_per_sm);
  printf("  FP32 đỉnh                    %.2f TFLOP/s  (%d SM x %d x 2 x %.2f GHz)\n",
         fp32_tflops, p.multiProcessorCount, fp32_per_sm, ghz);
  printf("  Tensor Core (4/SM Ampere)    %d\n", p.multiProcessorCount * 4);
  printf("  FP16 tensor đỉnh ~           %.2f TFLOP/s   (8x FP32 trên Ampere)\n",
         fp32_tflops * 8);
  printf("  INT8 tensor đỉnh ~           %.2f TOPS      (16x)\n", fp32_tflops * 16);
  printf("  ! Đây là ĐỈNH. Đo thật bằng samples/gpu/bench_decode.cu.\n");

  printf("\n[ OCCUPANCY THẬT của k_probe ]\n");
  int reg = 0, maxblocks = 0;
  cudaFuncAttributes fa;
  CK(cudaFuncGetAttributes(&fa, k_probe));
  reg = fa.numRegs;
  printf("  thanh ghi/thread             %d\n", reg);
  printf("  shared tĩnh                  %zu B\n", fa.sharedSizeBytes);
  printf("  %-8s %-10s %-12s %s\n", "block", "block/SM", "warp/SM", "occupancy");
  for (int bs = 32; bs <= 1024; bs *= 2) {
    CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxblocks, k_probe, bs, 0));
    int warps = maxblocks * bs / p.warpSize;
    double occ = 100.0 * warps / (p.maxThreadsPerMultiProcessor / p.warpSize);
    printf("  %-8d %-10d %-12d %.0f%%%s\n", bs, maxblocks, warps, occ,
           occ >= 100 ? "  <- bão hoà" : "");
  }
  printf("  Occupancy CAO không tự động = NHANH. Nó là khả năng CHE ĐỘ TRỄ;\n");
  printf("  kernel memory-bound cần cao, kernel dùng nhiều thanh ghi thì thấp vẫn tốt.\n");

  printf("\n[ CẢNH BÁO CHO TEGRA ]\n");
  if (p.integrated) {
    printf("  iGPU: RAM dùng CHUNG với CPU.\n");
    printf("   - cudaMemcpy H2D là copy RAM->RAM, ĐỐT băng thông 2 lần\n");
    printf("   - dùng cudaHostAlloc(cudaHostAllocMapped) hoặc cudaMallocManaged\n");
    printf("   - memoryClockRate ở trên KHÔNG phải EMC thật, đọc bằng:\n");
    printf("     sudo cat /sys/kernel/debug/bpmp/debug/clk/emc/rate\n");
    printf("   - nvpmodel/jetson_clocks quyết định clock; MAXN_SUPER = mode 2\n");
  } else {
    printf("  dGPU: VRAM riêng, memcpy H2D qua PCIe là thật và tốn kém.\n");
  }
  return 0;
}
