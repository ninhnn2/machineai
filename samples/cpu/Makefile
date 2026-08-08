# Thang tối ưu matvec trên CPU. Chạy trên x86-64 (AVX2), ARM Linux/Jetson (NEON)
# và macOS Apple Silicon (NEON).
#
#   make            # tự dò kiến trúc + OpenMP
#   make run        # build + chạy với model.bin của repo
#   make ARCH_FLAGS="-march=native"    # ép cờ khác
#   make NO_OPENMP=1                   # ép build 1 luồng
#
# Trên Jetson (Cortex-A78AE có dotprod) cờ mặc định đã đúng, không cần chỉnh.

CC      ?= cc
UNAME_M := $(shell uname -m)
UNAME_S := $(shell uname -s)

ifeq ($(UNAME_M),x86_64)
  ARCH_FLAGS ?= -mavx2 -mfma
else ifeq ($(UNAME_M),aarch64)
  # ARM Linux (Jetson, Raspberry Pi 64-bit, server ARM).
  # +dotprod bật vdotq_s32; bỏ đi nếu CPU cũ không hỗ trợ
  ARCH_FLAGS ?= -march=armv8.2-a+dotprod
else ifeq ($(UNAME_M),arm64)
  # macOS Apple Silicon. `uname -m` ở đây là "arm64", KHÔNG phải "aarch64" --
  # nhánh trên không khớp, nên phải có nhánh riêng này.
  # Không cần cờ nào: NEON và __ARM_FEATURE_DOTPROD đã bật sẵn trên M1/M2/M3
  # (kiểm bằng: cc -E -dM -x c /dev/null | grep DOTPROD).
  ARCH_FLAGS ?=
else
  ARCH_FLAGS ?=
endif

# ---------------------------------------------------------------- OpenMP
# Apple clang KHÔNG nhận `-fopenmp` trần -- nó cần libomp cài riêng (Homebrew) và
# cờ `-Xpreprocessor`. gcc/clang trên Linux thì nhận thẳng. Dò bằng cách thử biên
# dịch thật, thay vì đoán theo tên hệ điều hành.
#
# matvec_ladder.c đã bọc `#ifdef _OPENMP` ở mọi chỗ dùng OpenMP, nên khi không có
# nó chương trình vẫn build và chạy đúng, chỉ là 1 luồng (bậc L4 mất phần lợi).
ifdef NO_OPENMP
  OMP_MODE := off
else ifeq ($(shell printf 'int main(){return 0;}' | $(CC) -fopenmp -x c - -o /dev/null 2>/dev/null && echo y),y)
  OMP_MODE  := native
  OMP_FLAGS := -fopenmp
else ifneq ($(wildcard $(shell brew --prefix libomp 2>/dev/null)/include/omp.h),)
  LIBOMP    := $(shell brew --prefix libomp)
  OMP_MODE  := libomp
  OMP_FLAGS := -Xpreprocessor -fopenmp -I$(LIBOMP)/include
  OMP_LIBS  := -L$(LIBOMP)/lib -lomp
else
  OMP_MODE := off
endif

CFLAGS  = -O3 -Wall -Wextra $(ARCH_FLAGS) $(OMP_FLAGS)
LDFLAGS = -lm $(OMP_LIBS)

MODEL ?= ../../firmware/model/model.bin
REPS  ?= 200

all: banner matvec_ladder numerics

# In ra cấu hình đã dò được, để khi số đo lệch nhau giữa hai máy còn biết vì sao.
banner:
	@echo "cpu=$(UNAME_M) ($(UNAME_S))  arch_flags='$(ARCH_FLAGS)'  openmp=$(OMP_MODE)"
ifeq ($(OMP_MODE),off)
	@echo "  ! không có OpenMP -> build 1 luồng; bậc 'L4 + đa luồng' sẽ không nhanh lên."
ifeq ($(UNAME_S),Darwin)
	@echo "  ! macOS: chạy 'brew install libomp' rồi make lại để lấy lại đa luồng."
endif
endif

matvec_ladder: matvec_ladder.c ../../firmware/common/llm.h
	$(CC) $(CFLAGS) matvec_ladder.c -o $@ $(LDFLAGS)

numerics: numerics.c ../../firmware/common/llm.h
	$(CC) $(CFLAGS) numerics.c -o $@ $(LDFLAGS)

run: matvec_ladder
	./matvec_ladder $(MODEL) $(REPS)

# Kiểm chứng lý thuyết số học trong docs/10
verify-theory: numerics
	./numerics $(MODEL)

# So 1 luồng với đa luồng, để tách phần lợi của SIMD khỏi phần lợi của threads.
scaling: matvec_ladder
	@for n in 1 2 4 8; do \
	  echo "--- $$n luồng ---"; \
	  ./matvec_ladder $(MODEL) $(REPS) $$n 2>/dev/null | grep -E "L4"; \
	done

clean:
	rm -f matvec_ladder numerics

.PHONY: all banner run scaling verify-theory clean
