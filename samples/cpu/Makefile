# Thang tối ưu matvec trên CPU. Chạy được cả x86 (AVX2) và ARM (NEON).
#
#   make            # tự chọn cờ theo kiến trúc
#   make run        # build + chạy với model.bin của repo
#   make ARCH_FLAGS="-march=native"    # ép cờ khác
#
# Trên Jetson (Cortex-A78AE có dotprod):
#   make ARCH_FLAGS="-march=armv8.2-a+dotprod"

CC      ?= cc
UNAME_M := $(shell uname -m)

ifeq ($(UNAME_M),x86_64)
  ARCH_FLAGS ?= -mavx2 -mfma
else ifeq ($(UNAME_M),aarch64)
  # +dotprod bật vdotq_s32; bỏ đi nếu CPU cũ không hỗ trợ
  ARCH_FLAGS ?= -march=armv8.2-a+dotprod
else
  ARCH_FLAGS ?=
endif

CFLAGS  = -O3 -Wall -Wextra $(ARCH_FLAGS) -fopenmp
LDFLAGS = -lm

MODEL ?= ../../firmware/model/model.bin
REPS  ?= 200

all: matvec_ladder numerics

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

.PHONY: all run scaling verify-theory clean
