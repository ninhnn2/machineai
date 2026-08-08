# CUDA port of the PLE TinyLM. Same model.bin, same golden.txt as the ESP32 target.
#
#   make            # both binaries, ARCH tự dò từ GPU đang cắm
#   make ARCH=sm_87 # ép tay: Jetson Orin / Orin Nano Super / AGX Orin
#   make ARCH=sm_89 # RTX 4060 laptop, for developing before you touch the board
#
# On Jetson: export PATH=/usr/local/cuda/bin:$PATH
#
# Tự dò bằng `nvidia-smi --query-gpu=compute_cap`: hỏi thẳng GPU thay vì đoán
# theo tên board. compute_cap in ra dạng "8.7" -> sm_87. Nếu nvidia-smi không có
# (build chéo, container thiếu quyền) thì lùi về sm_87 = Orin, board phổ biến nhất.
DETECTED_CC := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.')
ifeq ($(DETECTED_CC),)
  ARCH ?= sm_87
else
  ARCH ?= sm_$(DETECTED_CC)
endif
NVCC  ?= nvcc
FLAGS  = -O3 -arch=$(ARCH) -lineinfo -Xcompiler -Wall

MODEL ?= ../model/model.bin
GOLD  ?= ../model/golden.txt
PEAK  ?= 66.8          # GB/s measured by jetson-optim/bench/bench_roofline.cu

TOKENS ?= 200
TEMP   ?= 0.8
TOPK   ?= 40

all: verify_cuda bench_cuda generate_cuda check_tok

verify_cuda: verify_cuda.cu llm_cuda.cuh ../common/llm.h
	$(NVCC) $(FLAGS) verify_cuda.cu -o $@

bench_cuda: bench_cuda.cu llm_cuda.cuh ../common/llm.h
	$(NVCC) $(FLAGS) bench_cuda.cu -o $@

# vocab.h: src/gen_assets.py --vocab <V> --out firmware/jetson/vocab.h
generate_cuda: generate_cuda.cu llm_cuda.cuh vocab.h bpe.h ../common/llm.h
	$(NVCC) $(FLAGS) generate_cuda.cu -o $@

# Tokenizer bằng C phải khớp Python, nếu không model nhận sai ids.
check_tok: check_tok.c vocab.h bpe.h
	$(CC) -O2 -Wall -o $@ check_tok.c

# Tier 1 of the validation chain: is the port correct?
verify: verify_cuda
	./verify_cuda $(MODEL) $(GOLD)

# Speed + per-stage roofline diagnosis.
bench: bench_cuda
	./bench_cuda $(MODEL) 200 $(PEAK)

# Xem model thực sự viết ra cái gì.
generate: generate_cuda
	./generate_cuda $(MODEL) -n $(TOKENS) -t $(TEMP) -k $(TOPK)

# Chế độ tương tác: gõ prompt, model viết tiếp.
chat: generate_cuda
	./generate_cuda $(MODEL) -i -n $(TOKENS) -t $(TEMP) -k $(TOPK)

# Tầng 0 của chuỗi kiểm chứng: tokenizer C == tokenizer Python?
tok: check_tok
	./check_tok tok_ref.txt

# The scalar C reference, for a like-for-like CPU number on the same board.
host_verify:
	$(CC) -O3 -o /tmp/verify_c ../host_verify/verify.c -lm && /tmp/verify_c $(MODEL) $(GOLD)

clean:
	rm -f verify_cuda bench_cuda generate_cuda check_tok

.PHONY: all verify bench generate chat tok host_verify clean
