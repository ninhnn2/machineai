# Bản C scalar chạy trên CPU host -- tầng kiểm chứng chung của MỌI target.
# Cùng llm.h, cùng model.bin, cùng golden.txt như ESP32 và Jetson; chỉ khác chỗ
# nó build được ở bất cứ đâu có trình biên dịch C.
#
#   make verify     # C có khớp golden của PyTorch không?  (câu hỏi ĐÚNG/SAI)
#   make ppl        # perplexity trên val.bin              (câu hỏi TỐT/XẤU)
#
# Không cần cờ kiến trúc: đây là code scalar thuần, cố ý không tối ưu -- nó là
# mốc tham chiếu, không phải bản chạy nhanh. Thang tối ưu nằm ở samples/cpu/.
#
# Đã chạy được trên: x86-64 Linux (gcc), macOS Apple Silicon (Apple clang),
# ARM64 Linux / Jetson (gcc).

CC     ?= cc
CFLAGS ?= -O3 -Wall
LDLIBS  = -lm

ROOT   := ../..
MODEL  ?= $(ROOT)/firmware/model/model.bin
GOLD   ?= $(ROOT)/firmware/model/golden.txt
# Khớp --vocab đã train: 4096 -> val.bin, 32768 -> val_v32768.bin
VAL    ?= $(ROOT)/data/val.bin
WINDOWS ?= 8

all: verify_c ppl_c

verify_c: verify.c $(ROOT)/firmware/common/llm.h
	$(CC) $(CFLAGS) verify.c -o $@ $(LDLIBS)

ppl_c: ppl.c $(ROOT)/firmware/common/llm.h
	$(CC) $(CFLAGS) ppl.c -o $@ $(LDLIBS)

# Tầng 1: port C có đúng không? So logits với golden do PyTorch sinh ra.
verify: verify_c
	./verify_c $(MODEL) $(GOLD)

# Tầng 2: model có còn tốt không? Perplexity mới bắt được lỗi mà argmax bỏ lọt.
ppl: ppl_c
	./ppl_c $(MODEL) $(VAL) $(WINDOWS)

clean:
	rm -f verify_c ppl_c

.PHONY: all verify ppl clean
