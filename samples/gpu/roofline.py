#!/usr/bin/env python3
"""Máy tính roofline cho LLM inference. Không cần dependency nào ngoài stdlib.

Trả lời: "trên phần cứng này, model này nhanh nhất có thể là bao nhiêu tok/s?"
Dùng con số đó làm mốc trước khi tối ưu bất cứ thứ gì (xem 01-roofline.md muc 1.5).

  python3 roofline.py --hw orin-nano-super --model llama-8b --bits 4.5 --ctx 4096
  python3 roofline.py --list
"""

import argparse

# bw_GBs   : băng thông DRAM thực tế đạt được (không phải con số marketing)
# fp16_TF  : peak FP16 tensor core dense
# int8_TOPS: peak INT8 tensor core dense
HW = {
    # ĐO THẬT trên board machineai@100.92.121.20 (Orin Nano Engineering Reference
    # Dev Kit Super, L4T R36.4.7 / JetPack 6.2), MAXN_SUPER + jetson_clocks, máy
    # rảnh. Xem MEASUREMENTS.md. Dùng mục này, KHÔNG dùng "-spec".
    "orin-nano-super": dict(bw_GBs=66.8, fp16_TF=10.14, int8_TOPS=12.8, ram_GB=8,
                            note="SỐ ĐO THẬT. EMC khoá 2133MHz -> trần DRAM 68.3 GB/s, "
                                 "đạt 66.8 (98%). cuBLAS FP16 10.1 TF, INT8 12.8 TOPS."),
    # Con số datasheet — để đối chiếu xem board bạn thiếu bao nhiêu.
    "orin-nano-super-spec": dict(bw_GBs=102, fp16_TF=16.7, int8_TOPS=33.5, ram_GB=8,
                                 note="DATASHEET, chưa chắc board bạn đạt được. "
                                      "Cần EMC 3199MHz mới có 102 GB/s."),
    "orin-nano-orig":  dict(bw_GBs=68,  fp16_TF=8.3,  int8_TOPS=20,   ram_GB=8,
                            note="Orin Nano trước bản Super BSP."),
    "orin-nx-16": dict(bw_GBs=102, fp16_TF=39.4, int8_TOPS=78.7, ram_GB=16,
                       note="Orin NX 16GB Super. Có 1 DLA."),
    "agx-orin-64": dict(bw_GBs=204, fp16_TF=106, int8_TOPS=137, ram_GB=64,
                        note="AGX Orin 64GB, 2 DLA."),
    "rtx-4060-laptop": dict(bw_GBs=272, fp16_TF=59, int8_TOPS=118, ram_GB=8,
                            note="VRAM riêng, không chia sẻ với CPU."),
}

# params_B, layers, d_model, n_kv_heads, head_dim
MODELS = {
    "llama-3.2-1b":  dict(params_B=1.24, layers=16, d_model=2048, kv_heads=8,  head_dim=64),
    "llama-3.2-3b":  dict(params_B=3.21, layers=28, d_model=3072, kv_heads=8,  head_dim=128),
    "llama-3.1-8b":  dict(params_B=8.03, layers=32, d_model=4096, kv_heads=8,  head_dim=128),
    "qwen2.5-7b":    dict(params_B=7.62, layers=28, d_model=3584, kv_heads=4,  head_dim=128),
    "qwen2.5-1.5b":  dict(params_B=1.54, layers=28, d_model=1536, kv_heads=2,  head_dim=128),
    "gemma-2-2b":    dict(params_B=2.61, layers=26, d_model=2304, kv_heads=4,  head_dim=256),
    "mistral-7b":    dict(params_B=7.24, layers=32, d_model=4096, kv_heads=8,  head_dim=128),
}


def kv_bytes(m, ctx, batch, kv_bits):
    """2 (K và V) × layers × kv_heads × head_dim × ctx × batch × bytes/phần tử."""
    return 2 * m["layers"] * m["kv_heads"] * m["head_dim"] * ctx * batch * (kv_bits / 8)


def report(hw_name, m_name, w_bits, kv_bits, ctx, batch, prompt_len, act_bits):
    hw, m = HW[hw_name], MODELS[m_name]
    bw = hw["bw_GBs"] * 1e9

    w_bytes = m["params_B"] * 1e9 * (w_bits / 8)
    kv = kv_bytes(m, ctx, batch, kv_bits)
    total_ram = w_bytes + kv

    print(f"\n=== {m_name} @ W{w_bits} / KV{kv_bits} trên {hw_name} ===")
    print(f"  {hw['note']}")
    print(f"  balance FP16 = {hw['fp16_TF']*1e12/bw:6.1f} FLOP/byte   "
          f"balance INT8 = {hw['int8_TOPS']*1e12/bw:6.1f} OP/byte")

    print("\n--- Bộ nhớ ---")
    print(f"  weights            {w_bytes/1e9:8.2f} GB")
    print(f"  KV cache @ctx{ctx} {kv/1e9:8.2f} GB  (batch={batch})")
    print(f"  tổng               {total_ram/1e9:8.2f} GB  / {hw['ram_GB']} GB")
    if total_ram / 1e9 > hw["ram_GB"] * 0.8:
        print("  ⚠️  VƯỢT ~80% RAM — sẽ swap hoặc OOM (OS ăn ~1.5GB)")

    # --- decode: đọc weights + KV mỗi token ---
    bytes_tok = w_bytes + kv
    ai_decode = (2 * m["params_B"] * 1e9 * batch) / bytes_tok
    tps_mem = bw / bytes_tok * batch
    tps_cmp = hw["fp16_TF"] * 1e12 / (2 * m["params_B"] * 1e9)

    print("\n--- Decode (sinh token) ---")
    print(f"  bytes/token        {bytes_tok/1e9:8.2f} GB")
    print(f"  arithmetic int.    {ai_decode:8.1f} FLOP/byte")
    bound = "MEMORY-bound" if tps_mem < tps_cmp else "COMPUTE-bound"
    print(f"  trần từ bandwidth  {tps_mem:8.1f} tok/s   <-- {bound}")
    print(f"  trần từ compute    {tps_cmp:8.1f} tok/s")
    print(f"  ➜ TRẦN THỰC TẾ     {min(tps_mem, tps_cmp):8.1f} tok/s")
    print(f"    kỳ vọng đạt được {min(tps_mem,tps_cmp)*0.5:5.1f}–"
          f"{min(tps_mem,tps_cmp)*0.7:.1f} tok/s (50–70% trần)")

    # --- prefill: mỗi weight dùng cho prompt_len token ---
    # Peak phụ thuộc ACTIVATION dtype, không phải weight dtype: W4A16 vẫn nhân
    # trên tensor core FP16 (dequant rồi mới nhân). Chỉ W8A8 mới chạm 33.5 TOPS.
    flops_pf = 2 * m["params_B"] * 1e9 * prompt_len * batch
    peak = hw["int8_TOPS"] * 1e12 if act_bits <= 8 else hw["fp16_TF"] * 1e12
    peak_label = "INT8 tensor core" if act_bits <= 8 else "FP16 tensor core"
    t_cmp = flops_pf / peak
    t_mem = (w_bytes + kv) / bw
    ai_pf = flops_pf / (w_bytes + kv)

    print(f"\n--- Prefill ({prompt_len} token prompt, A{act_bits:g} → {peak_label}) ---")
    print(f"  arithmetic int.    {ai_pf:8.1f} FLOP/byte")
    print(f"  {'COMPUTE' if t_cmp>t_mem else 'MEMORY'}-bound")
    print(f"  TTFT tối thiểu     {max(t_cmp,t_mem)*1000:8.1f} ms")
    print(f"  prefill throughput {prompt_len/max(t_cmp,t_mem):8.0f} tok/s")

    # --- batch tới hạn ---
    # AI của decode = 2P / (P × w_bits/8) = 16/w_bits. Batch tới hạn là B sao cho
    # B × AI_decode chạm machine balance.
    b_star = (peak / bw) / (16 / w_bits)
    print(f"\n--- Batching ---")
    print(f"  batch tới hạn      {b_star:8.1f}  (dưới mức này, tăng batch gần như miễn phí)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hw", default="orin-nano-super", choices=list(HW))
    ap.add_argument("--model", default="llama-3.1-8b", choices=list(MODELS))
    ap.add_argument("--bits", type=float, default=4.5,
                    help="bit/weight thực tế. Q4_K_M≈4.8, Q4_0≈4.5, Q8_0≈8.5, fp16=16")
    ap.add_argument("--kv-bits", type=float, default=16, help="16=fp16, 8=q8_0")
    ap.add_argument("--act-bits", type=float, default=16,
                    help="16 cho weight-only (Q4_K_M, AWQ). 8 chỉ khi thật sự W8A8 "
                         "(SmoothQuant, TensorRT INT8)")
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--prompt-len", type=int, default=512)
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()

    if a.list:
        print("Phần cứng:", ", ".join(HW))
        print("Model    :", ", ".join(MODELS))
        return
    report(a.hw, a.model, a.bits, a.kv_bits, a.ctx, a.batch, a.prompt_len, a.act_bits)


if __name__ == "__main__":
    main()
