---
name: gpu-box
description: Đồng bộ code và dữ liệu từ MacBook M3 sang máy PC có RTX 2060 rồi train, verify, lấy kết quả về. Dùng khi được yêu cầu train trên GPU, sync dữ liệu sang máy nhà, chạy thí nghiệm nặng, hoặc khi nhắc tới "máy PC", "GPU", "RTX", "machineai@192.168.1.138", hay khi một lệnh train sẽ mất hơn 10 phút trên Mac.
---

# Chạy việc nặng trên máy PC có GPU

Mac dùng để viết; PC dùng để train. Cùng cấu hình 2000 bước: **Mac M3 (MPS) mất
21,3 phút, RTX 2060 (CUDA) mất 5,3 phút**, nhanh gấp 4,05 lần. Mọi thí nghiệm dài
hơn 10 phút nên đẩy sang PC.

## Máy đích

```
alias SSH   : machineai            (đã cài key, không hỏi mật khẩu)
host        : 192.168.1.138
repo        : ~/machineai
venv        : ~/machineai/.venv    torch 2.5.1+cu121, cuda True
GPU         : RTX 2060 6GB, compute 7.5 -> nvcc -arch=sm_75
CUDA toolkit: /usr/local/cuda/bin  (KHÔNG có sẵn trong PATH, phải export)
CPU / RAM   : 32 nhân / 31 GB
```

Kiểm nhanh máy còn sống và key còn dùng được:

```bash
ssh -o BatchMode=yes machineai 'nvidia-smi --query-gpu=name,memory.used --format=csv,noheader'
```

`BatchMode=yes` để nếu key hỏng thì lệnh **hỏng ngay** thay vì treo chờ nhập mật
khẩu, hoặc âm thầm rơi về mật khẩu rồi báo thành công giả.

## Bốn thứ đi qua hai đường khác nhau

Đây là chỗ dễ nhầm nhất, vì không phải cái gì cũng đi bằng git:

| Thứ | Cách đồng bộ | Vì sao |
|---|---|---|
| Code (`src/`, `data/*.py`, `firmware/`) | `git push` rồi `git pull` trên PC | có version, có thể quay lui |
| Dữ liệu thô, `*.bin`, tokenizer mới | `rsync` trực tiếp | nằm trong `.gitignore`, và 150 MB không nên vào git |
| Checkpoint `runs/*.pt` | `rsync` chiều ngược lại | sinh ra trên PC, cần mang về Mac để viết bài |
| `firmware/model/*` | **cẩn thận** | là file ĐÃ COMMIT; `export.py` ghi đè tại chỗ |

## Quy trình đầy đủ

### 1. Đẩy code lên

```bash
git push origin main
ssh machineai 'cd ~/machineai && git status --short'      # phải rỗng trước khi pull
ssh machineai 'cd ~/machineai && git pull --ff-only origin main && git log --oneline -1'
```

Nếu `git status` không rỗng, gần như luôn là do lần trước chạy `export.py` trên PC
làm bẩn `firmware/model/`. Khôi phục rồi mới kéo:

```bash
ssh machineai 'cd ~/machineai && git checkout -- firmware/model/'
```

### 2. Đẩy dữ liệu lên

```bash
rsync -avh --progress /đường/dẫn/corpus/ machineai:~/machineai/data/mydata/
```

Dùng `rsync` chứ không `scp`: nó tiếp tục được khi đứt giữa chừng, và bỏ qua file
đã giống nhau. Với corpus vài trăm MB qua Wi-Fi thì khác biệt này rất đáng.

### 3. Chuẩn bị và train, LUÔN chạy nền

```bash
ssh machineai 'cd ~/machineai && nohup .venv/bin/python data/prepare.py \
  --input data/mydata --vocab 4096 --retrain-tokenizer > /tmp/prep.log 2>&1 &'

ssh machineai 'cd ~/machineai/src && nohup ../.venv/bin/python train.py \
  --arm ple --vocab 4096 --steps 2000 --tag mydata --seed 0 > /tmp/train.log 2>&1 &'
```

`nohup ... &` là bắt buộc, không phải cho gọn. Phiên SSH tới máy này **đã từng đứt
giữa chừng** (`Operation timed out`) trong lúc train; nhờ `nohup` mà tiến trình
sống sót và vẫn chạy tới `DONE`. Chạy trực tiếp qua SSH là mất trắng 20 phút.

Theo dõi:

```bash
ssh machineai 'tail -5 /tmp/train.log; nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader'
```

Chờ xong mà không phải bấm lại:

```bash
until ssh -o BatchMode=yes machineai 'grep -q DONE /tmp/train.log'; do sleep 20; done
```

### 4. Kiểm chứng ngay trên PC

Ba tầng, chạy theo thứ tự này:

```bash
ssh machineai 'cd ~/machineai/src && ../.venv/bin/python probe_check.py \
  --run ../runs/ple-mydata-s0.pt --manifest /tmp/probe/probe.json \
  --tokenizer ../data/bpe4096.json'                       # dữ liệu có vào model không

ssh machineai 'cd ~/machineai/src && ../.venv/bin/python export.py ple-mydata-s0'

ssh machineai 'cd ~/machineai && make -C firmware/host_verify verify'   # C khớp PyTorch

ssh machineai 'export PATH=/usr/local/cuda/bin:$PATH; cd ~/machineai/firmware/jetson && make verify'
```

`export.py` **ghi đè** `firmware/model/model.bin` và `golden.txt` tại chỗ. Muốn giữ
bản đang commit làm mốc thì sao lưu trước, và nhớ `git checkout -- firmware/model/`
trước lần `git pull` sau.

### 5. Mang kết quả về Mac

```bash
rsync -avh machineai:~/machineai/runs/ple-mydata-s0.pt ./runs/
rsync -avh machineai:~/machineai/runs/ple-mydata-s0.json ./runs/
```

## Những chỗ đã cắn thật, không phải giả định

**`nvcc` không có trong PATH.** Toolkit nằm ở `/usr/local/cuda/bin` nhưng shell
không tự nạp. Mọi lệnh đụng tới `firmware/jetson` phải có
`export PATH=/usr/local/cuda/bin:$PATH` đứng trước.

Bẫy ở chỗ lỗi **chỉ hiện khi thật sự cần biên dịch**. Nếu `verify_cuda` còn từ lần
build trước thì `make verify` chạy ngon lành dù thiếu PATH, nên rất dễ tưởng là
không cần. Sau một `make clean` thì nó hỏng ngay:

```
make: nvcc: No such file or directory
make: *** [Makefile:32: verify_cuda] Error 127
```

**Đĩa còn 175 GB nhưng đã dùng 90%.** Mỗi bộ dữ liệu 4096-vocab tốn khoảng 300 MB
(raw) + 150 MB (bins). Dọn `data/*_mix*.bin`, `data/*_v*.bin` sau mỗi thí nghiệm.

**`git status` bẩn chặn `git pull --ff-only`.** Nguyên nhân gần như luôn là
`firmware/model/` bị `export.py` ghi đè.

**Không dùng lại checkpoint khi đã đổi tokenizer.** Train lại tokenizer là đổi ý
nghĩa mọi hàng `tok_emb`, checkpoint cũ thành vô nghĩa. Xem `--init-from` trong
`train.py`: nó dùng `strict=True` cố ý để hỏng ngay thay vì nạp một nửa.

## Chi phí tham chiếu, đo trên chính máy này

| Việc | Thời gian |
|---|---|
| `prepare.py` TinyStories 300 MB | khoảng 6 phút |
| train 2000 bước, vocab 4096, d_model 128 | 315 giây (5,3 phút) |
| cùng việc đó trên Mac M3 (MPS) | 1277 giây (21,3 phút) |
| `export.py` | vài giây |
| `make verify` CUDA (gồm biên dịch) | dưới 1 phút |

## Bảo mật

Key riêng cho máy này ở `~/.ssh/id_ed25519_machineai`, không dùng chung với key
GitHub, nên thu hồi được độc lập bằng cách xoá một dòng trong `authorized_keys`
trên PC. Key không đặt passphrase để script chạy tự động được; đổi lại, ai đọc được
file key là vào được máy.

Máy vẫn đang bật đăng nhập bằng mật khẩu. Sau khi chắc chắn key hoạt động, cân nhắc
tắt nó đi, và **giữ một phiên SSH đang mở** trong lúc làm để còn đường sửa nếu sai:

```bash
sudo sed -i 's/^#*PasswordAuthentication.*/PasswordAuthentication no/' /etc/ssh/sshd_config
sudo systemctl restart ssh
```
