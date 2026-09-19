# VS Code 連接 Google Colab T4

## 1. 架構

```text
本機 VS Code
    │
    │ Google Colab Extension
    ▼
Google Colab Runtime
    │
    ├── Python / Jupyter kernel
    ├── /content
    └── NVIDIA GPU
```

VS Code 是本機 IDE，但 Notebook kernel 與 Colab Terminal 實際執行在 Google Colab Runtime。

## 2. 建立 GPU Runtime

在 VS Code 開啟 `.ipynb`，依序選擇：

```text
Select Kernel
→ Colab
→ New Colab Server
→ GPU
→ T4
```

如果當下資源允許，就會建立 T4 Runtime。

## 3. 選擇 Python Kernel

建立 Runtime 後通常選：

```text
Python 3 (ipykernel)
```

不要只靠 kernel 名稱判斷 GPU，應實際執行：

```bash
nvidia-smi
```

如果看到：

```text
Tesla T4
```

表示目前 Runtime 確實取得 T4。

## 4. 用 Python 驗證 CUDA

```python
import torch

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
```

## 5. 確認目前是在 Colab

```bash
pwd
```

通常會看到：

```text
/content
```

也可以：

```bash
hostname
nvidia-smi
```

## 6. 使用 Colab Terminal

從 VS Code 的 Colab panel 或 Command Palette 開啟 Colab Terminal。

Terminal prompt 可能類似：

```text
/content#
```

這個 shell 位於遠端 Colab VM，不是本機 Windows PowerShell。

## 7. `/content` 不是永久儲存

`/content` 屬於 Colab Runtime 的暫存檔案系統。

Runtime 被移除或回收後，檔案可能消失。

建議：

```text
程式碼
→ Git / GitHub

重要 dataset
→ Google Drive / Hugging Face / Cloud Storage

checkpoint / 實驗結果
→ 定期備份

暫存檔
→ /content
```

## 8. 正確關閉 Runtime

工作完成後，不要只關閉 VS Code。

依序：

```text
儲存程式碼 / checkpoint
→ 確認重要資料已有備份
→ Ctrl + Shift + P
→ Colab: Remove Server
→ 確認 server 已釋放
→ 關閉 VS Code
```

## 9. 常見問題

### Auto Connect 連到 CPU

如果需要指定 GPU，改用：

```text
Select Kernel
→ Colab
→ New Colab Server
→ GPU
→ T4
```

### GPU RAM 顯示 0 GB

如果：

```text
GPU RAM: 0.00 / 15.00 GB
```

通常只是代表目前還沒有模型或 tensor 佔用 GPU。

### Runtime 建立後沒有反應

可以嘗試：

```text
Ctrl + Shift + P
→ Developer: Reload Window
```

並確認 Colab Web 的 `Manage Sessions` 是否已有重複 Runtime。
