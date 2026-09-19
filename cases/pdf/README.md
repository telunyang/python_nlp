# PDF 剖析器
- PyMuPDF: [https://github.com/pymupdf/pymupdf](https://github.com/pymupdf/pymupdf)
- MinerU: [https://github.com/opendatalab/mineru](https://github.com/opendatalab/mineru)

## 安裝指令
```bash
# PyMuPDF
pip install -U pymupdf==1.28.2 fonttools==4.63.0 pymupdf-fonts==1.0.5

# MinerU
pip install -U "mineru[all]==3.4.5"
```

## MinerU 基本執行範例
```bash
# 有 GPU 可以用的時候
mineru -p <input_path> -o <output_path>

# 只有 CPU 可以用的時候
mineru -p <input_path> -o <output_path> -b pipeline
```
參考連結: [https://opendatalab.github.io/MinerU/usage/](https://opendatalab.github.io/MinerU/usage/)