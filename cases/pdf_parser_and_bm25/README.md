# pdf_parser_and_bm25
PDF 剖析器 與 BM25 (Best Match 25)

## BM25
- [wiki](https://en.wikipedia.org/wiki/Okapi_BM25)
- BM25 is a bag-of-words retrieval function that ranks a set of documents based on the query terms appearing in each document, regardless of their proximity within the document.
- BM25 是一種詞袋 (bag-of-words) 檢索功能，它根據每個文件 (document) 中出現的查詢術語 (terms of user's query) 對一組文件進行排名，不管它們在文件中的接近程度如何。

## 安裝套件
Python 套件:
```bash
# 一般使用
$ pip install -r requirements.txt
```


Git 工具:
- Windows
  - [連結](https://git-scm.com/download/win)
  - Standalone Installer
    - 64-bit Git for Windows Setup
- MacOS
  - [連結](https://git-scm.com/download/mac)
  - `brew install git`
- Linux
  - [連結](https://git-scm.com/book/zh-tw/v2/%E9%96%8B%E5%A7%8B-Git-%E5%AE%89%E8%A3%9D%E6%95%99%E5%AD%B8)
  - `sudo apt-get install git-all`


在 Windows 安裝 Tesseract 5.x:
- [Tesseract User Manual](https://github.com/tesseract-ocr/tessdoc)
  - [Tesseract installer for Windows](https://github.com/UB-Mannheim/tesseract/wiki)
  - [Install additional language and script models](https://github.com/UB-Mannheim/tesseract/wiki/Install-additional-language-and-script-models)
- 參考連結
  - [Tesseract OCR - 繁體中文【安裝篇】](https://vocus.cc/article/621cfdb3fd8978000162a2e8)
  - [Tesseract OCR - 繁體中文【簡易識別篇】](https://vocus.cc/article/621d0832fd8978000162bc2e)


在 Ubuntu Linux 安裝 Tesseract 5.x:
```bash
$ sudo apt-get update
$ sudo add-apt-repository ppa:alex-p/tesseract-ocr-devel
$ sudo apt-get install libtesseract-dev tesseract-ocr-eng tesseract-ocr-chi-tra
```


在 ~/.bashrc 最上方加入:
```bash
# 在檔案最上方加入:
export TESSBIN="/usr/bin/tesseract"
export TESSDATA_PREFIX="/usr/share/tesseract-ocr/5/tessdata/"
```
記得儲存後執行:
```bash
$ source ~/.bashrc
```


## 使用方式
```bash
# 1. 建立資料庫 (--db_path=股票代號.db)
$ python make_db.py --db_path=2633.db

# 2. 將 PDF 檔案剖析後寫入資料庫 (--db_path=股票代號.db --pdf_file=年報名稱.pdf)
$ python run_data_processing.py --db_file=2633.db --pdf_file=2021_2633.pdf
```