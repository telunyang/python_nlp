import fitz
import logging, time, re, os, io, subprocess, json
from argparse import ArgumentParser
import sqlite3
from PIL import Image

'''
ArgumentParser 設定
'''
# 建立 AugumentParser 物件
parser = ArgumentParser()

# 加入參數與說明
parser.add_argument("--db_file", help="資料庫名稱", default='2330.db', type=str)
parser.add_argument("--pdf_file", help="PDF名稱", default='pdf/2022_2330.pdf', type=str)

# 取得使用者資料 (來自指令參數)
args = parser.parse_args()

'''
資料庫連線
'''
db_file = args.db_file
conn = sqlite3.connect(db_file)
cursor = conn.cursor()

'''
取得 PDF 檔案資訊
'''
# 讀取 PDF 檔案
pdf_file = args.pdf_file
doc = fitz.open(pdf_file)

# 用於圖片名稱的局部檔名
img_suffix_name = pdf_file.split("/")[-1].split(".")[0]

'''
Logging 設定
'''
# 基本設定
logger = logging.getLogger('data_processing')

# 設定等級
logger.setLevel(logging.DEBUG)

# 設定輸出格式
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', "%Y-%m-%d %H:%M:%S")
# formatter = logging.Formatter('%(message)s')

# 儲存在 log 當中的事件處理器
fileHandler = logging.FileHandler(f'data_processing_{pdf_file.split("/")[-1]}.log', mode='w', encoding='utf-8') # a: append, w: write
fileHandler.setFormatter(formatter)

# 輸出在終端機介面上的事件處理器
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)

# 加入事件
logger.addHandler(console_handler)
logger.addHandler(fileHandler)

# 建立圖片暫存資料夾
img_dir = 'images'
if not os.path.exists(img_dir):
    os.makedirs(img_dir)

t1 = time.time()

'''
指令參(引)數設定

$ python run_data_processing.py --db_file=2633.db --pdf_file=pdf/2021_2633.pdf
'''

try:
    # 取得頁數
    for index, page in enumerate(doc):
        # 頁碼
        page_num = index + 1

        print(f"正在處理: 第 {page_num} 頁...")

        # 取得頁面文字
        text = page.get_text()
        # logger.error(text[:50])

        # SQL 字串初始化
        sql = ''

        # 判斷是否為圖片
        if len(text) < 3:
            logger.info(f'Page {page_num}: image')

            # 取得圖片
            images = page.get_images()
            for index, image in enumerate(images):
                # 儲存圖片
                data = doc.extract_image(image[index])
                with Image.open(io.BytesIO(data.get('image'))) as image:
                    image_path = f'{img_dir}/page_{img_suffix_name}_{page_num}_{index}.{data.get("ext")}'
                    image.save(image_path, mode='wb')

                    # 使用 tesseract ocr 5 來取得圖片文字
                    output_file = './output_ocr'
                    tesseract_cmd = f'tesseract {image_path} {output_file} -l chi_tra+eng --oem 1 --psm 3'
                    stdout = subprocess.run(tesseract_cmd, shell=True)
                    if stdout.returncode == 0:
                        # 讀取 ocr 結果
                        with open(f"{output_file}.txt", 'r', encoding='utf-8') as file:
                            ocr_text = file.read()

                            # 清理文字
                            ocr_text = re.sub(r'�|\'|\"', '', ocr_text)

                            # 寫入資料庫
                            try:
                                sql = f'''
                                INSERT INTO pdf (page_num, page_type, text_content) 
                                VALUES (?, ?, ?);
                                '''
                                data = (page_num, "image", ocr_text)
                                cursor.execute(sql, data) 
                            except Exception as e:
                                logger.debug(sql)
                    else:
                        logger.error(f"Page {page_num} 錯誤訊息: {stdout.stderr}")
        else:
            # 清理文字
            text = re.sub(r'�', '', text)

            # 旋轉頁面
            count = 0
            while count < 4:
                # 旋轉頁面
                page.set_rotation(90 * count)
                logger.info(f"頁面 {page_num} 旋轉: {90 * count} 度")

                # 判斷是否有表格
                tabs = page.find_tables()
                len_tables = len(tabs.tables)
                if len_tables > 0:
                    logger.info(f'Page {page_num}: word,table')

                    # 取得表格資訊
                    dict_ = dict()
                    for index, tab in enumerate(tabs):
                        header = tab.header
                        dict_[f"table_{index + 1}"] = {
                            "header_names": header.names,
                            "tables": tab.extract()
                        }

                    # 清理文字
                    table_format = re.sub(r'�', '', json.dumps(dict_, ensure_ascii=False))
                    
                    try:
                        # 寫入資料庫
                        sql = f"""
                        INSERT INTO pdf (page_num, page_type, text_content, table_format)
                        VALUES (?, ?, ?, ?);
                        """
                        data = (page_num, "word,table", text, table_format)
                        cursor.execute(sql, data)
                    except Exception as e:
                        logger.error(f"Page {page_num} 錯誤訊息: {e}")
                        logger.error(sql)

                    break

                count += 1
            
            # 旋轉幾次都沒有表格，便直接儲存文字
            if count > 3:
                logger.info(f'Page {page_num}: word,no-table')

                # 寫入資料庫
                sql = f"""
                INSERT INTO pdf (page_num, page_type, text_content)
                VALUES (?, ?, ?);
                """
                data = (page_num, "word,no-table", text)
                cursor.execute(sql, data)  

    conn.commit()
except Exception as e:
    conn.rollback()
finally:
    # 關閉檔案
    doc.close()

    # 關閉 sqlite
    cursor.close()
    conn.close()

t2 = time.time()

# 顯示執行時間
print(f"執行時間: {t2-t1} 秒 => {(t2-t1)/60} 分鐘 => {(t2-t1)/3600} 小時")