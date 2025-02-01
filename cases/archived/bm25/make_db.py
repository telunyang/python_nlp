'''
SQLite 資料類型
https://www.runoob.com/sqlite/sqlite-data-types.html
'''

# SQLite 資料庫
import sqlite3
import argparse

# 建立 .db
def make_db(db_path):
    # 檔案名稱與路徑
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    sql = '''DROP TABLE IF EXISTS `pdf`;'''
    cursor.execute(sql)

    sql = f'''
    CREATE TABLE "pdf" (
        "id"    INTEGER PRIMARY KEY AUTOINCREMENT,
        "page_num"  INT,
        "page_type" VARCHAR(15),
        "text_content"  TEXT,
        "table_format"  TEXT
    );
    '''
    cursor.execute(sql)

    conn.commit()
    
    # 關閉 sqlite
    cursor.close()
    conn.close()

if __name__ == "__main__":
    # 取得 cmd 引數
    parser = argparse.ArgumentParser(description='新增資料庫')
    parser.add_argument('--db_path', default='2633.db', help='資料庫檔案名稱', type=str)
    args = parser.parse_args()

    # 建立資料庫
    make_db(args.db_path)

'''
指令:

$ python make_db.py --db_path=2633.db
'''