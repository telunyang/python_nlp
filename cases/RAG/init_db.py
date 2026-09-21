import sqlite3
import json

# 建立資料庫和資料表
def create_database(db_name):
    # 建立 SQLite 資料庫連線
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    # 建立資料表
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS news (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            link TEXT,
            reporter TEXT,
            date TEXT,
            title TEXT,
            content TEXT,
            img_path TEXT,
            summary TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # 提交變更並關閉連線
    conn.commit()
    conn.close()

# 將新聞資料存入資料庫
def insert_news_data(news_data, db_name):
    # 建立 SQLite 資料庫連線
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    # 將新聞資料插入資料表
    for news in news_data:
        cursor.execute("""
            INSERT INTO news (link, reporter, date, title, content, img_path, summary)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (news["link"], news["reporter"], news["date"], news["title"], news["content"], news["img_path"], news["summary"]))

    # 提交變更並關閉連線
    conn.commit()
    conn.close()


if __name__ == "__main__":
   # 讀取 news_data.json 檔案
   with open("news_data.json", "r", encoding="utf-8") as f:
      news_data = json.load(f)
   db_name = "news.db"
   create_database(db_name)
   insert_news_data(news_data, db_name)