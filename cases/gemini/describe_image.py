'''
參考結果
https://i.imgur.com/o6QvMn9.png
'''

import os
from google import genai
from google.genai import types
from dotenv import load_dotenv
load_dotenv()

# 讀取 API Key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# 建立 Client
client = genai.Client(api_key=GOOGLE_API_KEY)

# 讀取圖片
YOUR_IMAGE_PATH = "./example01.jpg"
YOUR_IMAGE_MIME_TYPE = "image/jpeg"
with open(YOUR_IMAGE_PATH, "rb") as f:
    image_bytes = f.read()

# 進行圖片描述
for chunk in client.models.generate_content_stream(
    model="gemini-1.5-flash", # *-8b 也可以看圖片，只是比較沒那麼準確
    contents=[
        "請形容這張圖片的內容，其中有一個物件上面寫著字，請問上面寫什麼？",
        types.Part.from_bytes(data=image_bytes, mime_type=YOUR_IMAGE_MIME_TYPE),
    ],
):
    print(chunk.text, end="")