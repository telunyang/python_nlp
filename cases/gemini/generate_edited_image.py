'''
生成圖像
https://ai.google.dev/gemini-api/docs/image-generation?hl=zh-tw

適用語系
EN、es-MX、ja-JP、zh-CN、hi-IN
'''

import os
del os.environ["GOOGLE_API_KEY"]

from google import genai
from google.genai import types
from PIL import Image
from io import BytesIO
import base64
from dotenv import load_dotenv
load_dotenv(override=True)

if "SSL_CERT_FILE" in os.environ:
    del os.environ["SSL_CERT_FILE"]

# 讀取 API Key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# 建立 Client
client = genai.Client(api_key=GOOGLE_API_KEY)

# 讀取圖片
image = Image.open('./example01.jpg')

# 你的 prompt
text_input = ('Hi, Can you add a cute car watermark next to the name "Darren"?',)

# 取得回應
response = client.models.generate_content(
    model="gemini-2.0-flash-exp-image-generation",
    contents=[text_input, image], # 結合你的 prompt 與上傳照片來生成圖片
    config=types.GenerateContentConfig(
      response_modalities=['TEXT', 'IMAGE']
    )
)

# 將回傳的圖片顯示出來
for part in response.candidates[0].content.parts:
    if part.text is not None:
        print(part.text)
    elif part.inline_data is not None:
        image = Image.open(BytesIO(part.inline_data.data))
        image.save('gemini_edited_image.png')
        image.show()