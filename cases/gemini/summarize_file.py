import os
from google import genai
from dotenv import load_dotenv
load_dotenv()

# 讀取 API Key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# 建立 Client
client = genai.Client(api_key=GOOGLE_API_KEY)

# 上傳檔案
file = client.files.upload(file='./poem.txt')

# Generate content
response = client.models.generate_content(
    model="gemini-1.5-flash-8b",
    contents=["你能夠對這個檔案進行摘要嗎？", file]
)
print(response.text)