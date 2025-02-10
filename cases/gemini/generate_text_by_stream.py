import os
from google import genai
from dotenv import load_dotenv
load_dotenv()

# 讀取 API Key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# 建立 Client
client = genai.Client(api_key=GOOGLE_API_KEY)

# Generate content
for chunk in client.models.generate_content_stream(
    model="gemini-1.5-flash-8b",
    contents=[["請簡單地解釋 AI 是如何運作的，只要 100 個字。"]]
):
    print(chunk.text, end="")
