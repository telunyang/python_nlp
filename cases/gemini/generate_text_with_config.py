import os
from google import genai
from google.genai import types
from dotenv import load_dotenv
load_dotenv()

# 讀取 API Key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# 建立 Client
client = genai.Client(api_key=GOOGLE_API_KEY)

# Generate content
response = client.models.generate_content(
    model="gemini-1.5-flash-8b",
    contents=["請簡單地解釋 AI 是如何運作的，只要 100 個字。"],
    config=types.GenerateContentConfig(
        system_instruction="A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.",
        temperature=0,
        top_p=0.95,
        top_k=20,
        candidate_count=1,
        seed=42,
        max_output_tokens=500,
        stop_sequences=['STOP!'],
        presence_penalty=0.0,
        frequency_penalty=0.0,
    )
)
print(response.text)