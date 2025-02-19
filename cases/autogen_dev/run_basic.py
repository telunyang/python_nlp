import os
import asyncio
from autogen_core.models import UserMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient


'''
使用 Gemini
'''
from dotenv import load_dotenv
load_dotenv(override=True)


# 輸入你的 Google API Key
api_key = os.getenv('GOOGLE_API_KEY')
if not api_key:
	raise ValueError('GOOGLE_API_KEY is not set')

# 設定模型
model = "gemini-1.5-flash-8b"

# 模型變數初始化
model_client = OpenAIChatCompletionClient(
    model=model,
    api_key=api_key,
	model_info={
        "vision": False,
        "function_calling": False,
        "json_output": False,
        "family": "unknown",
    },
)


'''
使用 ollama
'''
# 模型變數初始化 (在這裡使用 ollama，下載 llama 3.3 70b 量化模型)
# model_client = OpenAIChatCompletionClient(
#     model="llama3.3:latest",
#     base_url="http://localhost:11434/v1",
#     api_key="placeholder",
#     model_info={
#         "vision": False,
#         "function_calling": True,
#         "json_output": False,
#         "family": "unknown",
#     },
# )

# 主程式
async def main() -> None:
    response = await model_client.create([UserMessage(content="What is the capital of France?", source="user")])
    print(response)

# 呼叫主程式
asyncio.run(main())
