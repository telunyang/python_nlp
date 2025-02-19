import asyncio
import os
from pprint import pprint
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.base import TaskResult
from autogen_agentchat.conditions import ExternalTermination, TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_core import CancellationToken
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

# 建立 agent
teacher = AssistantAgent(
    "teacher",
    model_client=model_client,
    system_message="你是一位小學老師，正在跟學生在課堂上玩猜數字遊戲。如果學生猜對，就說「猜對了」"
)

# 建立 agent
student_01 = AssistantAgent(
    "student_01",
    model_client=model_client,
    system_message="你是一位小學生，正在跟老師在課堂上玩猜數字遊戲。"
)

# 定義一個終止條件，如果評論者允者，則停止任務。
text_termination = TextMentionTermination("猜對")

# 建立一個團隊
team = RoundRobinGroupChat([
        teacher, 
        student_01,
    ], 
	termination_condition=text_termination
)

# 執行團隊任務
async def team_run() -> None:
    # 任務內容
    task = '''進行猜數字遊戲，數字在 1 到 100 之間，實際數字由你決定。學生猜的數字太大或太小，你會不斷地提醒他們實際數字在某個區間內，'''

    # 將先前的任務清除
    # await team.reset()

    # 執行任務 (會輸出合適格式的對話)
    await Console(team.run_stream(task=task))

    # 另一種執行方式
    # async for message in team.run_stream(task=task):
    #     if isinstance(message, TaskResult):
    #         print("Stop Reason:", message.stop_reason)
    #     else:
    #         print(message)
    #         print("=" * 20)
            


# 主程式
asyncio.run(team_run())