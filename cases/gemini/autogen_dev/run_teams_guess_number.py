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
    api_key=api_key
)

# 建立 agent
teacher = AssistantAgent(
    "teacher",
    model_client=model_client,
    system_message='''你是一位小學老師，正在跟學生在課堂上玩猜數字遊戲，例如 1 到 100 之間。
你會不斷地提示學生，學生猜的數字太大或太小，你會提醒他們實際數字在某個區間內，
例如答案是 20，學生一開始猜 35，你就會說「太大了，答案在 1 到 35 之間」，
如果學生猜 6，你就會說「太小了，答案在 6 到 35 之間」，依此類推。
如果其中有一位學生猜對了，你就會說「恭喜學生 {agent name} 答對了！」
'''
)

# 建立 agent
student_01 = AssistantAgent(
    "student_01",
    model_client=model_client,
    system_message="你是一位小學生，正在跟老師在課堂上玩猜數字遊戲。"
)

# 建立 agent
student_02 = AssistantAgent(
    "student_02",
    model_client=model_client,
    system_message="你是一位小學生，正在跟老師在課堂上玩猜數字遊戲。"
)

# 建立 agent
student_03 = AssistantAgent(
    "student_03",
    model_client=model_client,
    system_message="你是一位小學生，正在跟老師在課堂上玩猜數字遊戲。"
)

# 定義一個終止條件，如果評論者允者，則停止任務。
text_termination = TextMentionTermination("答對了")

# 建立一個團隊
team = RoundRobinGroupChat([
        teacher, 
        student_01,
        student_02,
        student_03
    ], 
	termination_condition=text_termination
)

# 執行團隊任務
async def team_run() -> None:
    # 任務內容
    task = "請使用繁體中文來進行猜數字遊戲。"

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


# 主程式
asyncio.run(team_run())