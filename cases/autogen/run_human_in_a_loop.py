import os
import asyncio
from pprint import pprint
from autogen_agentchat.agents import (
    AssistantAgent, 
    UserProxyAgent
)
from typing import Sequence
from itertools import cycle
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.ui import Console
from autogen_agentchat.messages import AgentEvent, ChatMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.models.ollama import OllamaChatCompletionClient


'''
使用本地的 ollama 伺服器 (使用 OllamaChatCompletionClient)
'''
# 模型變數初始化
model_client = OllamaChatCompletionClient(
    model="mistral-small3.1:24b",
    model_info={
        "vision": False,
        "function_calling": False,
        "json_output": False,
        "family": "unknown",
        "structured_output": True,
    },
)

# 建立 agent
teacher = AssistantAgent(
    "teacher",
    model_client=model_client,
    system_message='''你是一位小學老師，使用繁體中文，正在跟學生在課堂上玩猜數字遊戲，過程會不斷地提示範圍。如果學生猜對就說 "猜對了"，並結束遊戲。'''
)

# 自己
human = UserProxyAgent(
    "human",
    description="首次加入這個班級的小學生。",
)

# 建立 agent
student_01 = AssistantAgent(
    "student_01",
    model_client=model_client,
    system_message="你是一位小學生。"
)

# 建立 agent
student_02 = AssistantAgent(
    "student_02",
    model_client=model_client,
    system_message="你是一位小學生。"
)

# 定義一個終止條件，如果評論者允者，則停止任務。
text_termination = TextMentionTermination("猜對了！")

# 執行團隊任務
async def team_run() -> None:
    # 任務內容
    task = '''進行猜數字遊戲，數字在 1 到 100 之間。'''

    # 將先前的任務清除
    # await team.reset()

    # 定義所有的 agent
    agents = [teacher, human, student_01, student_02]

    # 設定發言順序。cycle() 會不斷重複迭代，直到所有 agent 都發言過一次。
    speaker_cycle = cycle(agent.name for agent in agents[1:])

    # 自訂選擇函數，控制發言順序
    def selector_func(messages: Sequence[AgentEvent | ChatMessage]) -> str | None:
        # 如果沒有訊息，則由老師開始
        if not messages:
            return "teacher"
        
        # 如果最後一個訊息是老師發送的，則由下一位學生開始
        last_message = messages[-1]
        if last_message.source == "teacher":
            # 透過 next() 取得下一位 agent
            return next(speaker_cycle)
        else:
            # 如果最後一個訊息是學生發送的，則由老師開始
            return "teacher"
    
    # 建立一個團隊
    team = SelectorGroupChat(
        agents,
        model_client=model_client,
        selector_func=selector_func,
        termination_condition=text_termination
    )

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