import asyncio
import os, pprint
from unittest import result
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.base import TaskResult
from autogen_agentchat.conditions import ExternalTermination, TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_core import CancellationToken
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

# 建立 primary
primary_agent = AssistantAgent(
    "primary",
    model_client=model_client,
    system_message="You are a helpful AI assistant.",
)

# 建立 critic
critic_agent = AssistantAgent(
    "critic",
    model_client=model_client,
    system_message="Provide constructive feedback. Respond with 'APPROVE' when your feedbacks are addressed.",
)

# 定義一個終止條件，如果評論者允者，則停止任務。
text_termination = TextMentionTermination("APPROVE")

# 建立一個團隊，由 primary 和 critic 這兩個代理組成。
team = RoundRobinGroupChat([
	    primary_agent, 
		critic_agent
	], 
	termination_condition=text_termination
)

# 執行團隊任務
async def team_run() -> None:
    await Console(team.run_stream(task="寫一個富有幽默感的笑話。"))

    # 另一種執行方式
    # result = await team.run(task="Write a short poem about the fall season.")
    # print(result)


# 主程式
asyncio.run(team_run())