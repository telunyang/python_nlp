import asyncio
import pprint
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.models.ollama import OllamaChatCompletionClient
from autogen_ext.code_executors.local import LocalCommandLineCodeExecutor
from autogen_ext.tools.code_execution import PythonCodeExecutionTool
from autogen_agentchat.messages import (
    ToolCallRequestEvent,
    ToolCallExecutionEvent
)

'''
使用本地的 ollama 伺服器 (使用 OllamaChatCompletionClient)
'''
# 模型變數初始化
model_client = OllamaChatCompletionClient(
    model="gpt-oss:120b",
    model_info={
        "vision": False,
        "function_calling": True,
        "json_output": False,
        "family": "unknown",
        "structured_output": True,
    },
)

# 主程式
async def main() -> None:
    # 自訂 tool
    tool = PythonCodeExecutionTool(
        LocalCommandLineCodeExecutor(work_dir="coding")
    )

    # 建立 AssistantAgent
    agent = AssistantAgent(
        name="Assistant",
        model_client=model_client,
        tools=[tool],
        reflect_on_tool_use=True
    )

    # 進入對話
    await Console(
        agent.run_stream(
            task='''Create a plot of MSFT stock prices between 2024 and 2025 and save it to a file, afterwards, save the .py file to a file. Use yfinance and matplotlib.'''
        )
    )

    # 另一種執行方式
    # async for event in agent.run_stream(
    #             task='''Create a plot of MSFT stock prices between 2024 and 2025 and save it to a file, afterwards, save the .py file to a file. Use yfinance and matplotlib.'''
    # ):
    #     print("=" * 80)
    #     '''
    #     pprint.pprint(type(event))
    #     <class 'autogen_agentchat.messages.AgentStartedEvent'>
    #     <class 'autogen_agentchat.messages.TextMessage'>
    #     <class 'autogen_agentchat.messages.ToolCallRequestEvent'>
    #     <class 'autogen_agentchat.messages.ToolCallExecutionEvent'>
    #     <class 'autogen_agentchat.messages.TextMessage'>
    #     <class 'autogen_agentchat.base._task.TaskResult'>
    #     '''
    #     if type(event) == ToolCallRequestEvent:
    #         pprint.pprint(event.__dict__)

# 執行主程式
asyncio.run(main())