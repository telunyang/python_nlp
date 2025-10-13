import os
import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
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

# 定義一個簡單的函式 (tool function)
async def web_search(query: str) -> str:
    """Find information on the web"""
    return "AutoGen is a programming framework for building multi-agent applications."

# 定義一個助理
agent = AssistantAgent(
    name="Assistant",
    model_client=model_client,
    tools=[web_search],
    system_message="Use tools to solve tasks.",
)

# 主程式
async def assistant_run() -> None:
    response = await agent.on_messages(
        [TextMessage(content="Find information on AutoGen", source="user")],
        cancellation_token=CancellationToken(),
    )
    print("=" * 50)
    print(response.inner_messages)
    print("=" * 50)
    print(response.chat_message)

    '''
    ==================================================
    [
        ToolCallRequestEvent(
            source='Assistant', 
            models_usage=RequestUsage(
                prompt_tokens=22, completion_tokens=6
            ), 
            content=[
                FunctionCall(id='', arguments='{"query":"AutoGen"}', name='web_search')
            ], 
            type='ToolCallRequestEvent'
        ), 
        ToolCallExecutionEvent(
            source='Assistant', 
            models_usage=None, 
            content=[
                FunctionExecutionResult(
                    content='AutoGen is a programming framework for building multi-agent applications.', 
                    call_id='', 
                    is_error=False
                )
            ], 
            type='ToolCallExecutionEvent'
        )
    ]
    ==================================================
    source='Assistant' 
    models_usage=None 
    content='AutoGen is a programming framework for building multi-agent applications.' 
    type='ToolCallSummaryMessage'
    '''

# 呼叫主程式
asyncio.run(assistant_run())
