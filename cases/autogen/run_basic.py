import os
import pprint
import asyncio
from autogen_core.models import UserMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.models.ollama import OllamaChatCompletionClient


'''
使用 Gemini
'''
# from dotenv import load_dotenv
# load_dotenv(override=True)

# 輸入你的 Google API Key
# api_key = os.getenv('GOOGLE_API_KEY')
# if not api_key:
# 	raise ValueError('GOOGLE_API_KEY is not set')

# 設定模型
# model = "gemini-2.0-flash-lite"

# 模型變數初始化
# model_client = OpenAIChatCompletionClient(
#     model=model,
#     api_key=api_key,
# 	model_info={
#         "vision": False,
#         "function_calling": False,
#         "json_output": False,
#         "family": "unknown",
#         "structured_output": True,
#     },
# )

'''
使用遠端的 ollama 伺服器 (使用 OpenAIChatCompletionClient)
'''
# # 模型變數初始化
# model_client = OpenAIChatCompletionClient(
#     model="gpt-oss:120b",
#     base_url="http://127.0.0.1:11434/v1",
#     api_key="placeholder",
#     model_info={
#         "vision": False,
#         "function_calling": False,
#         "json_output": False,
#         "family": "unknown",
#         "structured_output": True,
#     },
# )

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



# 主程式
async def main() -> None:
    response = await model_client.create([
        UserMessage(content="What is the capital of France?", source="user")
    ])

    '''
    response:
    CreateResult(finish_reason='stop', content='The capital of France is **Paris**.', usage=RequestUsage(prompt_tokens=74, completion_tokens=54), cached=False, logprobs=None, thought=None)

    response.__dict__:
    {
        'cached': False,
        'content': 'The capital of France is **Paris**.',
        'finish_reason': 'stop',
        'logprobs': None,
        'thought': None,
        'usage': RequestUsage(prompt_tokens=74, completion_tokens=46)
    }
    '''
    pprint.pprint(response)
    pprint.pprint(response.__dict__)

# 呼叫主程式
asyncio.run(main())
