'''
Note:
pip install -U duckduckgo-search
'''

from pprint import pprint
import asyncio
from duckduckgo_search import DDGS
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken
from autogen_ext.models.openai import OpenAIChatCompletionClient


'''
使用 Gemini
'''
# from dotenv import load_dotenv
# load_dotenv(override=True)


# # 輸入你的 Google API Key
# api_key = os.getenv('GOOGLE_API_KEY')
# if not api_key:
# 	raise ValueError('GOOGLE_API_KEY is not set')

# # 設定模型
# model = "gemini-1.5-flash-8b"

# # 模型變數初始化
# model_client = OpenAIChatCompletionClient(
#     model=model,
#     api_key=api_key,
# 	model_info={
#         "vision": False,
#         "function_calling": False,
#         "json_output": False,
#         "family": "unknown",
#     },
# )


'''
使用 ollama
'''
# 模型變數初始化 (在這裡使用 ollama，下載 llama 3.3 70b 量化模型)
model_client = OpenAIChatCompletionClient(
    model="llama3.3:latest",
    base_url="http://localhost:11434/v1",
    api_key="placeholder",
    model_info={
        "vision": False,
        "function_calling": True,
        "json_output": False,
        "family": "unknown",
    },
)

# 定義一個簡單的函式 (tool function)
async def web_search(query: str) -> str:
    # 使用 DuckDuckGo 搜尋
    results = DDGS().text(query, max_results=5)
    return results

# 定義一個助理
agent = AssistantAgent(
    name="Assistant",
    model_client=model_client,
    tools=[web_search],
    system_message="Use tools to find information.",
)

# 主程式
async def assistant_run() -> None:
    # 輸入一個訊息
    response = await agent.on_messages(
        [TextMessage(content="Find information on telunyang", source="user")],
        cancellation_token=CancellationToken(),
    )

    # pprint(response.chat_message)
    '''
    ToolCallSummaryMessage(
        source='Assistant', 
        models_usage=None, 
        content="[
            {'title': 'telunyang (楊德倫) · GitHub', 'href': 'https://github.com/telunyang', 'body': 'Te-Lun Yang (telunyang) / Darren. telunyang has 51 repositories available. Follow their code on GitHub.'}, 
            {'title': '擺渡人_楊德倫 - YouTube', 'href': 'https://www.youtube.com/channel/UCUqT6-mTPkQkCyGjlbm3IMA', 'body': 'For years, I have dedicated myself to the development of several projects and the authoring of several papers in the field of Natural Language Processing (NLP). With the knowledge and expertise I ...'}, 
            {'title': 'GitHub - telunyang/telunyang: 關於我', 'href': 'https://github.com/telunyang/telunyang', 'body': '關於我. Contribute to telunyang/telunyang development by creating an account on GitHub.'}, 
            {'title': 'telunyang/python_web_scraping - GitHub', 'href': 'https://github.com/telunyang/python_web_scraping', 'body': 'Web scraping (網路爬蟲). Contribute to telunyang/python_web_scraping development by creating an account on GitHub.'}, 
            {'title': 'telunyang - YouTube', 'href': 'https://www.youtube.com/telunyang', 'body': 'Share your videos with friends, family, and the world'}
        ]", type='ToolCallSummaryMessage')
    '''
    pprint(eval(response.chat_message.content))

# 呼叫主程式
asyncio.run(assistant_run())
