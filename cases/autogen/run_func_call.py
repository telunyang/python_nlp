from pprint import pprint
import asyncio
# from googlesearch import search
from ddgs import DDGS
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken
from autogen_ext.models.ollama import OllamaChatCompletionClient


'''
使用本地的 ollama 伺服器 (使用 OllamaChatCompletionClient)
'''
# 模型變數初始化
model_client = OllamaChatCompletionClient(
    model="mistral-small3.1:24b",
    model_info={
        "vision": False,
        "function_calling": True,
        "json_output": False,
        "family": "unknown",
        "structured_output": True,
    },
)

# 定義一個簡單的函式 (tool function)
def web_search(query: str) -> str:
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
        [
            TextMessage(content='''Find information on "楊德倫"''', source="user")
        ],
        cancellation_token=CancellationToken(),
    )

    # pprint(response.chat_message)
    '''
    ToolCallSummaryMessage(
        id='7a609775-3935-4bac-9f8d-a0a4d82785d9', 
        source='Assistant', 
        models_usage=None, 
        metadata={}, 
        created_at=datetime.datetime(2025, 10, 14, 2, 43, 21, 850819, tzinfo=datetime.timezone.utc), 
        content='[
            {\'title\': \'楊德倫 - Google 學術搜尋 - Google Scholar\', \'href\': \'https://scholar.google.com/citations?user=Td99utMAAAAJ&hl=zh-TW\', \'body\': \'楊德倫 其他名字 National Taiwan University 在 ntu.edu.tw 的電子郵件地址已通過驗證 - 首頁\'}, 
            {\'title\': \'楊德倫 - Research Assistant | LinkedIn\', \'href\': \'https://tw.linkedin.com/in/telunyang\', \'body\': \'楊德倫 說讚 I recently switched my go-to research model from Llama 3 to Qwen3. So I implemented Qwen3 from scratch to make it more accessible for tinkering (and… 楊德倫 說讚 #NTU, 筑波大学, …\'}, 
            {\'title\': \'楊德倫 - Facebook\', \'href\': \'https://www.facebook.com/100001203891098/\', \'body\': \'楊德倫 is on Facebook. Join Facebook to connect with 楊德倫 and others you may know. Facebook gives people the power to share and makes the world more open and connected.\'}, 
            {\'title\': \'擺渡人_ 楊德倫 (@darreninfo.cc) • Instagram photos and videos\', \'href\': \'https://www.instagram.com/darreninfo.cc/\', \'body\': \'232 Followers, 353 Following, 1,796 Posts - 擺渡人_楊德倫 (@darreninfo.cc) on Instagram: "Darren 臺大資工系網媒所 電腦補習班打工仔"\'}, 
            {\'title\': \'telunyang ( 楊德倫 ) · GitHub\', \'href\': \'https://github.com/telunyang\', \'body\': \'For years, I have dedicated myself to the development of several projects and the authoring of several papers in the field of Natural Language Processing (NLP). With the knowledge and …\'}
        ]', 
        type='ToolCallSummaryMessage', 
        tool_calls=[
            FunctionCall(id='0', arguments='{"query": "\\u694a\\u5fb7\\u502b"}', name='web_search')
        ], 
        results=[
            FunctionExecutionResult(
                content='[
                    {\'title\': \'楊德倫 - Google 學術搜尋 - Google Scholar\', \'href\': \'https://scholar.google.com/citations?user=Td99utMAAAAJ&hl=zh-TW\', \'body\': \'楊德倫 其他名字 National Taiwan University 在 ntu.edu.tw 的電子郵件地址已通過驗證 - 首頁\'}, 
                    {\'title\': \'楊德倫 - Research Assistant | LinkedIn\', \'href\': \'https://tw.linkedin.com/in/telunyang\', \'body\': \'楊德倫 說讚 I recently switched my go-to research model from Llama 3 to Qwen3. So I implemented Qwen3 from scratch to make it more accessible for tinkering (and… 楊德倫 說讚 #NTU, 筑波大学, …\'}, 
                    {\'title\': \'楊德倫 - Facebook\', \'href\': \'https://www.facebook.com/100001203891098/\', \'body\': \'楊德倫 is on Facebook. Join Facebook to connect with 楊德倫 and others you may know. Facebook gives people the power to share and makes the world more open and connected.\'}, 
                    {\'title\': \'擺渡人_ 楊德倫 (@darreninfo.cc) • Instagram photos and videos\', \'href\': \'https://www.instagram.com/darreninfo.cc/\', \'body\': \'232 Followers, 353 Following, 1,796 Posts - 擺渡人_楊德倫 (@darreninfo.cc) on Instagram: "Darren 臺大資工系網媒所 電腦補習班打工仔"\'}, 
                    {\'title\': \'telunyang ( 楊德倫 ) · GitHub\', \'href\': \'https://github.com/telunyang\', \'body\': \'For years, I have dedicated myself to the development of several projects and the authoring of several papers in the field of Natural Language Processing (NLP). With the knowledge and …\'}
                ]', 
            name='web_search', 
            call_id='0', 
            is_error=False)
        ]
    )
    '''

    pprint(eval(response.chat_message.content))

# 呼叫主程式
asyncio.run(assistant_run())
