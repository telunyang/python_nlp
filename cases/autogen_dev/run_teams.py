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

# 建立 player
player_agent = AssistantAgent(
    "player",
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

# 建立一個團隊，由 主要 和 評論 這兩個代理組成。
team = RoundRobinGroupChat([
	    player_agent, 
		critic_agent
	], 
	termination_condition=text_termination
)

# 執行團隊任務
async def team_run() -> None:
    result = await team.run(task="Write a short poem about the fall season.")
    pprint(result)
    '''
    TaskResult(
        messages=[
            TextMessage(
                source='user', 
                models_usage=None, 
                content='Write a short poem about the fall season.', 
                type='TextMessage'
            ),
            TextMessage(
                source='player', 
                models_usage=RequestUsage(
                    prompt_tokens=16, 
                    completion_tokens=93
                ), 
                content="Crimson leaves descend, a gentle rain,\nCrisp air whispers secrets, a golden refrain.\nSummer's warmth fades, a bittersweet adieu,\nAutumn's hues ignite, a vibrant, fiery hue.\nPumpkins plump and squash, a harvest's sweet embrace,\nNature's final flourish, time and space.\nCooler nights arrive, with cozy, warm delight,\nFall's embrace is welcome, both day and night.\n", 
                type='TextMessage'
            ),
            TextMessage(
                source='critic', 
                models_usage=RequestUsage(
                    prompt_tokens=119, 
                    completion_tokens=3
                ), 
                content='APPROVE\n', 
                type='TextMessage'
            )
        ],
        stop_reason="Text 'APPROVE' mentioned"
    )
    '''

    '''
    生成結果：
    Crimson leaves descend, a gentle rain, Crisp air whispers secrets, a golden refrain. 
    Summer's warmth fades, a bittersweet adieu, Autumn's hues ignite, a vibrant, fiery hue. 
    Pumpkins plump and squash, a harvest's sweet embrace, Nature's final flourish, time and space. 
    Cooler nights arrive, with cozy, warm delight, Fall's embrace is welcome, both day and night.
    
    深紅的樹葉落下，一場輕柔的雨，清爽的空氣低語著秘密，金色的副歌。
    夏天的溫暖褪去，苦樂參半的告別，秋天的色彩點燃，充滿活力、火熱的色彩。
    南瓜飽滿而擠壓，豐收的甜蜜擁抱，大自然最後的繁榮，時間和空間。
    涼爽的夜晚來臨，帶著舒適、溫暖的喜悅，無論白天或夜晚，都歡迎秋天的擁抱。
    '''


# 主程式
asyncio.run(team_run())