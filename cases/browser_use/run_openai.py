from browser_use import Agent, ChatOpenAI
import os
import asyncio
from dotenv import load_dotenv
load_dotenv(override=True)

api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
	raise ValueError('OPENAI_API_KEY is not set')

llm = ChatOpenAI(
      model='gpt-4o-mini',
      api_key=api_key
)

async def main():
    task = "查詢臺北到臺南的高鐵時刻表，並列出今天下午三點後的班次。"
    agent = Agent(
        task=task,
        llm=llm,
    )
    result = await agent.run()
    print(result)

if __name__ == "__main__":
    asyncio.run(main())