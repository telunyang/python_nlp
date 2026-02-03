from browser_use import Agent, ChatGoogle
import os
import asyncio
from dotenv import load_dotenv
load_dotenv(override=True)

api_key = os.getenv('GOOGLE_API_KEY')
if not api_key:
	raise ValueError('GOOGLE_API_KEY is not set')

llm = ChatGoogle(
      model='gemini-2.5-flash-lite',
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