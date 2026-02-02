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
    tast = "到網路搜尋 IG 名為「擺渡人_楊德倫」的帳號，找到帳號的簽名檔，並回傳簽名檔的內容。"
    agent = Agent(
        task=tast,
        llm=llm,
    )
    result = await agent.run()
    print(result)

if __name__ == "__main__":
    asyncio.run(main())