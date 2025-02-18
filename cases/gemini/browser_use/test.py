from langchain_google_genai import ChatGoogleGenerativeAI
from browser_use import Agent
import asyncio
import os
from pydantic import SecretStr
from dotenv import load_dotenv
load_dotenv(override=True)

api_key = os.getenv('GOOGLE_API_KEY')
if not api_key:
	raise ValueError('GOOGLE_API_KEY is not set')

llm = ChatGoogleGenerativeAI(
	model='gemini-1.5-flash-8b', 
	api_key=SecretStr(api_key)
)

async def main():
    agent = Agent(
        task="Go to Reddit, search for 'browser-use', click on the first post and return the first comment.",
        llm=llm,
    )
    result = await agent.run()
    print(result)

asyncio.run(main())