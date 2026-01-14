import os
import asyncio

from dotenv import load_dotenv
load_dotenv(override=True)

from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import SecretStr
from browser_use import Agent
from browser_use.agent.views import (
	AgentHistoryList
)
from browser_use.browser.browser import Browser, BrowserConfig
from browser_use.browser.context import (
	BrowserContextConfig,
	BrowserContextWindowSize,
)


# 輸入你的 Google API Key
api_key = os.getenv('GOOGLE_API_KEY')
if not api_key:
	raise ValueError('GOOGLE_API_KEY is not set')

# 建立 ChatGoogleGenerativeAI 物件
llm = ChatGoogleGenerativeAI(
	model='gemini-2.0-flash-lite', 
	api_key=SecretStr(api_key)
)

# 新增資料夾，用來存放檔案
# folderPath = os.path.join(os.getcwd(), 'files')
folderPath = './files'
os.makedirs(folderPath, exist_ok=True)

# 建立 Browser 物件
browser = Browser(
	config=BrowserConfig(
		# 設定下載檔案的路徑以及自動化控制瀏覽器的視窗大小
		new_context_config=BrowserContextConfig(
			save_downloads_path=folderPath,
			browser_window_size=BrowserContextWindowSize(width=1500, height=700)
		),
	)
)

# 進行搜尋
async def run_search(task: str):
	agent = Agent(
		task=task,
		llm=llm,
		max_actions_per_step=50,
		browser=browser,
	)

	history: AgentHistoryList = await agent.run(max_steps=100)

	# logger.info('Final Result:')
	# logger.info(history.final_result())

	# logger.info('\nErrors:')
	# logger.info(history.errors())

	# logger.info('\nModel Outputs:')
	# logger.info(history.model_actions())

	# logger.info('\nThoughts:')
	# logger.info(history.model_thoughts())

	# 關閉瀏覽器
	await browser.close()



if __name__ == '__main__':
	asyncio.run(run_search("Go to the website \"Google\". Type the term \"楊德倫\" on search bar and press enter. Click the GitHub link."))