import os
import sys
import asyncio
from pathlib import Path
from pprint import pprint
import logging

from dotenv import load_dotenv
load_dotenv(override=True)

from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import SecretStr
from browser_use import Agent, Controller
from browser_use.agent.views import (
	ActionResult,
	AgentHistoryList
)
from browser_use.browser.browser import Browser, BrowserConfig
from browser_use.browser.context import (
	BrowserContext,
	BrowserContextConfig,
	BrowserContextWindowSize,
)

'''
Logging 設定
'''
# 基本設定
logger = logging.getLogger('browser_use')

# 設定等級
logger.setLevel(logging.INFO)

# 設定輸出格式
# formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', "%Y-%m-%d %H:%M:%S")
formatter = logging.Formatter('%(message)s')

# 儲存在 log 當中的事件處理器
fileHandler = logging.FileHandler('browser_use.log', mode='w', encoding='utf-8')
fileHandler.setFormatter(formatter)

# 輸出在終端機介面上的事件處理器
# console_handler = logging.StreamHandler()
# console_handler.setFormatter(formatter)

# 加入事件
# logger.addHandler(console_handler)
logger.addHandler(fileHandler)


# 輸入你的 Google API Key
api_key = os.getenv('GOOGLE_API_KEY')
if not api_key:
	raise ValueError('GOOGLE_API_KEY is not set')

# 建立 ChatGoogleGenerativeAI 物件
llm = ChatGoogleGenerativeAI(
	model='gemini-1.5-flash-8b', 
	api_key=SecretStr(api_key)
)

# 新增資料夾，用來存放檔案
# folderPath = os.path.join(os.getcwd(), 'files')
folderPath = './files'
if not os.path.exists(folderPath):
    os.makedirs(folderPath)

browser = Browser(
	config=BrowserConfig(
		# 備註: 若要自訂 chrome webdriver 路徑，要先關閉當前開啟的所有 Chrome 視窗，不然無法使用 Debug 模式
		# chrome_instance_path='./chromedriver.exe',

		# 設定下載檔案的路徑以及自動化控制瀏覽器的視窗大小
		new_context_config=BrowserContextConfig(
			save_downloads_path=folderPath,
			browser_window_size=BrowserContextWindowSize(width=1500, height=700)
		),
	)
)



# 進行搜尋
async def run_search(li_tasks: list):
	for index, task in enumerate(li_tasks):
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
	# 任務
	li_tasks = [
f'''
The tasks you have to follow sequentially are as follows:

### Task 1
Step 1. Go to the website "Google".
Step 2. Type the term "楊德倫" on search bar and press enter.
Step 3. Click the GitHub link.

### Task 2
Step 1. Click link "Repositories".
Step 2. Click the "first" repository link.
Step 3. You may try to scroll up or down to find the link "README.md" and click it.
Step 4. Summarize the content of "README.md" page.
Step 5. Task finished.
''',
f'''
The tasks you have to follow sequentially are as follows:

### Task 1
Step 1. Go to the website "DuckDuckGo".
Step 2. Type the term "楊德倫" on search bar and press enter.
Step 3. Click the GitHub link.

### Task 2
Step 1. Click link "Repositories".
Step 2. Click the "second" repository link.
Step 3. You may try to scroll up or down to find the link "README.md" and click it.
Step 4. Summarize the content of "README.md" page.
Step 5. Task finished.
''',
]

	asyncio.run(run_search(li_tasks))