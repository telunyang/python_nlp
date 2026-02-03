import os
import asyncio
from browser_use import Agent, Browser, ChatGoogle
from dotenv import load_dotenv
load_dotenv(override=True)



# 輸入你的 Google API Key
api_key = os.getenv('GOOGLE_API_KEY')
if not api_key:
	raise ValueError('GOOGLE_API_KEY is not set')

# 建立 LLM 物件
llm = ChatGoogle(
      model='gemini-2.5-flash-lite',
      api_key=api_key
)

# 新增資料夾，用來存放檔案
folderPath = os.path.join(os.getcwd(), 'files')
os.makedirs(folderPath, exist_ok=True)

# 建立 Browser 物件
browser = Browser(
	headless=False,  # 顯示瀏覽器視窗
	download_path=folderPath,  # 設定下載路徑
	window_size={'width': 1000, 'height': 700},  # 設定視窗大小
)

# 進行搜尋
async def run_search(task: str):
	agent = Agent(
		task=task, # 要完成的任務描述
		llm=llm, # 使用的 LLM 模型
		max_actions_per_step=50, # 每一步驟最多執行的動作數，所謂動作，是指瀏覽器的操作，例如點擊、輸入文字、滾動頁面等
		browser=browser, # 使用的瀏覽器物件
	)

	history = await agent.run(
		max_steps=100 # 最多執行的步驟數，意思是說，整個任務最多可以分成多少個步驟來完成
	)

	# logger.info('Final Result:')
	# logger.info(history.final_result())

	# logger.info('\nErrors:')
	# logger.info(history.errors())

	# logger.info('\nModel Outputs:')
	# logger.info(history.model_actions())

	# logger.info('\nThoughts:')
	# logger.info(history.model_thoughts())

if __name__ == '__main__':
	task = "查詢臺北到臺南的高鐵時刻表，並列出今天下午三點後的班次。"
	asyncio.run(run_search(task))