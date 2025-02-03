# python_nlp
Python 自然語言處理講義與範例

## 提問
- 通則
  - 「結業前」可提問、討論，要把多餘時間和資源，留給當前上課的學員。
- 寫信
	- E-mail: `darren@darreninfo.cc`
	- 信件標題寫上你的**班別和姓名**，或是在哪裡參與我的課程，例如 `[資展 BDSEXX / 臺大計中 / 聯成]` 你的主旨 ○○○。
	- 提問的內容要與本專案有關，**其它課程的部分，去請益原本授課的老師**。
	- **不要把程式碼寄給我**，可能沒時間看，討論儘量以解決問題的方向為主。
	- 不符合以上幾點，將**直接刪除**，敬請見諒。

## 作業
- 僅限授課學員。
- 同學之間可以互相討論，但千萬不要抄襲。
- 修改 `bert_finetue` 的範例，從二元分類，改成多元分類，使用的資料集如下：
  - [Datasets:Johnson8187/Chinese_Multi-Emotion_Dialogue_Dataset](https://huggingface.co/datasets/Johnson8187/Chinese_Multi-Emotion_Dialogue_Dataset)
  - 下載資料集的方法：
    - 按下頁面右邊的 `Use this dataset`，然後選擇合適的 `library`，例如 `pandas`，然後複製官方提供的範例。
	- 按下 `Files and versions`，裡面有 `data.csv`，按下 `Download file` (一個下載的 icon)，可以直接另存新檔到硬碟當中。
  - labels 轉換的方法：
	- 使用 `sklearn.preprocessing.LabelEncoder` 來轉換。`也可以自訂轉換方式`
		```
		例如:
		0: "平淡語氣",
		1: "關切語調",
		2: "開心語調",
		3: "憤怒語調",
		4: "悲傷語調",
		5: "疑問語調",
		6: "驚奇語調",
		7: "厭惡語調"
		```
	- 轉換後的 labels 會是 `0` 到 `7`。
  - 預測以下文字，並且輸出預測的結果，以及 confidence:
    ```
	texts = [
        "我每天都能跟她一起上學，我好開心！",
		"最好的朋友要離開臺灣了，以後可能不容易再見面...",
		"我覺得我快不行了...",
		"剛剛收到研究所錄取的通知書！",
		"今年的冬天好像比較晚來。"
    ]
	```
	- 範例:
	  ```
	  他們兩個竟然牽手了! => 驚奇語調 (0.8)
	  有人在背後說我胖！幹! => 憤怒語調 (0.9)
	  我媽終於要讓我養狗了 => 開心語調 (0.8)
	  ...
	  ```
  - `80` 分條件
    - 讀取自行微調 (finetune) 好的 bert 模型，並且預測以上的文字。
	- 不用給我看程式碼，只要錄製的時候執行進行預測 (不需要錄如何微調)，按照範例來顯示結果。
  - `100` 分條件 (基於 `80` 分條件)
      - 使用 `GitHub` 平台來提交作業，並且將 `github repo 連結` 以及 `影片連結` 連結寄給我。
        - Git 與 GitHub 使用教學: [程式與網頁開發者必備技能！Git 和 GitHub 零基礎快速上手，輕鬆掌握版本控制的要訣！](https://www.youtube.com/watch?v=FKXRiAiQFiY)
        - Markdown 語法: [如何使用 Markdown 語言撰寫技術文件](https://experienceleague.adobe.com/zh-hant/docs/contributor/contributor-guide/writing-essentials/markdown)
      - `repository` 裡面至少要有 `finetune.ipynb` 或 `finetune.py`，`predict.ipynb` 或 `predict.py`，還有 `README.md`，最重要的是你微調後的模型 `output` 資料夾。
	    - 上傳大型檔案到 github 上，請參考：
		  - [Git Large File Storage - An open source Git extension for versioning large files](https://git-lfs.com/)
		  - [我如何使用 Git LFS 來託付大型 Git 檔案？](https://www.webdong.dev/zh-tw/post/how-i-use-git-lfs-to-manage-large-git-files/)
        ```
		output/ (這裡放置你微調後的模型)
        finetune.ipynb (或 .py，微調用)
		predict.ipynb (或 .py，預測用)
        README.md
        ```
      - `README.md` 要有說明 (用 `.py` 執行要額外說明執行指令或方法)，例如:
        ```markdown
        # 中文句子情緒分類

		## 訓練資料來源
		- [Datasets:Johnson8187/Chinese_Multi-Emotion_Dialogue_Dataset](https://huggingface.co/datasets/Johnson8187/Chinese_Multi-Emotion_Dialogue_Dataset)

		## 基礎模型
		- [google-bert/bert-base-chinese](https://huggingface.co/google-bert/bert-base-chinese)

        ## 安裝套件
        - torch (版本號)
		- torchvision (版本號)
		- torchaudio (版本號)
		- transformers (版本號)
		- datasets (版本號)
		- evaluate (版本號)
		- accelerate (版本號)
		- scikit-learn (版本號)
		(版本號可用 pip list，或是 conda list 來檢視)
        ...

		## 說明
		(介紹你微調後的模型，主要用來做什麼的，例如你使用模型進行情緒分類，分成幾類…等等，再放上作業要求的 texts 預測結果，自由發揮)

        ## 成果
        ![](執行過程的擷圖或說明圖片)
        ...
        [影片名稱或其它標題](你的影片連結)
        ...

        ## 其它你想要補充標題和內容
        ...
        ...
        ```
	  - 可以參考以前學長的 README 撰寫方式: [FaceBook FanPage Scraper with selenium](https://github.com/nana89823/facebook_scraper)
    - 沒交：`0` 分。
- 繳交時間
  - 原則上最後一堂課結束後 2 週內，準確時間上課說明。