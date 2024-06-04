# 112-2 師大科技系資料結構
---

### 授課教師：蔡芸琤
### 姓名：溫苡含
### 系級：科技系2年級

## 課程筆記區
  3/12<br>
  - 二元樹 Binary Tree<br>
    每個節點最多只有兩個分支，分支被稱為左子樹及右子樹
    
    - 二元樹種類
      - perfect binary tree<br>
        各層節點全滿。
        
      - complete binary tree<br>
        一棵樹的node按照Full Binary Tree的次序排列(由上至下，由左至右)。
        
    - 二元樹走訪
      - 前序遍歷 Preorder Traversal<br>
        順序-根、左子樹、右子樹

      - 中序遍歷 Inorder Traversal<br>
        順序-左子樹、根、右子樹

      - 後序遍歷 Postorder Traversal<br>
        順序-左子樹、右子樹、根

      - 層序遍歷 Level-order Traversal<br>
        順序-由根節點一層一層往下，由左往右


## 專題連結區
**Final Project**

專題名稱：Infant Caregiver<br>
專題連結：[ds_LLM_babyboss](https://colab.research.google.com/drive/1BF-IPPRmj68i8540-3rY8B-4iQWJ2vdc#scrollTo=z93fyaqfsxoL)

本專案旨在開發一個智能寶寶監控系統。<br>
目標利用影像辨識辨別寶寶狀態，同時抓取即時的溫濕度資訊並做出相對應控制電器的動作。最後使用Line Bot應用程式回應用戶的訊息並使用AI提供建議。

### 需求
```bash
- requests
- drive
- from IPython.display import display, Javascript, Image
- from google.colab.output import eval_js
- from base64 import b64decode, b64encode
- cv2
- import numpy as np
- import PIL
- import io
- import html
- import time
- gspread
- from google.oauth2.service_account import Credentials
- from datetime import datetime
- from zoneinfo import ZoneInfo
- getpass
- from pyngrok import ngrok, conf
- from linebot import LineBotApi, WebhookHandler
- from linebot.exceptions import InvalidSignatureError
- from linebot.models import MessageEvent, TextMessage, TextSendMessage, StickerSendMessage, ImageSendMessage, LocationSendMessage
- line-bot-sdk
```
### 安裝

1. 安裝所需套件：
```bash
    pip install Flask pyngrok line-bot-sdk requests
    
    pip install opencv-python

    pip install fastapi

    pip install line-bot-sdk

    pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib

    pip install pyngrok
```

2. 確保您已經擁有 Goole API及Line Bot 的存取權杖和密鑰，並將其替換至 access_token 和 channel_secret 變數中。

### 使用

1. 啟動應用程式：

```bash
    python ds_llm_babyboss.py
``` 

2. 應用程式將會使用 ngrok 建立一個公共 URL。請記住這個 URL，並將其設定在您的 Line Bot Webhook 設定中。

### 功能

- **影像辨識**：使用 Hugging Face 的模組，透過影像辨識擷取當前寶寶狀態，並透過亮起的燈號，表示控制對應的電器 ( 以電燈作為代替 )

     嬰兒哭泣：Light 11 ( 以亮燈表示響起溫和的警報聲 - **提醒照顧者小孩醒了** )
  
     嬰兒睡覺：Light 12 ( 以亮燈表示關閉房間電燈、播放搖籃曲及搖嬰兒的搖籃 - **幫助小孩更好的入眠** )

     嬰兒起床：Light 13 ( 以亮燈表示開啟房間電燈及窗簾， - **提醒照顧者小孩已經醒了** )
  
     嬰兒面部朝下：Light 14 (以亮燈表示響起危險的警報聲 - **狀況十分危急，需要立即處理**  )<br>
  
- **溫濕度**：抓取openweathermap以獲取當前地點的溫度和濕度資訊，若 擷取到的溫度 ≥ 28ﾟC 就開啟冷氣

     溫度 ≥ 28ﾟC：Light 15 (以亮燈表示開冷氣 )
  
- **Line Bot**：串接LLM整理AI給的建議，當使用者在聊天室傳送 "寶寶狀態" ，可以獲取寶寶的當前狀態；傳送 "建議" 可以獲取AI的建議。


### 結構
- 各組功能funciton定義
- Server function - 呼叫這個function就會開始運作
- 影像辨識
- 抓取地點、氣溫、濕度，並根據溫度自動開關空調
- start()
- 儲存資料到Google sheet
- 根據接收到的訊息，決定回應的內容並使用 Line Bot API 回應用戶。

### 授權

此專案採用 MIT 授權條款。詳見 [LICENSE](LICENSE) 文件。

### 聯絡作者

- [溫苡含](https://github.com/sophieuen2003/DS)
- [林元方](https://github.com/Duckucy/112-2-Data-Structure)
- [楊思瑜](https://github.com/szuyu830)
- [詹喬崴](https://github.com/chiaoweichan/Data-Strucure)

