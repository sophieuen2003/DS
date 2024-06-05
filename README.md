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
專題連結 (完整程式碼)：[ds_LLM_babyboss](https://colab.research.google.com/drive/1BF-IPPRmj68i8540-3rY8B-4iQWJ2vdc#scrollTo=z93fyaqfsxoL)

本專案旨在開發一個智能寶寶監控系統。<br>
目標利用影像辨識辨別寶寶狀態，同時抓取即時的溫濕度資訊並做出相對應控制電器的動作。最後使用Line Bot應用程式回應用戶的訊息並使用AI提供建議。

### 控制燈泡開關 (這裡以11號燈泡為例-共6顆燈泡)
```bash
import requests
# 11 on
def turn_11_on():
  # Turn light on
  url = 'http://211.21.113.190:8155/api/webhook/-hFNoCcZKB31gtiZzhIabeI0d'
  headers = {
      'Authorization': 'Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiI4YWM0MDEzODIwNDU0MDE0ODdjNzIwZTc2ZDBmYzdjYSIsImlhdCI6MTY5ODgwNzExNSwiZXhwIjoyMDE0MTY3MTE1fQ.7KaCwPUcjAr_zne04qili2fwQO1QoWTPzsmV1v_LLIc'
    }

  response = requests.post(url, headers=headers)
  return response.text

response_text_on = turn_11_on()  # 開啟第 11 盞燈
print(response_text_on)

# 11 off
def turn_11_off():
  # Turn light off
  url = 'http://211.21.113.190:8155/api/webhook/-qtfn4apfmywr78fHHNZYiclU'
  headers = {
      'Authorization': 'Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiI4YWM0MDEzODIwNDU0MDE0ODdjNzIwZTc2ZDBmYzdjYSIsImlhdCI6MTY5ODgwNzExNSwiZXhwIjoyMDE0MTY3MTE1fQ.7KaCwPUcjAr_zne04qili2fwQO1QoWTPzsmV1v_LLIc'
    }

  response = requests.post(url, headers=headers)
  return response.text

response_text_off = turn_11_off()  # 關閉第 11 盞燈
print(response_text_off)
```
### 函式

1. 情緒判斷：
```bash  
API_URL_1 = "https://api-inference.huggingface.co/models/trpakov/vit-face-expression"
headers = {"Authorization": "Bearer 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"}

def sad(filename):
  with open(filename, "rb") as f:
      data = f.read()
  response = requests.post(API_URL_1, headers=headers, data=data)
  return response.json()
```

2. 眼睛睜合：
```bash  
API_URL_2 = "https://api-inference.huggingface.co/models/dima806/closed_eyes_image_detection"
headers = {"Authorization": "Bearer hf_hf_MXjrVmNcgRCEDEeveDNuQKofzCVqTSztxb"}

def eyes(filename):
  with open(filename, "rb") as f:
      data = f.read()
  response = requests.post(API_URL_2, headers=headers, data=data)
  return response.json()
```

3. 臉部覆蓋：
```bash  
API_URL_3 = "https://api-inference.huggingface.co/models/AliGhiasvand86/gisha_coverd_uncoverd_face"
headers = {"Authorization": "Bearer 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"}

def face(filename):
  with open(filename, "rb") as f:
      data = f.read()
  response = requests.post(API_URL_3, headers=headers, data=data)
  return response.json()
```


4. 串接LLM
```bash
import requests

API_URL_LLM = "https://api-inference.huggingface.co/models/meta-llama/Meta-Llama-3-8B-Instruct"
headers = {"Authorization": "Bearer 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"}

def LLM(payload):
    try:
        response = requests.post(API_URL_LLM, headers=headers, json=payload)
        response.raise_for_status()  # 檢查請求是否成功
    except requests.exceptions.RequestException as e:
        print(f"HTTP請求失敗: {e}")
        return None

    try:
        return response.json()
    except requests.exceptions.JSONDecodeError as e:
        print(f"JSON解析失敗: {e}")
        return None
# https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct LLM使用的模組
```

5. 定義判斷後的結果
```bash
def checkCry(filename):
    emotion_probabilities = sad(filename)
    highest_emotion = max(emotion_probabilities, key=lambda x: x['score'])
    if highest_emotion['label'] == 'sad':
        return 1 # 傷心
    else:
        return 0 # 不傷心

def checkEyes(filename):
    result = eyes(filename)
    close_eye_probability = 0
    open_eye_probability = 0
    for prediction in result:
        if prediction["label"] == "closeEye":
            close_eye_probability = prediction["score"]
        elif prediction["label"] == "openEye":
            open_eye_probability = prediction["score"]

    if close_eye_probability > open_eye_probability:
        return 1 # 閉眼
    else:
        return 0


def checkFace(filename):
    result = face(filename)
    uncovered_probability = 0
    covered_probability = 0
    for prediction in result:
        if prediction["label"] == "uncovered":
            uncovered_probability = prediction["score"]
        elif prediction["label"] == "covered":
            covered_probability = prediction["score"]

    if covered_probability > uncovered_probability:
        return 1 # 臉被遮住
    else:
        return 0
```

### Server function 

1. 設定拍照參數
```bash
# function to convert the JavaScript object into an OpenCV image
def js_to_image(js_reply):
  """
  Params:
          js_reply: JavaScript object containing image from webcam
  Returns:
          img: OpenCV BGR image
  """
  # decode base64 image
  image_bytes = b64decode(js_reply.split(',')[1])
  # convert bytes to numpy array
  jpg_as_np = np.frombuffer(image_bytes, dtype=np.uint8)
  # decode numpy array into OpenCV BGR image
  img = cv2.imdecode(jpg_as_np, flags=1)

  return img

# function to convert OpenCV Rectangle bounding box image into base64 byte string to be overlayed on video stream
def bbox_to_bytes(bbox_array):
  """
  Params:
          bbox_array: Numpy array (pixels) containing rectangle to overlay on video stream.
  Returns:
        bytes: Base64 image byte string
  """
  # convert array into PIL image
  bbox_PIL = PIL.Image.fromarray(bbox_array, 'RGBA')
  iobuf = io.BytesIO()
  # format bbox into png for return
  bbox_PIL.save(iobuf, format='png')
  # format return string
  bbox_bytes = 'data:image/png;base64,{}'.format((str(b64encode(iobuf.getvalue()), 'utf-8')))

  return bbox_bytes
```

2.
```bash
face_cascade = cv2.CascadeClassifier(cv2.samples.findFile(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'))
def take_photo(filename='photo.jpg', quality=0.8):
  js = Javascript('''
    async function takePhoto(quality) {
      const div = document.createElement('div');
      const capture = document.createElement('button');
      capture.textContent = 'Capture';
      div.appendChild(capture);

      const video = document.createElement('video');
      video.style.display = 'block';
      const stream = await navigator.mediaDevices.getUserMedia({video: true});

      document.body.appendChild(div);
      div.appendChild(video);
      video.srcObject = stream;
      await video.play();

      // Resize the output to fit the video element.
      google.colab.output.setIframeHeight(document.documentElement.scrollHeight, true);

      // Wait for Capture to be clicked.
      await new Promise((resolve) => capture.onclick = resolve);

      const canvas = document.createElement('canvas');
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      canvas.getContext('2d').drawImage(video, 0, 0);
      stream.getVideoTracks()[0].stop();
      div.remove();
      return canvas.toDataURL('image/jpeg', quality);
    }
    ''')
  display(js)

  # get photo data
  data = eval_js('takePhoto({})'.format(quality))
  # get OpenCV format image
  img = js_to_image(data)
  # grayscale img
  gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
  print(gray.shape)
  # get face bounding box coordinates using Haar Cascade
  faces = face_cascade.detectMultiScale(gray)
  # draw face bounding box on image
  for (x,y,w,h) in faces:
      img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
  # save image
  cv2.imwrite(filename, img)

  return filename
```

3.儲存拍攝照片
```bash
try:
  filename = take_photo('photo.jpg')
  print('Saved to {}'.format(filename))

  # Show the image which was just taken.
  display(Image(filename))
except Exception as err:
  # Errors will be thrown if the user does not have a webcam or if they do not
  # grant the page permission to access it.
  print(str(err))
```

### HW3 外部資訊 
串聯外部API，獲得使用者的經緯度後，去抓當地的天氣資料，並根據溫度判斷是否要開冷氣
```bash
import requests
import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime
from zoneinfo import ZoneInfo

# 假設我們有一個分析影像的函數
def analyze_image(img):
    # 在此處新增圖像分析程式碼並返回適當的狀態
    # 範例回傳值
    if checkFace(img) == 0:
        if checkCry(img) == 0:
          if checkEyes(img) == 0:
            return "The baby is awake!"
          elif checkEyes(img) == 1:
            return "The baby is sleeping!"
        elif checkCry(img) == 1:
          return "The baby is crying!"
    elif checkFace(img) == 1:
      return "Warning!The baby's face is covered!"

def kelvin_to_celsius(kelvin):
    return kelvin - 273.15

def get_weather_by_coordinates(api_key, lat, lon):
    url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        return data
    else:
        return {"error": f"Failed to retrieve data. Status code: {response.status_code}"}

def write_to_google_sheets(sheet_id, row_data):
    scope = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive.file']
    creds = Credentials.from_service_account_file('/content/llmtwins.json', scopes=scope)
    client = gspread.authorize(creds)

    sheet = client.open_by_key(sheet_id).sheet1

    # 清空現有數據
    #sheet.clear()

    # 新增標題行
    #headers = ["日期", "時間", "寶寶狀態", "溫度 (°C)", "濕度 (%)", "城市", "天氣描述"]
    #sheet.append_row(headers)

    # 新增數據
    sheet.append_row(row_data)

def store_data(api_key, lat, lon, sheet_id, img):
    # 取得影像分析結果
    status = analyze_image(img)

    # 取得天氣資訊
    weather_data = get_weather_by_coordinates(api_key, lat, lon)

    # 檢查是否有錯誤
    if "error" in weather_data:
        print(weather_data["error"])
        return

    temperature_celsius = round(kelvin_to_celsius(weather_data["main"]["temp"]), 1)
    humidity = weather_data["main"]["humidity"]
    city = weather_data["name"]
    description = weather_data["weather"][0]["description"]

    tz = ZoneInfo("Asia/Taipei")

    # 判斷是否要開冷氣
    if temperature_celsius >= 28:
      turn_15_on()
      AC = "On"
    elif temperature_celsius < 28:
      turn_15_off()
      AC = "Off"

    payload = {
    "inputs": f"I am a babysitter, if I found the situation that {analyze_image(img)} How can I solve? Don't repeat my question.Just give me the adivces.\n"
    }
    suggestion = LLM(payload)
    print(suggestion)
    if isinstance(suggestion, list) and len(suggestion) > 0 and 'generated_text' in suggestion[0]:
        generated_text = suggestion[0]['generated_text']
    else:
        generated_text = "No suggestion available"

    # 提取建議部分
    advice = generated_text.split('\n', 1)[-1].strip()

    # 取得當前日期和時間
    today_date = datetime.now(tz).strftime('%Y-%m-%d')
    today_time = datetime.now(tz).strftime('%H:%M:%S')

    # 準備寫入 Google Sheets 的數據
    row_data = [today_date, today_time, status, temperature_celsius, humidity, city, description, AC, advice]

    # 將資料寫入 Google Sheets
    write_to_google_sheets(sheet_id, row_data)
```

依據HuggingFace回傳的判斷結果及openweathermap抓到溫度和濕度資訊做出相對應的動作
```bash
import time

# 替換為你的 API Key
api_key = "b55132e7d0d008aa945b18e4cd2f5a02"

# 縮寫經緯度
latitude = 25.027383
longitude = 121.529965

# Google Sheet ID
sheet_id = "1UabozEXhPsifoI15OqkWUJ-6ljlQmnaHZJDHYVvHTA0"

# 呼叫函數處理數據
def start():
    global img
    img = '/content/photo.jpg'
    checkFace(img)
    checkCry(img)
    checkEyes(img)
    time.sleep(3)
    while True:
      store_data(api_key, latitude, longitude, sheet_id, img)
      if checkFace(img) == 0:
        turn_14_off()
        if checkCry(img) == 0:
          turn_11_off()
          if checkEyes(img) == 0:
            turn_13_on()
            break
          elif checkEyes(img) == 1:
            turn_12_on()
            break
        elif checkCry(img) == 1:
          turn_11_on()
          break
      elif checkFace(img) == 1:
          turn_14_on()
          break
```

執行程式
```bash
start()
```

### HW4 Line bot
```bash
import gspread
from google.oauth2.service_account import Credentials

def getBaby():
  scope = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive.file']

  creds = Credentials.from_service_account_file('/content/llmtwins.json', scopes=scope)
  client = gspread.authorize(creds)

  sheet_id = "1UabozEXhPsifoI15OqkWUJ-6ljlQmnaHZJDHYVvHTA0"
  sheet = client.open_by_key(sheet_id).sheet1
  # 取得整張表的數據
  all_data = sheet.get_all_values()

  last_non_empty_row = None
  for row in all_data[1:]:  # 跳過標題行
      if row[2]:  # C列是第3列（索引為2）
          last_non_empty_row = row[2]
  return last_non_empty_row

def getAdvice():

  scope = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive.file']

  creds = Credentials.from_service_account_file('/content/llmtwins.json', scopes=scope)
  client = gspread.authorize(creds)

  sheet_id = "1UabozEXhPsifoI15OqkWUJ-6ljlQmnaHZJDHYVvHTA0"
  sheet = client.open_by_key(sheet_id).sheet1
  # 取得整張表的數據
  all_data = sheet.get_all_values()

  last_non_empty_row = None
  for row in all_data[1:]:  # 跳過標題行
      if row[8]:  # I列是第9列（索引為8）
          last_non_empty_row = row[8]
  return last_non_empty_row
```
```bash
from google.colab import drive
drive.mount('/content/drive', force_remount=True)

!mkdir -p /drive
#umount /drive
!mount --bind /content/drive/My\ Drive /drive
!mkdir -p /drive/ngrok-ssh
!mkdir -p ~/.ssh
```

```bash
import getpass

from pyngrok import ngrok, conf

print("Enter your authtoken, which can be copied from https://dashboard.ngrok.com/auth")
conf.get_default().auth_token = getpass.getpass()

# Open a TCP ngrok tunnel to the SSH server
connection_string = ngrok.connect("22", "tcp").public_url

ssh_url, port = connection_string.strip("tcp://").split(":")
print(f" * ngrok tunnel available, access with `ssh root@{ssh_url} -p{port}`")
```

```bash
from flask import Flask, request
from pyngrok import ngrok   # Colab 環境需要，本機環境不需要
import json, time, requests

# 載入 LINE Message API 相關函式庫
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage, StickerSendMessage, ImageSendMessage, LocationSendMessage


app = Flask(__name__)

# Colab 環境需要下面這三行，本機環境不需要
port = "5000"
public_url = ngrok.connect(port).public_url
print(f" * ngrok tunnel \"{public_url}\" -> \"http://127.0.0.1:{port}\" ")

access_token = 'MyQEPwE3fRdpUtTDusHkaeW4E2gvQFvEAqGG+XcEmYUAAJD4LD7ohZ0v0sfl7NVkQ6BNbctuiKZ1pGlHD0eG84SadNHkB8HF+XGP5rxlLEfP+69UaOMfWxcVBeS0PEwsDpL6VXRPNsezhfyuMO5XRwdB04t89/1O/w1cDnyilFU='
channel_secret = '69ad2b3b47650aec61e4d53c1814f1c8'

@app.route("/", methods=['POST'])
def linebot():
    body = request.get_data(as_text=True)                    # 取得收到的訊息內容
    try:
        line_bot_api = LineBotApi(access_token)     # 確認 token 是否正確
        handler = WebhookHandler(channel_secret)    # 確認 secret 是否正確
        signature = request.headers['X-Line-Signature']             # 加入回傳的 headers
        handler.handle(body, signature)      # 綁定訊息回傳的相關資訊
        json_data = json.loads(body)         # 轉換內容為 json 格式
        reply_token = json_data['events'][0]['replyToken']    # 取得回傳訊息的 Token ( reply message 使用 )
        user_id = json_data['events'][0]['source']['userId']  # 取得使用者 ID ( push message 使用 )
        print(json_data)
        msg = json_data['events'][0]['message']['text']                 # 印出內容
        type = json_data['events'][0]['message']['type']
        if type == 'text':
            if msg == "寶寶狀態":
              text = json_data['events'][0]['message']['text']
              text_message = TextSendMessage(text=getBaby())
              line_bot_api.reply_message(reply_token,text_message)
            elif msg == "建議":
              text = json_data['events'][0]['message']['text']
              text_message = TextSendMessage(text=getAdvice())
              line_bot_api.reply_message(reply_token,text_message)
    except Exception as e:
        print(e)
    return 'OK'                 # 驗證 Webhook 使用，不能省略

if __name__ == "__main__":
    app.run()
```

### 授權

此專案採用 MIT 授權條款。詳見 [LICENSE](LICENSE) 文件。

### 聯絡作者

- [溫苡含](https://github.com/sophieuen2003/DS)
- [林元方](https://github.com/Duckucy/112-2-Data-Structure)
- [楊思瑜](https://github.com/szuyu830)
- [詹喬崴](https://github.com/chiaoweichan/Data-Strucure)

