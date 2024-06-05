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

### 控制燈泡開關(這裡以11為例-共6顆燈泡)
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

5. 取得並判斷辨識結果
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

### 儲存資料到Google sheet
### Server function  只要呼叫這個function就會開始運作
```
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
```
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

