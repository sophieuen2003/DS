# -*- coding: utf-8 -*-
"""
DS_LLM-BabyBoss.ipynb
"""

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

# 12 on
def turn_12_on():
    url = 'http://211.21.113.190:8155/api/webhook/-TJO7MQn5u--KlqSH4Mw2JHA7'
    headers = {
        'Authorization': 'Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiI4YWM0MDEzODIwNDU0MDE0ODdjNzIwZTc2ZDBmYzdjYSIsImlhdCI6MTY5ODgwNzExNSwiZXhwIjoyMDE0MTY3MTE1fQ.7KaCwPUcjAr_zne04qili2fwQO1QoWTPzsmV1v_LLIc'
    }

    response = requests.post(url, headers=headers)
    return response.text

response_text_on = turn_12_on()  # 開啟第 12 盞燈
print(response_text_on)

# 12 off
def turn_12_off():
    url = 'http://211.21.113.190:8155/api/webhook/-jAxX99jM2ghu4SVD29Ht8Flx'
    headers = {
        'Authorization': 'Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiI4YWM0MDEzODIwNDU0MDE0ODdjNzIwZTc2ZDBmYzdjYSIsImlhdCI6MTY5ODgwNzExNSwiZXhwIjoyMDE0MTY3MTE1fQ.7KaCwPUcjAr_zne04qili2fwQO1QoWTPzsmV1v_LLIc'
    }

    response = requests.post(url, headers=headers)
    return response.text

response_text_off = turn_12_off()  # 關閉第 12 盞燈
print(response_text_off)

# 13 on
def turn_13_on():
    url = 'http://211.21.113.190:8155/api/webhook/-GB_PabGDpQlRGcGChEhun6uj'
    headers = {
        'Authorization': 'Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiI4YWM0MDEzODIwNDU0MDE0ODdjNzIwZTc2ZDBmYzdjYSIsImlhdCI6MTY5ODgwNzExNSwiZXhwIjoyMDE0MTY3MTE1fQ.7KaCwPUcjAr_zne04qili2fwQO1QoWTPzsmV1v_LLIc'
    }

    response = requests.post(url, headers=headers)
    return response.text

# 使用函式
response_text_on = turn_13_on()  # 開啟第 13 盞燈
print(response_text_on)

# 13 off
def turn_13_off():
    url = 'http://211.21.113.190:8155/api/webhook/1-3-off-fgzskQbLSxNVl3_SpTPlo7QP'
    headers = {
        'Authorization': 'Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiI4YWM0MDEzODIwNDU0MDE0ODdjNzIwZTc2ZDBmYzdjYSIsImlhdCI6MTY5ODgwNzExNSwiZXhwIjoyMDE0MTY3MTE1fQ.7KaCwPUcjAr_zne04qili2fwQO1QoWTPzsmV1v_LLIc'
    }

    response = requests.post(url, headers=headers)
    return response.text

response_text_off = turn_13_off()  # 關閉第 13 盞燈
print(response_text_off)

# 14 on
def turn_14_on():
    url = 'http://211.21.113.190:8155/api/webhook/-lkvcCfPU2wOmNLvhDjrEKpkb'
    headers = {
        'Authorization': 'Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiI4YWM0MDEzODIwNDU0MDE0ODdjNzIwZTc2ZDBmYzdjYSIsImlhdCI6MTY5ODgwNzExNSwiZXhwIjoyMDE0MTY3MTE1fQ.7KaCwPUcjAr_zne04qili2fwQO1QoWTPzsmV1v_LLIc'
    }

    response = requests.post(url, headers=headers)
    return response.text

response_text_on = turn_14_on()  # 開啟第 14 盞燈
print(response_text_on)

# 14 off
def turn_14_off():
    url = 'http://211.21.113.190:8155/api/webhook/-jfyKgpRXg6fgI9IXNIy0GNSn'
    headers = {
        'Authorization': 'Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiI4YWM0MDEzODIwNDU0MDE0ODdjNzIwZTc2ZDBmYzdjYSIsImlhdCI6MTY5ODgwNzExNSwiZXhwIjoyMDE0MTY3MTE1fQ.7KaCwPUcjAr_zne04qili2fwQO1QoWTPzsmV1v_LLIc'
    }

    response = requests.post(url, headers=headers)
    return response.text

response_text_off = turn_14_off()  # 關閉第 14 盞燈
print(response_text_off)

# 15 on
def turn_15_on():
    url = 'http://211.21.113.190:8155/api/webhook/-TqF-jZh-M8QqFBEnfgXEAxY7'
    headers = {
        'Authorization': 'Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiI4YWM0MDEzODIwNDU0MDE0ODdjNzIwZTc2ZDBmYzdjYSIsImlhdCI6MTY5ODgwNzExNSwiZXhwIjoyMDE0MTY3MTE1fQ.7KaCwPUcjAr_zne04qili2fwQO1QoWTPzsmV1v_LLIc'
    }

    response = requests.post(url, headers=headers)
    return response.text

response_text_on = turn_15_on()  # 開啟第 15 盞燈
print(response_text_on)

# 15 off
def turn_15_off():
    url = 'http://211.21.113.190:8155/api/webhook/--gONieqwhFofJ_6WjVAqxZtg'
    headers = {
        'Authorization': 'Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiI4YWM0MDEzODIwNDU0MDE0ODdjNzIwZTc2ZDBmYzdjYSIsImlhdCI6MTY5ODgwNzExNSwiZXhwIjoyMDE0MTY3MTE1fQ.7KaCwPUcjAr_zne04qili2fwQO1QoWTPzsmV1v_LLIc'
    }

    response = requests.post(url, headers=headers)
    return response.text

response_text_off = turn_15_off()  # 關閉第 15 盞燈
print(response_text_off)

# 16 on
def turn_16_on():
    url = 'http://211.21.113.190:8155/api/webhook/-d2Qi-uvQnpb_KcVcp_Jm9iJK'
    headers = {
        'Authorization': 'Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiI4YWM0MDEzODIwNDU0MDE0ODdjNzIwZTc2ZDBmYzdjYSIsImlhdCI6MTY5ODgwNzExNSwiZXhwIjoyMDE0MTY3MTE1fQ.7KaCwPUcjAr_zne04qili2fwQO1QoWTPzsmV1v_LLIc'
    }

    response = requests.post(url, headers=headers)
    return response.text

response_text_on = turn_16_on()  # 開啟第 16 盞燈
print(response_text_on)

# 16 off
def turn_16_off():
    url = 'http://211.21.113.190:8155/api/webhook/-pAW1x-AJO9s-b9JXDm89Dp_P'
    headers = {
        'Authorization': 'Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiI4YWM0MDEzODIwNDU0MDE0ODdjNzIwZTc2ZDBmYzdjYSIsImlhdCI6MTY5ODgwNzExNSwiZXhwIjoyMDE0MTY3MTE1fQ.7KaCwPUcjAr_zne04qili2fwQO1QoWTPzsmV1v_LLIc'
    }

    response = requests.post(url, headers=headers)
    return response.text

response_text_off = turn_16_off()  # 關閉第 16 盞燈
print(response_text_off)

"""2. 各組功能funciton定義（須依照自己的情境改寫，以下是pseudocode--以檢查跌倒為例）"""

import requests

#情緒判斷

API_URL_1 = "https://api-inference.huggingface.co/models/trpakov/vit-face-expression"
headers = {"Authorization": "Bearer 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"}

def sad(filename):
    with open(filename, "rb") as f:
        data = f.read()
    response = requests.post(API_URL_1, headers=headers, data=data)
    return response.json()

#眼睛睜合

API_URL_2 = "https://api-inference.huggingface.co/models/dima806/closed_eyes_image_detection"
headers = {"Authorization": "Bearer hf_hf_MXjrVmNcgRCEDEeveDNuQKofzCVqTSztxb"}


def eyes(filename):
    with open(filename, "rb") as f:
        data = f.read()
    response = requests.post(API_URL_2, headers=headers, data=data)
    return response.json()

#臉部覆蓋

API_URL_3 = "https://api-inference.huggingface.co/models/AliGhiasvand86/gisha_coverd_uncoverd_face"
headers = {"Authorization": "Bearer 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"}

def face(filename):
    with open(filename, "rb") as f:
        data = f.read()
    response = requests.post(API_URL_3, headers=headers, data=data)
    return response.json()

#串接LLM

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

#https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct LLM

from google.colab import drive
drive.mount('/content/drive')

"""3. 儲存資料到Google sheet

4. Server function - 呼叫這個function就會開始運作（須依照自己的情境改寫，以下是pseudocode）
"""
"""
!pip install opencv-python
"""

from IPython.display import display, Javascript, Image
from google.colab.output import eval_js
from base64 import b64decode, b64encode
import cv2
import numpy as np
import PIL
import io
import html
import time

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

try:
    filename = take_photo('photo.jpg')
    print('Saved to {}'.format(filename))

    # Show the image which was just taken.
    display(Image(filename))
except Exception as err:
    # Errors will be thrown if the user does not have a webcam or if they do not
    # grant the page permission to access it.
    print(str(err))

# HW3 外部資訊

import requests
import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime
from zoneinfo import ZoneInfo

# 假设我们有一个分析图像的函数
def analyze_image(img):
    # 在此处添加图像分析代码并返回适当的状态
    # 示例返回值
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

    # 添加標題行
    #headers = ["日期", "時間", "寶寶狀態", "溫度 (°C)", "濕度 (%)", "城市", "天氣描述"]
    #sheet.append_row(headers)

    # 添加新數據
    sheet.append_row(row_data)

def store_data(api_key, lat, lon, sheet_id, img):
    # 获取图像分析结果
    status = analyze_image(img)

    # 获取天气信息
    weather_data = get_weather_by_coordinates(api_key, lat, lon)

    # 检查是否有错误
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

    # 提取建议部分
    advice = generated_text.split('\n', 1)[-1].strip()

    # 获取当前日期和时间
    today_date = datetime.now(tz).strftime('%Y-%m-%d')
    today_time = datetime.now(tz).strftime('%H:%M:%S')

    # 准备写入 Google Sheets 的数据
    row_data = [today_date, today_time, status, temperature_celsius, humidity, city, description, AC, advice]

    # 将数据写入 Google Sheets
    write_to_google_sheets(sheet_id, row_data)

import time

# 替换为你的 API Key
api_key = "b55132e7d0d008aa945b18e4cd2f5a02"

# 縮寫經緯度
latitude = 25.027383
longitude = 121.529965

# Google Sheet ID
sheet_id = "1UabozEXhPsifoI15OqkWUJ-6ljlQmnaHZJDHYVvHTA0"

# 调用函数处理数据
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

start()

import gspread
from google.oauth2.service_account import Credentials

def getBaby():
    scope = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive.file']

    creds = Credentials.from_service_account_file('/content/llmtwins.json', scopes=scope)
    client = gspread.authorize(creds)

    sheet_id = "1UabozEXhPsifoI15OqkWUJ-6ljlQmnaHZJDHYVvHTA0"
    sheet = client.open_by_key(sheet_id).sheet1
    # 获取整张表的数据
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
    # 获取整张表的数据
    all_data = sheet.get_all_values()

    last_non_empty_row = None
    for row in all_data[1:]:  # 跳過標題行
        if row[8]:  # I列是第9列（索引為8）
            last_non_empty_row = row[8]
    return last_non_empty_row

#HW4 Line bot

from google.colab import drive
drive.mount('/content/drive', force_remount=True)

'''
!mkdir -p /drive
#umount /drive
!mount --bind /content/drive/My\ Drive /drive
!mkdir -p /drive/ngrok-ssh
!mkdir -p ~/.ssh

!pip install fastapi
!pip install line-bot-sdk
!pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib
!pip install pyngrok

!/ngrok authtoken 2h4nWQBR7244VzWwaAK2GZQJZlC_6z2EMA5otEtrAyXhPRhn2
'''

import getpass

from pyngrok import ngrok, conf

print("Enter your authtoken, which can be copied from https://dashboard.ngrok.com/auth")
conf.get_default().auth_token = getpass.getpass()

# Open a TCP ngrok tunnel to the SSH server
connection_string = ngrok.connect("22", "tcp").public_url

ssh_url, port = connection_string.strip("tcp://").split(":")
print(f" * ngrok tunnel available, access with `ssh root@{ssh_url} -p{port}`")

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