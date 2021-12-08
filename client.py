import requests
import time
import json

def send_post_message(msg,url):
    post_msg = {"msg":str(msg),"post_time":str(time.time())}  ## 定義發送信息
    headers = {"Content-Type":"application/json"}             ## 定義請求頭
    _post_data = json.dumps(post_msg,sort_keys=True,separators=(',', ':'))  ## 將信息打包為json格式
    req = requests.post(url, data=_post_data, headers=headers)              ## 使用requests，以POST形式發送信息
    if req.status_code == requests.codes.ok:  ## 如果請求正常並收到回復
        print('Sending ok')
        result = req.json()  ##讀取回復
        print(result)

if __name__ == '__main__':
    while(True):
        input_str = input("輸入訊息:")
        # send_post_message(input_str,'http://127.0.0.1:8080')
        send_post_message(input_str,'http://140.112.187.97:9101/')

