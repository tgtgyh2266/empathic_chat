import logging
import time
import web
import json
import numpy as np
import random
import torch
import pandas as pd
from sentence_transformers import SentenceTransformer
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration
import googletrans

#########################################################################################################
#   total gpu memory cost : almost 7G
#
#   current method to buid the system:
#   Detect the emotion of the 2 most recent sentences that user said.
#   But when it comes to the chatting models, use the whole conversation context as input. (up to 512 words)
#   If the emotion is negative -> feed the context into the retrieval model.
#   If positive or ambiguous -> feed the context into Blender bot.
#########################################################################################################

SAVE_PATH_RET = 'retrieval_model_mpnet'   # path to retrieval model
SAVE_PATH_EMO = 'emo_model_roberta'   # path to emotion_classification model
RET_FILE_PATH = 'pruned_train.csv' # path to retrieval dataset

label_dict = {0:'positive', 1:'negative', 2:'ambiguous'}
random_seed = 87
# Set random states for reproducibility
random.seed(random_seed)
np.random.seed(random_seed)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_ret = SentenceTransformer(SAVE_PATH_RET, device=device)
tokenizer_emo = RobertaTokenizer.from_pretrained('roberta-base')
model_emo = RobertaForSequenceClassification.from_pretrained(SAVE_PATH_EMO)
model_chitchat = BlenderbotForConditionalGeneration.from_pretrained('facebook/blenderbot-400M-distill')
tokenizer_chitchat = BlenderbotTokenizer.from_pretrained('facebook/blenderbot-400M-distill')
model_chitchat = model_chitchat.to(device)
model_emo = model_emo.to(device)
model_emo.eval()
model_ret.eval()

train_df = pd.read_csv(RET_FILE_PATH)
all_responses = []
for x in train_df['response']:
    all_responses.append(x)
all_responses_emb = model_ret.encode(all_responses)
input_context = []
total_length = 0
translator = googletrans.Translator()

class web_server_template:  ##宣告一個class,在下文的web.application實例化時，會根據定義將對應的url連接到這個class
    def __init__(self):  ##初始化類別
        print('initial in {}'.format(time.time()))

    def POST(self):  ##當server收到一個指向這個class URL的POST請求，會觸發class中命名為POST的函數，GET請求同理
        recive = json.loads(str(web.data(),encoding='utf-8'))  ##使用json.loads將json格式讀取為字典
        print('[Message] Post message recive:{}'.format(recive))
        result = True
        received_data = recive["msg"]

        global total_length
        global input_context

        if received_data == '清除':
            total_length=0
            input_context.clear()
            
            passing_information = 'history cleared!'
            return_json = {'results':result,'return_message':passing_information}
            return_data = json.dumps(return_json,sort_keys=True,separators=(',',':'),ensure_ascii=False) ##打包回傳信息為json
            return return_data  ##回傳

        text = received_data
        text = translator.translate(text, dest='en').text

        input_context.append(text)
        print(input_context)
        total_length += len(text)
        while total_length>511:
            total_length -= len(input_context[0])
            input_context.pop(0)
        context_emb = '. '.join(input_context)
        #   print(context_emb)

        inputs_emo = input_context
        if len(inputs_emo)>2:
            inputs_emo = inputs_emo[-2:]
        print('inputs_emo: ',inputs_emo)
        inputs_emo = '. '.join(inputs_emo)
        inputs_emo = tokenizer_emo(inputs_emo, return_tensors="pt").to(device)    #emotion detection
        outputs_emo = model_emo(**inputs_emo)
        emotion = torch.argmax(outputs_emo.logits).item()
        emotion = label_dict[emotion]
        print(emotion)

        if emotion == 'negative':
            context_emb = model_ret.encode(context_emb)
            max_idx = -1
            max_dot = 0
            for i in range(len(all_responses)):     #find out the response that has the highest dot product with input_context
                temp = np.dot(np.array(context_emb), np.array(all_responses_emb[i]))
                if temp > max_dot:
                    max_dot = temp
                    max_idx = i
            #   print(max_idx)
            print('dot product: ', max_dot)
            res = all_responses[max_idx]
        else :
            inputs_chitchat = tokenizer_chitchat([context_emb], return_tensors='pt').to(device) 
            reply_ids = model_chitchat.generate(**inputs_chitchat)
            res = tokenizer_chitchat.decode(reply_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        print(res)
        # input_context.append(res)
        # total_length += len(res)
        res = translator.translate(res, dest='zh-tw').text
        print(res)
        passing_information = res
        return_json = {'results':result,'return_message':passing_information}
        return_data = json.dumps(return_json,sort_keys=True,separators=(',',':'),ensure_ascii=False) ##打包回傳信息為json

        return return_data  ##回傳

    def GET(self):
        return 'Hello World!'


if __name__ == '__main__':
    logging.basicConfig()
    URL_main = ("/","web_server_template")  ##宣告URL與class的連接
    app = web.application(URL_main,globals())  ##初始化web application，默認地址為127.0.0.1:8080，locals()代表web.py會在當前文件內尋找url對應的class
    app.run()  ##運行web application

    #python server.py 9111
