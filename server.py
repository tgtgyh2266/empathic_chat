import logging
import time
import web
import json
import numpy as np
import random
import torch
import pandas as pd
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration
from transformers import AutoModelForCausalLM, AutoTokenizer
# import googletrans
from deep_translator import GoogleTranslator

#########################################################################################################
#   total gpu memory cost : almost 4G
#
#   current method to buid the system:
#   Detect the emotion of the 2 most recent sentences that user said.
#   But when it comes to the chatting models, use the whole conversation context as input. (up to 4 turns)
#   If the emotion is negative -> feed the context into the generative model.
#   If positive or ambiguous -> feed the context into Blender bot and clear the history.
#
#   Special situation:
#   If greeting, thank or goodbye words are detected, clear the history and reply with template.
#
#   Keywords template is integrated in this version. Each keyword corresponds to a [empathy, encouragement, advice] response list.
#   If keywords are detected, we first provide empathy response.
#   After providing empathy response, if no other keywords are detected in the next input, we provide encouragement response. Same for advice response.
#########################################################################################################

SAVE_PATH_GEN = 'model_neg_only/checkpoint-7623'   # path to generative model
SAVE_PATH_EMO = '../retrieval_emo_sup/go_emo/roberta_5epochs/checkpoint-16281'   # path to emotion_classification model

greeting_input = ['你好', '嗨', '哈囉', 'hi', 'Hi', 'HI']
goodbye_input = ['再見', '掰掰', 'Bye', 'bye']
thank_input = ['謝謝', '感謝']

with open('template.json', newline='', encoding="utf-8") as f:
    template = json.load(f)

key_words = []
template_responses = []
prev_keyword_idx = -1
used_response_num = [0]*len(key_words)
for key in template:
    key_words.append(key)
    temp_list = []
    if len(template[key]['empathy'])>0:
        temp_list.append(template[key]['empathy'][0])
    if len(template[key]['encouragement'])>0:
        temp_list.append(template[key]['encouragement'][0])
    if len(template[key]['advice'])>0:
        temp_list.append(template[key]['advice'][0])
    template_responses.append(temp_list)

label_dict = {0:'positive', 1:'negative', 2:'ambiguous'}
random_seed = 87
# Set random states for reproducibility
random.seed(random_seed)
np.random.seed(random_seed)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

tokenizer_gen = AutoTokenizer.from_pretrained('microsoft/DialoGPT-small')
model_gen = AutoModelForCausalLM.from_pretrained(SAVE_PATH_GEN)
model_gen.resize_token_embeddings(len(tokenizer_gen))
model_gen = model_gen.to(device)
tokenizer_emo = RobertaTokenizer.from_pretrained('roberta-base')
model_emo = RobertaForSequenceClassification.from_pretrained(SAVE_PATH_EMO)
model_chitchat = BlenderbotForConditionalGeneration.from_pretrained('facebook/blenderbot-400M-distill')
tokenizer_chitchat = BlenderbotTokenizer.from_pretrained('facebook/blenderbot-400M-distill')
model_chitchat = model_chitchat.to(device)
model_emo = model_emo.to(device)
model_emo.eval()
model_gen.eval()

input_context = []
total_length = 0
# translator = googletrans.Translator()

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
        global prev_keyword_idx
        global used_response_num

        has_keyword = False

        if received_data == '清除':
            total_length=0
            input_context.clear()
            prev_keyword_idx = -1
            used_response_num = [0]*len(key_words)
            passing_information = 'history cleared!'
            has_keyword = True

        if not has_keyword:
            for x in greeting_input:
                if x in received_data:
                    passing_information = '嗨，今天想和我聊些什麼呢?'
                    total_length=0
                    input_context.clear()
                    prev_keyword_idx = -1
                    used_response_num = [0]*len(key_words)
                    has_keyword = True
                    break

        if not has_keyword:
            for x in goodbye_input:
                if x in received_data:
                    passing_information = '再見，隨時歡迎你再來找我聊天'
                    total_length=0
                    input_context.clear()
                    prev_keyword_idx = -1
                    used_response_num = [0]*len(key_words)
                    has_keyword = True 
                    break

        if not has_keyword:
            for x in thank_input:
                if x in received_data:
                    passing_information = '不用客氣，很高興能幫上你的忙'
                    total_length=0
                    input_context.clear()
                    prev_keyword_idx = -1
                    used_response_num = [0]*len(key_words)
                    has_keyword = True
                    break

        if not has_keyword:
            for i in range(len(key_words)):
                if key_words[i] in received_data and used_response_num[i] < len(template_responses[i]):   # keyword detected and response available
                    passing_information = template_responses[i][used_response_num[i]]
                    used_response_num[i] += 1
                    prev_keyword_idx = i
                    text = received_data
                    # text = translator.translate(text, dest='en').text
                    text = GoogleTranslator(source='auto', target='en').translate(text)
                    input_context.append(text)
                    total_length += len(text)
                    text = passing_information
                    # text = translator.translate(text, dest='en').text
                    text = GoogleTranslator(source='auto', target='en').translate(text)
                    total_length += len(text)
                    input_context.append(text)
                    has_keyword = True
                    break

        if not has_keyword:
            if prev_keyword_idx != -1 and used_response_num[prev_keyword_idx]<len(template_responses[prev_keyword_idx]):  # no new keywords detected after replying empathy response
                passing_information = template_responses[prev_keyword_idx][used_response_num[prev_keyword_idx]]
                used_response_num[prev_keyword_idx] += 1
                text = received_data
                # text = translator.translate(text, dest='en').text
                text = GoogleTranslator(source='auto', target='en').translate(text)
                input_context.append(text)
                total_length += len(text)
                text = passing_information
                # text = translator.translate(text, dest='en').text
                text = GoogleTranslator(source='auto', target='en').translate(text)
                total_length += len(text)
                input_context.append(text)
                has_keyword = True
        
        if not has_keyword:
            prev_keyword_idx = -1
            text = received_data
            # text = translator.translate(text, dest='en').text
            if not text.isdigit():
                text = GoogleTranslator(source='auto', target='en').translate(text)

            input_context.append(text)
            total_length += len(text)
            while len(input_context)>4:
                total_length -= len(input_context[0])
                input_context.pop(0)

            inputs_emo = input_context
            if len(inputs_emo)>2:
                inputs_emo = inputs_emo[-2:]
            print('input for emo classify: ',inputs_emo)
            inputs_emo = '. '.join(inputs_emo)
            inputs_emo = tokenizer_emo(inputs_emo, return_tensors="pt").to(device)    #emotion detection
            outputs_emo = model_emo(**inputs_emo)
            emotion = torch.argmax(outputs_emo.logits).item()
            emotion = label_dict[emotion]
            print('emotion: ', emotion)

            if emotion == 'negative':
                print('input context: ', input_context)         #use generative model to generate response
                context_emb = tokenizer_gen.eos_token.join(input_context)
                context_emb += tokenizer_gen.eos_token
                context_emb = tokenizer_gen.encode(context_emb)
                context_emb = torch.tensor(context_emb)
                context_emb = torch.unsqueeze(context_emb, 0)
                context_emb = context_emb.to(device)
                res_emb = model_gen.generate(context_emb, max_length=1000, pad_token_id=tokenizer_gen.eos_token_id)
                res = tokenizer_gen.decode(res_emb[:, context_emb.shape[-1]:][0], skip_special_tokens=True)
    
            else :
                print('input context: ', text)      #use chitchat model to generate response
                total_length=0
                input_context.clear()
                context_emb = text
                inputs_chitchat = tokenizer_chitchat([context_emb], return_tensors='pt').to(device) 
                reply_ids = model_chitchat.generate(**inputs_chitchat)
                res = tokenizer_chitchat.decode(reply_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
            print(res)
            input_context.append(res)
            total_length += len(res)
            # res = translator.translate(res, dest='zh-tw').text
            if not res.isdigit():
                res = GoogleTranslator(source='auto', target='zh-tw').translate(res)
            print(res)
            passing_information = res

        return_json = {'results':result,'return_message':passing_information}
        return_data = json.dumps(return_json,sort_keys=True,separators=(',',':'),ensure_ascii=False) ##打包回傳信息為json

        return return_data  ##回傳

    def GET(self):
        return 'Hello World!'

URL_main = ("/","web_server_template")  ##宣告URL與class的連接

if __name__ == '__main__':
    logging.basicConfig()
    app = web.application(URL_main,globals(), autoreload = False)  ##初始化web application，默認地址為127.0.0.1:8080，locals()代表web.py會在當前文件內尋找url對應的class
    app.run()  ##運行web application

    #python server.py 9111
