from re import T
import pandas as pd
from scipy import stats
import numpy as np
import time
import os
import torch
import json
import pprint
from transformers import T5Tokenizer
import spacy
spacy.prefer_gpu()
nlp = spacy.load("en_core_web_trf")

with open('recent_news.json') as f:
    dataset = json.load(f)

tokenizer = T5Tokenizer.from_pretrained('google/t5-large-ssm')

sentinels=[]
for i in range(200):
    sentinels.append(f'<extra_id_{i}>')

def ssm(index, text):
    if text=='':
        return None
    input = ""
    target = ""
    sentinel_cnt=0
    previous_end=0
    doc = nlp(text)
    for ent in doc.ents:
        start_index = ent.start_char
        end_index = ent.end_char
        word = ent.text
        input = input + text[previous_end:start_index] + sentinels[sentinel_cnt]
        target = target + sentinels[sentinel_cnt]+" " + word +" "
        previous_end = end_index
        sentinel_cnt+=1
    input = input + text[previous_end:]
    target = target + sentinels[sentinel_cnt]
    print(index)
    return index, text, input, target

length_limit = 250 #The limit of words per input
article_index = 0
recent_news = []

for entry in dataset:
    #print(article_index)
    article_index+=1
    if article_index==20000:
        break
    text = entry['text']
    title = entry['title']
    date = entry['date']
    tokens = tokenizer.encode(text)
    token_len = len(tokens)
    #For debug setting
    #if token_len>200:
    #    continue
    #For getting rid of articles before 05.2020
    date = date.split(' ')[0]
    date_s = date.split('-')
    if date_s[0]=='2020' and int(date_s[1])<5:
        continue
    text = text.replace('\n', '')
    if len(text.split()) > length_limit:
        word_list = text.split()
        seg1 = word_list[:length_limit]
        try:
            segment1, seg2_a = (' '.join(seg1)).rsplit('.',1)
        except ValueError as e:
            seg2_a = ''
        segment2 = seg2_a + (' '.join(word_list[length_limit:]))
        output = ssm(article_index, segment1)
        if output: recent_news.append(output)

        while(len(segment2.split()) > length_limit):
            word_list = segment2.split()
            seg1_ = word_list[:length_limit]
            if '.' in ' '.join(seg1_):
                segment1_, seg2_a_ = (' '.join(seg1_)).rsplit('.',1)
                segment2 = seg2_a_ + (' '.join(word_list[length_limit:]))
            else:
                segment1_ = ' '.join(seg1_)
                segment2 = (' '.join(word_list[length_limit:]))
            output = ssm(article_index, segment1_)
            if output:  recent_news.append(output)
    else:
        output=ssm(article_index, text)
        if output: 
            recent_news.append(output) 
    

pd.DataFrame(recent_news, columns=['index','original', 'input', 'output']).to_csv('recent_news_debug.csv')