import re
import json
from gensim.models import FastText
from konlpy.tag import Komoran

with open('corpus_token.txt', 'r', encoding="utf-8") as f:
    text = f.readlines()
    data = json.loads(text[0])

embedding = FastText(data, vector_size=100, window=7, negative=3, min_count=5)
embedding.save('fasttext.model')  # weight vector 모델을 저장합니다.

model = FastText.load('fasttext.model')
print(model.wv['서울'])
print(model.wv.most_similar('서울'))