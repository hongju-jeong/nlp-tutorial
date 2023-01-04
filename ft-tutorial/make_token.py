import re
import json
from gensim.models import FastText
from konlpy.tag import Komoran

def make_token(input_file, output_file):
    komoran = Komoran()
    token_txt_file = open(output_file, "w", encoding="utf-8")
    list = []

    with open(input_file, 'r', encoding="utf-8") as f:
        text = f.readlines()
        num = 0

        for i in range(0, len(text)):
            sentence = text[i].strip()
            morphs = komoran.morphs(sentence)
            #print(morphs)
            list.append(morphs)

        #print(num)

        my_json_string = json.dumps(list, ensure_ascii=False)
        token_txt_file.write(my_json_string)

if __name__ == "__main__":
    make_token("corpus.txt", "corpus_token.txt")