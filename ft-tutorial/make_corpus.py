import re
import json
from gensim.models import FastText
from konlpy.tag import Komoran

def make_corpus(input_file, output_file):
    txt_file = open(output_file, "w", encoding="utf-8")

    with open(input_file, 'r', encoding="utf-8") as f:
        text = f.readlines()
        num = 0

        for i in range(0, len(text)):
            sentence_list = text[i].strip() #양쪽공백제거
            sentence = sentence_list.split('.')

            for j in range(0, len(sentence)):
                if len(sentence[j].strip()) > 3:
                    last_sentence = sentence[j].strip()
                    #print(last_sentence)
                    txt_file.write(last_sentence + "\n")
                    num += 1


if __name__ == "__main__":
    make_corpus("Han.txt", "corpus.txt")

