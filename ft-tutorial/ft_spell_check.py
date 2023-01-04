from gensim.models import FastText
import enchant
import nltk

model_fasttext = FastText.load('fasttext.model')

fasttext_min_similarity = 0.6
enchant_kr = enchant.Dict('ko_KR')

def include_spell_mistake(word, similar_word, score):
    edit_distance_threshold = 1 if len(word) <= 4 else 2
    score_1 = score > fasttext_min_similarity
    score_2 = len(similar_word) > 3
    score_3 = not enchant_kr.check(similar_word)
    score_4 = word[0] == similar_word[0]
    score_5 = nltk.edit_distance(word, similar_word) <= edit_distance_threshold
    score = score_1 + score_2 + score_3 + score_4 + score_5
    if score > 3:
        return True
    else:
        return False

def spell_check(word):
    w2m = []
    most_similar = model_fasttext.wv.most_similar(word, topn=50)
    for similar_word, score in most_similar:
        if include_spell_mistake(word, similar_word, score):
            w2m.append(similar_word)
    output = {'prediction' : w2m[:3]}
    return output

print(spell_check('스울'))