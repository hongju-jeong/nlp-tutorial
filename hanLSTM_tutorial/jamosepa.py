from jamo import h2j, j2hcj
from unicode import join_jamos

fr = open("CORPUS11.txt", 'r')
fw = open("./input/input.txt", "w", encoding="utf-8")


lines = fr.readlines()
i = 0
for line in lines:
    sentence = j2hcj(h2j(line))
    fw.write(sentence)

fr.close()
fw.close()