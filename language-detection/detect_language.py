from io import StringIO
from langdetect import detect

import pandas as pd

# result = detect("RT @BrentRivera: following people who like my new YouTube video and share it on here :) I really think you guys are going ")
# print(result)


f = open("./data.txt", encoding="utf8")
g = open("myfile.txt", "w")
for x in f:
    word_arr = x.split(',')
    lang = detect(word_arr[0].lower())
    res = "{}\n".format(lang in word_arr[1])
    g.write(res)
