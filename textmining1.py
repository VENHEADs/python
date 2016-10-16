import scipy.spatial
import numpy as np
import re

data = open("senten.txt", "r") # открываем
sent = data.readlines()   #считываем все строки
print(sent)

i = 0
for sentence in sent: # берем первую строку
    sentence = re.split('[^a-z]', sentence.lower()) # разбиваем на слова
    sent[i] = filter(None, sentence) # убираем пустые слова из первой строки и сохраняем
    i += 1
    
word_index = dict() # создаем пустой словарь
i = 0 
for sentence in sent: # берем первую строку
    for word in sentence: # берем первое слово в первой строке
        if word not in word_index: # если слова нету в словаре то  
            word_index[word] = i # добавляем его по индексом 0 в словарь
            i += 1 # следующий индекс будет на единицу больше
            

m = np.zeros((len(sent), len(word_index))) # создаем массив размерность строки*слова
m.shape

for sent_i in xrange(0, len(sent)): # смотрим первую строку
    for word in sent[sent_i]: # смотрим слова в первой строке
        word_i = word_index[word] # находим индекс этого слова
        m[sent_i][word_i] += 1 # вносим это слово в массив, изначально там нули но как только в стркое нахоидм слово с этим индексов 
    
distances = list() # смотрим теперь каждую строку
for i in xrange(0, len(sent)):
    distance = scipy.spatial.distance.cosine(m[0,:],m[i,:]) # считаем косинусную дистанцию
    distances.append((i,distance))
    
sort = sorted(distances,key=lambda tup: tup[1]) # сортируем
print(sort[1],sort[2])
