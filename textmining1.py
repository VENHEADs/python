import scipy.spatial
import numpy as np
import re

data = open("senten.txt", "r")
sent = data.readlines()   
print(sent)

i = 0
for sentence in sent:
    sentence = re.split('[^a-z]', sentence.lower())
    sent[i] = filter(None, sentence)
    i += 1
    
word_index = dict()
i = 0
for sentence in sent:
    for word in sentence:
        if word not in word_index:
            word_index[word] = i
            i += 1
            

m = np.zeros((len(sent), len(word_index)))
m.shape

for sent_i in xrange(0, len(sent)):
    for word in sent[sent_i]:
        word_i = word_index[word]
        m[sent_i][word_i] += 1
    
distances = list()
for i in xrange(0, len(sent)):
    distance = scipy.spatial.distance.cosine(m[0,:],m[i,:])
    distances.append((i,distance))
    
sort = sorted(distances,key=lambda tup: tup[1])
print(sort[1],sort[2])
