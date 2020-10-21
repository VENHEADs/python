#!/usr/bin/env python
# coding: utf-8

# In[126]:


import numpy as np


# In[127]:


class Node():
    
    def __init__(self,parent,value):
        self.value = value
        self.parent = parent
        self.l_child = None
        self.r_child = None


# In[128]:


list_pr = []
with open('input_random_48_10000.txt') as f:
    for line in f:
        list_pr.append(int(line.strip().split('\t')[0]))
list_pr = list_pr[1:]
# list_pr = np.unique(list_pr)


# In[129]:


values = np.sort(list_pr)
# values = values* 0.00000000001
# values[0] = 99993321**2
q_1 = [Node(None,value) for value in values]
q_2 = []

depth = 0
while q_1 or len(q_2) > 1:

    if not q_1:
        node_1 = q_2.pop(0)
    elif not q_2:
        node_1 = q_1.pop(0)
    else:
        if q_1[0].value < q_2[0].value:
            node_1 = q_1.pop(0)
        else:
            node_1 = q_2.pop(0)
            
    if not q_1:
        node_2 = q_2.pop(0)
    elif not q_2:
        node_2 = q_1.pop(0)
    else:
        if q_1[0].value < q_2[0].value:
            node_2 = q_1.pop(0)
        else:
            node_2 = q_2.pop(0)
    merge = node_1.value + node_2.value
    new_node = Node(None,merge)
    new_node.r_child = node_2
    new_node.l_child = node_1
    q_2.append(new_node)


# In[130]:


results = {}
def traverse(node,path):
    global results
    if node.l_child != None:
        traverse(node.l_child,path + '1')
    if not node.r_child and not node.l_child:
        results[node.value] = path
    if node.r_child != None:
        traverse(node.r_child,path + '0') 


# In[131]:


traverse(q_2[0],'')


# In[ ]:




