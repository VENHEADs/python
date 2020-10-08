#!/usr/bin/env python
# coding: utf-8


import numpy as np



class Min_Heap():
    
    def  __init__(self):
        self.array  = []
        self.value_to_ind = {}
        self.ind_to_value = {}
        self.closest_node_path = {}
        self.indexes = []
        
    def __len__(self):
        return len(self.array)
    @property
    def min_el(self):
        return self.array[0]
        
    def _get_parent(self,index):
        if index > 0:
            return (index+1)//2 -1
        else:
            return 0
        
    def _get_child(self,index):
        c_index = (index+1)*2
        return c_index-1,c_index
        
        
    def _swap(self,p_ind,c_ind):
        p_value,c_value = self.ind_to_value[p_ind],self.ind_to_value[c_ind]
        self.array[p_ind],self.array[c_ind] = self.array[c_ind],self.array[p_ind]
        self.ind_to_value[p_ind],self.ind_to_value[c_ind] = c_value,p_value
        self.value_to_ind[p_value],self.value_to_ind[c_value] = c_ind,p_ind
        c_ind = p_ind
        p_ind = self._get_parent(c_ind)
        return c_ind,p_ind
        
    def _insert(self,key):
        self.array += [key[0]]
        c_ind = len(self.array)-1
        c_key = key[0]
        self.value_to_ind[key[1]] = c_ind
        self.ind_to_value[c_ind] = key[1]
        p_ind = self._get_parent(c_ind)
        p_key = self.array[p_ind]
        while self.array[c_ind] < self.array[p_ind]:
              c_ind,p_ind = self._swap(p_ind,c_ind)
        self.closest_node_path[key[1]] = key[2]
        
            
    def extract_min(self):
        return self._delete()
    
    def _delete(self,index=0):
        min_ind = index
        min_value = self.ind_to_value[min_ind]
        min_key = self.array[min_ind]
    
        last_ind = len(self.array)-1
        last_value = self.ind_to_value[last_ind]
        last_key = self.array[last_ind]
        
        
        self.array[min_ind] = last_key
        self.ind_to_value[min_ind] = last_value
        self.value_to_ind[last_value] = min_ind
        
        del self.value_to_ind[min_value],self.ind_to_value[last_ind],self.array[last_ind]
        
        if min_ind == len(self.array):
            return min_key,min_value
        
        p_ind = self._get_parent(min_ind)
        l_c,r_c = self._get_child(min_ind)
        while self.array and self.array[min_ind] < self.array[p_ind]:
              min_ind,p_ind = self._swap(p_ind,min_ind)
            
                

        
        while l_c < len(self.array)  and r_c < len(self.array) :
            c_keys = [self.array[l_c],self.array[r_c]]
            sm_c_keys = np.min(c_keys)
            sm_c_ind = np.argmin(c_keys)
            sm_c_ind = [l_c,r_c][sm_c_ind]
            if last_key <= sm_c_keys:
                return min_key,min_value
            min_ind,c_ind = self._swap(sm_c_ind,min_ind)
            l_c,r_c = self._get_child(min_ind)
            
        if l_c < len(self.array) and last_key > self.array[l_c]:
            min_ind,c_ind = self._swap(l_c,min_ind)
            
 
        return min_key,min_value

    def insert(self,node):
        key,value,prev_value = node
        if value not in self.value_to_ind:
            self._insert(node)
        else:
            index = self.value_to_ind[value]
            old_key = self.array[index]
            if old_key > key:
                _,__ = self._delete(index)
                self._insert(node)
                





class Max_Heap():
    
    def  __init__(self):
        self.array  = []
        self.value_to_ind = {}
        self.ind_to_value = {}
        self.closest_node_path = {}
        self.indexes = []
        
    @property
    def max_el(self):
        return self.array[0]

    def __len__(self):
        return len(self.array)
            
    def _get_parent(self,index):
        if index > 0:
            return (index+1)//2 -1
        else:
            return 0
        
    def _get_child(self,index):
        c_index = (index+1)*2
        return c_index-1,c_index
        
        
    def _swap(self,p_ind,c_ind):
        p_value,c_value = self.ind_to_value[p_ind],self.ind_to_value[c_ind]
        self.array[p_ind],self.array[c_ind] = self.array[c_ind],self.array[p_ind]
        self.ind_to_value[p_ind],self.ind_to_value[c_ind] = c_value,p_value
        self.value_to_ind[p_value],self.value_to_ind[c_value] = c_ind,p_ind
        c_ind = p_ind
        p_ind = self._get_parent(c_ind)
        return c_ind,p_ind
        
    def _insert(self,key):
        self.array += [key[0]]
        c_ind = len(self.array)-1
        c_key = key[0]
        self.value_to_ind[key[1]] = c_ind
        self.ind_to_value[c_ind] = key[1]
        p_ind = self._get_parent(c_ind)
        p_key = self.array[p_ind]
        while self.array[c_ind] > self.array[p_ind]:
              c_ind,p_ind = self._swap(p_ind,c_ind)
        self.closest_node_path[key[1]] = key[2]
        
            
    def extract_max(self):
        return self._delete()
    
    def _delete(self,index=0):
        min_ind = index
        min_value = self.ind_to_value[min_ind]
        min_key = self.array[min_ind]
    
        last_ind = len(self.array)-1
        last_value = self.ind_to_value[last_ind]
        last_key = self.array[last_ind]
        
        
        self.array[min_ind] = last_key
        self.ind_to_value[min_ind] = last_value
        self.value_to_ind[last_value] = min_ind
        
        del self.value_to_ind[min_value],self.ind_to_value[last_ind],self.array[last_ind]
        
        if min_ind == len(self.array):
            return min_key,min_value
        
        p_ind = self._get_parent(min_ind)
        l_c,r_c = self._get_child(min_ind)
        while self.array and self.array[min_ind] > self.array[p_ind]:
              min_ind,p_ind = self._swap(p_ind,min_ind)
            
                

        
        while l_c < len(self.array)  and r_c < len(self.array) :
            c_keys = [self.array[l_c],self.array[r_c]]
            sm_c_keys = np.max(c_keys)
            sm_c_ind = np.argmax(c_keys)
            sm_c_ind = [l_c,r_c][sm_c_ind]
            if last_key >= sm_c_keys:
                return min_key,min_value
            min_ind,c_ind = self._swap(sm_c_ind,min_ind)
            l_c,r_c = self._get_child(min_ind)
            
        if l_c < len(self.array) and last_key < self.array[l_c]:
            min_ind,c_ind = self._swap(l_c,min_ind)
            
 
        return min_key,min_value

    def insert(self,node):
        key,value,prev_value = node
        if value not in self.value_to_ind:
            self._insert(node)
        else:
            index = self.value_to_ind[value]
            old_key = self.array[index]
            if old_key > key:
                _,__ = self._delete(index)
                self._insert(node)
                




heap_max = Max_Heap()
heap_min = Min_Heap()





iters = 0
while True:
    array = []
    heap_max = Max_Heap()
    heap_min = Min_Heap()
    for i in np.random.normal(10000,10000,np.random.randint(1,1000)):
        array.append(i)


        if len(heap_min) == 0:
            heap_min.insert((i,i,i))
            continue

        if len(heap_max) == 0:
            heap_max.insert((i,i,i))
            if heap_max.max_el > heap_min.min_el:
                max_el = heap_max.extract_max()[0]
                min_el = heap_min.extract_min()[0]
                heap_min.insert((max_el,max_el,max_el))
                heap_max.insert((min_el,min_el,min_el))
            continue

        if i > heap_max.max_el:
            heap_min.insert((i,i,i))
        else:
            heap_max.insert((i,i,i))

        if len(heap_max)-2 == len(heap_min):
            max_heap_max = heap_max.extract_max()[0]
            heap_min.insert((max_heap_max,max_heap_max,max_heap_max))
        elif len(heap_max)+2 == len(heap_min):
            min_heap_min = heap_min.extract_min()[0]
            heap_max.insert((min_heap_min,min_heap_min,min_heap_min))

    if len(heap_min) == len(heap_max):   
        assert np.median(array) == (heap_min.min_el + heap_max.max_el)/2
    elif len(heap_min) > len(heap_max):
        assert np.median(array) == heap_min.min_el 
    else:
        assert np.median(array) == heap_max.max_el 
    iters += 1
    if iters%1000 == 0:
        print(iters)



