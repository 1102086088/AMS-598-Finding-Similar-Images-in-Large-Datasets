#!/usr/bin/env python
# coding: utf-8

# In[131]:


from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import random as rd


# In[107]:


def read_img(path):
    img = Image.open(path)
    return np.round(np.array(img.resize((15, 20))) / 255)

def convert_img(img):
    for i in range(len(img)):
        for j in range(len(img[i])):
            if sum(img[i][j]) == 1 or sum(img[i][j]) == 2:
                img[i][j] = np.array([0., 0., 0.])

def to_2D_matrix(img):
    matrix = []
    for i in img:
        array = []
        for j in i:
            array.append(j[0])
        matrix.append(array)
    return np.asarray(matrix)
          
def to_matrix(path):
    img = read_img(path)
    convert_img(img)
    return to_2D_matrix(img)


# In[108]:


def shingling(img):
    k = 5
    s = []
    for i in range(img.shape[0] - k):
        for j in range(img.shape[1] - k):
            img_pc = img[i:i+k, j:j+k].flatten()
            s.append(str(img_pc))
    return s


# In[149]:


images = []
all_shingles = []
for i in os.listdir('test'):
    name = i
    matrix = to_matrix('/Users/zhezhou/Downloads/test/'+i)
    shingles = shingling(matrix)
    images.append((name, shingles))
    all_shingles += shingles
    all_shingles = list(set(all_shingles))
    
matrix = []
for i in images:
    name = i[0]
    shingles = i[1]
    col = []
    for j in all_shingles:
        if j in shingles:
            col.append(1)
        else:
            col.append(0)
    matrix.append(col)
matrix = np.transpose(np.asarray(matrix))


# In[169]:


def minHash(matrix):
    row = matrix.shape[0]
    col = matrix.shape[1]
    num = 100
    perm = list(range(row))
    sig = np.zeros([num, col])

    for n in range(num):
        rd.shuffle(perm)
        for i in range(col):
            v = 1
            for j in perm:
                if matrix[j, i] == 1:
                    sig[n, i] = v
                    break
                else:
                    v += 1

    return sig


# In[171]:


sign = minHash(matrix)


# In[174]:


sign


# In[136]:


INT_MAX = 99999
np.zeros([2, 1]) + INT_MAX


# In[100]:


img = Image.open('/Users/zhezhou/Downloads/test/' + '1991.jpg')
img = np.round(np.array(img.resize((60, 80))) / 255)
plt.imshow(img)


# In[ ]:


# 缺点
    round会带来较大的误差，手臂会被认为白色
    像素点少，计算快，正确率高
    应该有跟便捷的算法


# In[165]:


a = np.matrix('1 2 3; 4 5 6')
a[1]


# In[152]:


for i in range(600000000):
    if i / 2 == 0:
        j = 1
    


# In[166]:


len(matrix[1])


# In[ ]:




