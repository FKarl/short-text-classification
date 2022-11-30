#!/usr/bin/env python
# coding: utf-8

# 

# # Creation of the NICE dataset

# In[1]:


# define paths
PATH_TO_LABELS = 'source/labels.xml'
PATH_TO_TEXTS = 'source/texts.xml'

PATH_TO_BINARY_RESULTS = 'NICE_binary/NICE'
PATH_TO_RESULTS = 'NICE/NICE'


# In[2]:


# define parameters
RANDOM_STATE = 42
TRAIN_SPLIT = 0.7


# ### Load the labels and texts

# In[3]:


# read labels.xml file
import xml.etree.ElementTree as ET

tree = ET.parse(PATH_TO_LABELS)
root = tree.getroot()

labels = dict()
for cl in root:
    gs = cl.attrib['isGoodOrService']
    cl_number = cl.attrib['classNumber']
    for elem in cl:
        labels[elem.attrib['id']] = gs, cl_number


# In[4]:


# read texts.xml file
import xml.etree.ElementTree as ET

tree = ET.parse(PATH_TO_TEXTS)
root = tree.getroot()

texts = dict()
# skip ClassesTexts and iterate over GoodsAndServicesTexts
for text in root[1]:
    # indication > labels
    texts[text.attrib['idRef']] = text[0][0].text


# ### Write the results

# In[5]:


# preprocessing
import re
import string
import unicodedata

def preprocess(str):
    # lowercase
    str = str.lower()
    # remove text inside [] brackets
    str = re.sub(r'\[.*?\]', '', str)
    # remove punctuation
    str = str.translate(str.maketrans('', '', string.punctuation))
    # remove accents
    str = ''.join(c for c in unicodedata.normalize('NFD', str) if unicodedata.category(c) != 'Mn')
    return str


# In[6]:


# create splits with fixed seed
import random
random.seed(RANDOM_STATE)

length = len(texts)
split_length = int(length * TRAIN_SPLIT)
train_indices = random.sample(range(length), split_length)
test_indices = [i for i in range(length) if i not in train_indices]

print('Train:', len(train_indices), 'Test:', len(test_indices))

keys = list(texts.keys())
test_keys = [keys[i] for i in test_indices]
train_keys = [keys[i] for i in train_indices]


# In[7]:


# create binary dataset
assert labels.keys() == texts.keys()

# train split
with open(PATH_TO_BINARY_RESULTS + '_train.txt' , 'w', encoding="utf-8") as f:
    # write all except the last line
    for key in train_keys[:-1]:
        f.write(labels[key][0] + '\t' + preprocess(texts[key]) + '\n')
    # write the last line
    last_key = train_keys[-1]
    f.write(labels[last_key][0] + '\t' + preprocess(texts[last_key]))

# test split
with open(PATH_TO_BINARY_RESULTS + '_test.txt' , 'w', encoding="utf-8") as f:
    # write all except the last line
    for key in test_keys[:-1]:
        f.write(labels[key][0] + '\t' + preprocess(texts[key]) + '\n')
    # write the last line
    last_key = test_keys[-1]
    f.write(labels[last_key][0] + '\t' + preprocess(texts[last_key]))


# In[8]:


# create dataset
assert labels.keys() == texts.keys()

# train split
with open(PATH_TO_RESULTS + '_train.txt' , 'w', encoding="utf-8") as f:
    # write all except the last line
    for key in train_keys[:-1]:
        f.write(labels[key][1] + '\t' + preprocess(texts[key]) + '\n')
    # write the last line
    last_key = train_keys[-1]
    f.write(labels[last_key][1] + '\t' + preprocess(texts[last_key]))

# test split
with open(PATH_TO_RESULTS + '_test.txt' , 'w', encoding="utf-8") as f:
    # write all except the last line
    for key in test_keys[:-1]:
        f.write(labels[key][1] + '\t' + preprocess(texts[key]) + '\n')
    # write the last line
    last_key = test_keys[-1]
    f.write(labels[last_key][1] + '\t' + preprocess(texts[last_key]))


# In[ ]:




