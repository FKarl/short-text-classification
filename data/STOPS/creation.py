#!/usr/bin/env python
# coding: utf-8

# # Short Texts Of Products and Services (STOPS)

# In[16]:


# define paths
PATH_TO_YELP = 'source/yelp/yelp_academic_dataset_business.json'
PATH_TO_MAVE_POS = 'source/mave/mave_positives.jsonl'
PATH_TO_MAVE_NEG = 'source/mave/mave_negatives.jsonl'

PATH_TO_BINARY_RESULTS = 'STOPS-2-long/STOPS-2'
PATH_TO_RESULTS = 'STOPS-long/STOPS'


# In[17]:


# define parameters
RANDOM_STATE = 42
TRAIN_SPLIT = 0.7


# ## Load Data

# In[18]:


import json


def read_jsonline(file_path):
    # read file line by line
    with open(file_path, 'r', encoding="utf8") as f:
        while True:
            jsonline = f.readline()
            if not jsonline:
                break
            yield json.loads(jsonline)


# In[19]:


def countWords(sentence):
    words = sentence.split()
    return len(words)


# In[20]:


import string
import unicodedata


def preprocess(text):
    # lowercase
    text = text.lower()
    # remove multiple spaces
    text = ' '.join(text.split())
    # remove punctuation
    text = text.translate(text.maketrans('', '', string.punctuation))
    # map to unicode
    text = ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')
    return text


# ### Mave

# In[21]:


def count_mave(path, label_counter, limit_words, limits_cat, min_char_count):
    for line in read_jsonline(path):
        if not line:
            break
        if (len(line['paragraphs']) >= 1) and (line['paragraphs'][0]['source'] == 'title') and (line['category']):
            text = line['paragraphs'][0]['text']
            if (countWords(text) <= limit_words) and (len(text) >= min_char_count):
                label = line['category']
                if label not in label_counter:
                    label_counter[label] = 0
                if label_counter[label] < limits_cat:
                    label_counter[label] += 1


# In[22]:


def load_mave(path, texts, label_counter, top_cats, limit_words, limits_cat, min_char_count):
    for line in read_jsonline(path):
        if not line:
            break
        if (len(line['paragraphs']) >= 1) and (line['paragraphs'][0]['source'] == 'title') and (line['category']):
            text = line['paragraphs'][0]['text']
            label = line['category']
            if label not in label_counter:
                label_counter[label] = 0
            if (label in top_cats) and (label_counter[label] < limits_cat):
                texts.append((label, preprocess(text)))
                label_counter[label] += 1


# In[23]:


size = 5000
min_char_count = 5
limit_words = 7
mave_texts = []
label_counter = dict()

# count most frequent labels
count_mave(PATH_TO_MAVE_POS, label_counter, limit_words, size, min_char_count)
count_mave(PATH_TO_MAVE_NEG, label_counter, limit_words, size, min_char_count)

top_cats = sorted(label_counter.items(), key=lambda x: x[1], reverse=True)
# select 20 most frequent labels
top_cats = top_cats[:20]
top_cats = [cat for cat, count in top_cats]


# In[24]:


# load most frequent labels
label_counter = dict()

load_mave(PATH_TO_MAVE_POS, mave_texts, label_counter, top_cats, limit_words, size, min_char_count)
load_mave(PATH_TO_MAVE_NEG, mave_texts, label_counter, top_cats, limit_words, size, min_char_count)


# ### Yelp Business Data

# In[25]:


# count label occurences
count_labels = dict()
for line in read_jsonline(PATH_TO_YELP):
    if not line:
        break

    if line['categories'] and line['name']:
        label = line['categories']

        categories = label.replace(", ", ",").split(",")
        for category in categories:
            if category not in count_labels:
                count_labels[category] = 0
            count_labels[category] += 1


# In[26]:


# order by occurrences
count_labels = sorted(count_labels.items(), key=lambda x: x[1], reverse=True)


# In[27]:


# map data points to the most frequent label
yelp_texts = []
limit = 12000
label_counter = dict()

for line in read_jsonline(PATH_TO_YELP):
    if not line:
        break

    if line['categories'] and line['name']:
        categories = line['categories'].replace(", ", ",").split(",")
        # find first label in label list
        for label, _ in count_labels:
            if label in categories:
                if label not in label_counter:
                    label_counter[label] = 0
                label_counter[label] += 1

                if label_counter[label] <= limit:
                    text = str(line['name'])
                    yelp_texts.append((label, preprocess(text)))

                break


# In[28]:


# print length
print("YELP length:",len(yelp_texts))
print("MAVE length:",len(mave_texts))

# print labels
yelp_set = set()
for label, _ in yelp_texts:
    yelp_set.add(label)
mave_set = set()
for label, _ in mave_texts:
    mave_set.add(label)
print("YELP labels:",yelp_set)
print("MAVE labels:",mave_set)


# ## Create Dataset

# In[29]:


# create splits with fixed seed
import random

random.seed(RANDOM_STATE)

# yelp
yelp_length = len(yelp_texts)
yelp_train_length = int(yelp_length * TRAIN_SPLIT)
yelp_train_indices = random.sample(range(yelp_length), yelp_train_length)
yelp_test_indices = [i for i in range(yelp_length) if i not in yelp_train_indices]

# mave
mave_length = len(mave_texts)
mave_train_length = int(mave_length * TRAIN_SPLIT)
mave_train_indices = random.sample(range(mave_length), mave_train_length)
mave_test_indices = [i for i in range(mave_length) if i not in mave_train_indices]

print("yelp train:", len(yelp_train_indices), "test:", len(yelp_test_indices))
print("mave train:", len(mave_train_indices), "test:", len(mave_test_indices))




# ### Create Binary Dataset

# In[30]:


# train
with open(PATH_TO_BINARY_RESULTS + '_train.txt', 'w', encoding="utf8") as f:
    # write all except the last line
    for i in yelp_train_indices:
        f.write('service\t' + yelp_texts[i][1] + '\n')
    for i in mave_train_indices[:-1]:
        f.write('product\t' + mave_texts[i][1] + '\n')
    f.write('product\t' + mave_texts[mave_train_indices[-1]][1])

# test
with open(PATH_TO_BINARY_RESULTS + '_test.txt', 'w', encoding="utf8") as f:
    # write all except the last line
    for i in yelp_test_indices:
        f.write('service\t' + yelp_texts[i][1] + '\n')
    for i in mave_test_indices[:-1]:
        f.write('product\t' + mave_texts[i][1] + '\n')
    f.write('product\t' + mave_texts[mave_test_indices[-1]][1])


# ### Create Multiclass Dataset

# In[31]:


# train
with open(PATH_TO_RESULTS + '_train.txt', 'w', encoding="utf8") as f:
    # write all except the last line
    for i in yelp_train_indices:
        f.write(yelp_texts[i][0] + '\t' + yelp_texts[i][1] + '\n')
    for i in mave_train_indices[:-1]:
        f.write(mave_texts[i][0] + '\t' + mave_texts[i][1] + '\n')
    f.write(mave_texts[mave_train_indices[-1]][0] + '\t' + mave_texts[mave_train_indices[-1]][1])

# test
with open(PATH_TO_RESULTS + '_test.txt', 'w', encoding="utf8") as f:
    # write all except the last line
    for i in yelp_test_indices:
        f.write(yelp_texts[i][0] + '\t' + yelp_texts[i][1] + '\n')
    for i in mave_test_indices[:-1]:
        f.write(mave_texts[i][0] + '\t' + mave_texts[i][1] + '\n')
    f.write(mave_texts[mave_test_indices[-1]][0] + '\t' + mave_texts[mave_test_indices[-1]][1])

