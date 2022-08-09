#!/usr/bin/env python
# coding: utf-8

# Data augmentation to TQI

# In[2]:

import copy
import json
import pandas as pd
import random
import numpy as np
import nltk
from nltk.corpus import wordnet
from copy import deepcopy
import pprint

# In[11]:

# import json as list of dictionaries

tqi_data = []
articles = 0

with open("sentence_level_tokens.json", 'r') as f:
    for line in f:
        tqi_data.append(json.loads(line.strip())) 

# In[12]:

# checkout data

print(type(tqi_data))
example = tqi_data[0]
print(type(example))
print("id: ",example["id"])
print("tokens: ",example["tokens"])
print("tags: ",example["tags"])

# print(len(tqi_data))

# In[41]:

def ListDictToGroups(data):

    lst_NEs =[]

    for sentence in range(0,len(data)):
        for word in range(0,len(data[sentence]["tokens"])):
            if data[sentence]["tags"][word] != "O":
                lst_NEs.append([data[sentence]["tokens"][word],data[sentence]["tags"][word]])

    df_NEs = pd.DataFrame(lst_NEs, columns = ["token","tag"])

    groups = [df_NEs for _, df_NEs in df_NEs.groupby("tag")]

    return groups
groups = ListDictToGroups(tqi_data)

def collectMentions(data):

    data = copy.deepcopy(data)
    lst_NEs =[]

    for sentence in range(0,len(data)):
        segment = [data[sentence]["tokens"][0]]

        for word in range(1,len(data[sentence]["tokens"])):

            if data[sentence]["tags"][word] == data[sentence]["tags"][word-1]:
                segment.append(data[sentence]["tokens"][word])
            else:
                lst_NEs.append([segment, data[sentence]["tags"][word-1]])
                segment = [data[sentence]["tokens"][word]]

    df_NEs = pd.DataFrame(lst_NEs, columns = ["mention","tag"])
    groups = [df_NEs for _, df_NEs in df_NEs.groupby("tag")]

    return groups
groups_mentions = collectMentions(tqi_data)

# Data augmentation methods

def LabelWiseTokenReplacement(rate, data, groups):
    
    data_mod = copy.deepcopy(data)

    group_list = []

    for group in groups:
        group_list.append(group["tag"].unique()[0])

    group_definitions = { group_list[i]: i for i in range(len(group_list))}
    
    
    for sentence in range(0,len(data_mod)):
        for word in range(0,len(data_mod[sentence]["tokens"])):
            if data_mod[sentence]["tags"][word] != "O":
                if np.random.binomial(1, rate, size=None) == 1:
                
                    group_tag = groups[group_definitions[data_mod[sentence]["tags"][word]]]
                    data_mod[sentence]["tokens"][word]= group_tag.sample()["token"].item()
                
    return data_mod

def SynonymReplacement(rate, data):

    data = copy.deepcopy(data)

    for sentence in range(0,len(data)):

        for word in range(0,len(data[sentence]["tokens"])):
            if np.random.binomial(1, rate, size=None) == 1:
                try:
                    if data[sentence]["tokens"][word] not in ["in","In","It","it","does","Does","IAEA","have","Have","be","Be","less","Less","He","he","Pesos","Inc","inc","acts","Acts","an","An","units"]:
                        syn = wordnet.synsets(str(data[sentence]["tokens"][word]))[0].lemma_names()[0]
                        data[sentence]["tokens"][word] = syn
                except:
                    pass
    return data

def SegmentShuffle(rate, data):

    data = copy.deepcopy(data)

    for sentence in range(0,len(data)):
        segment = [data[sentence]["tokens"][0]]

        for word in range(1,len(data[sentence]["tokens"])):

            if data[sentence]["tags"][word] == data[sentence]["tags"][word-1]:
                segment.append(data[sentence]["tokens"][word])
            else:
                if np.random.binomial(1, rate, size=None):
                    random.shuffle(segment)
                    for k in range(0,len(segment)):
                        data[sentence]["tokens"][word-len(segment)+k] = segment[k]
                    
                segment = [data[sentence]["tokens"][word]]

        return data

def MentionReplacement(rate, data, groups_mentions):
    data = copy.deepcopy(data)
    group_list = []

    for group in groups_mentions:
        group_list.append(group["tag"].unique()[0])

    group_definitions = { group_list[i]: i for i in range(len(group_list))}

    for sentence in range(0,len(data)):
        
        sentence_segments = []
        sentence_tags = []
        segment_token = [data[sentence]["tokens"][0]]
        segment_tags = [data[sentence]["tags"][0]]

        for word in range(1,len(data[sentence]["tokens"])):
            # print(data[sentence]["tags"][word] )

            if data[sentence]["tags"][word] == data[sentence]["tags"][word-1] and ((word+1)!=len(data[sentence]["tokens"])):
                segment_token.append(data[sentence]["tokens"][word])
                segment_tags.append(data[sentence]["tags"][word])
            else:
                sentence_segments.append(segment_token)
                sentence_tags.append(segment_tags)
                segment_token = [data[sentence]["tokens"][word]]
                segment_tags = [data[sentence]["tags"][word]]
        # print(sentence_segments)
        for segment in range(0,len(sentence_segments)):
            # print(sentence_tags[segment])
            if sentence_tags[segment][0] != "O":
                if np.random.binomial(1, rate, size=None):
                    group_tag = groups_mentions[group_definitions[sentence_tags[segment][0]]]
                    sentence_segments[segment] = group_tag.sample()["mention"].item()
                    sentence_tags[segment] = [sentence_tags[segment][0]]*len(sentence_segments[segment])

        new_sentence = [x for xs in sentence_segments for x in xs]
        data[sentence]["tokens"] = new_sentence
        new_tags = [x for xs in sentence_tags for x in xs]
        data[sentence]["tags"] = new_tags
    return data

# In[51]:

for df_group in groups_mentions:
    # print(df_group["mention"])
    
    for row in df_group.iterrows():
        print(row[1][0])
        print(" ".join(map(str,row[1][0])))
        row[1][0] = " ".join(map(str,row[1][0]))

    print(df_group)

    # mention = 

#%%

groups_mentions_unique = []

for df_group in range(0,len(groups_mentions)):
    groups_mentions_unique.append(groups_mentions[df_group].drop_duplicates())

print(groups_mentions_unique)


#%%
for df in groups_mentions_unique:
    print(df.iloc[0])

#%%

dash = '-' * 40

print("# of sentences in labeled TQI Data: ", len(tqi_data))

print(dash)
print("{:<15}{:^10}{:<25}".format("Label:","NEs","examples"))
print(dash)
for group in groups_mentions_unique:
    print("{:<15}{:^10}{:<20}".format(group["tag"].unique()[0],len(group["tag"]),group.sample()["mention"].iloc[0]),", ",group.sample()["mention"].iloc[0])
    # print("\t\t",type(group.sample()["mention"].iloc[0]))


#%%
# mention = ["D", "-", "Wave", "Systems"]
# mention = " ".join(map(str,mention))
# print(mention)


# print(MR_DA[:1])
# print(tqi_data[:1])

rate = 1

MR_DA = MentionReplacement(rate,tqi_data[:1],groups_mentions)
LwTR_DA = LabelWiseTokenReplacement(1, tqi_data, groups)

SIS_DA = SegmentShuffle(rate,tqi_data)
SR_DA = SynonymReplacement(rate, tqi_data)


#%%

# Printing for Example
def unique(list1):
    x = np.array(list1)
    y = np.unique(x)
    return y

ids = []
for i in range(0,len(tqi_data)):
    ids.append(tqi_data[i]["id"])

print(ids)

ids_unique = unique(ids)
print(ids_unique)
print(len(ids_unique))




# In[53]:

#run all of LwTR (64 times)

LwTR_output = copy.deepcopy(tqi_data)

rate = 1

for i in range(0,63):
    da = LabelWiseTokenReplacement(rate, tqi_data, groups)
    LwTR_output = LwTR_output + da

with open('TQI_DA_lwtr_rate70.json', 'w') as f:
    json.dump(LwTR_output, f,indent = 4)

#%%

#run all of SynonymReplacement (64 times)

SR_output = copy.deepcopy(tqi_data)

rate = 0.7

for i in range(0,63):
    da = SynonymReplacement(rate, tqi_data)
    SR_output = SR_output + da



# In[11]:

SIS_output = copy.deepcopy(tqi_data)

rate = 0.7

for i in range(0,63):
    da = SegmentShuffle(rate,tqi_data)
    SIS_output = SIS_output + da

print(len(SIS_output))

with open("TQI_DA_SIS_rate70.json",'w') as out:
    json.dump(SIS_output, out,indent = 4)

# In[11]:

MR_output = copy.deepcopy(tqi_data)
print(len(MR_output))
rate = 1

for i in range(0,63):
    da = MentionReplacement(rate,tqi_data,groups_mentions)
    MR_output = MR_output + da

print(len(MR_output))

print(MR_output[0])
print(MR_output[2755])

with open("TQI_DA_MR_rate100.json",'w') as out:
    json.dump(MR_output, out,indent = 4)

# In[ ]:

with open("TQI_DA_SR_rate70.json",'w') as out:
    json.dump(SR_output, out,indent = 4)

# In[11]:


# In[ ]:


# In[ ]:

rate = 1
MR_done = MentionReplacement(rate, tqi_data, groups_mentions)

print(len(MR_done))
print(type(MR_done))
print(type(MR_done[0]))

# In[ ]:

with open("TQI_DA_SIS_rate70.json",'r') as f:
    test = json.load(f)
    print(len(test))
    print(test[2755])

#%%

# Run all 3 types consecutively

LwTR_output = copy.deepcopy(tqi_data)

rate = 0.3

for i in range(0,62):
    da = LabelWiseTokenReplacement(rate, tqi_data, groups)
    LwTR_output = LwTR_output + da


da = SynonymReplacement(rate, LwTR_output)
da_all = SegmentShuffle(rate,da)

da_all = tqi_data + da_all

with open("TQI_DA_all_rate30.json",'w') as out:
    json.dump(da_all, out,indent = 4)

len(da_all)

# In[11]:

SIS_output = copy.deepcopy(tqi_data)

print(len(SIS_output))

rate = 0.7

for i in range(0,63):
    
    SIS_output = SIS_output + da

print(len(SIS_output))


