from tqdm import tqdm
import pandas as pd
from pandas import DataFrame
import pickle
import numpy as np
import re
import random
import os
import json
def seed_torch(seed=42):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
	np.random.seed(seed)

'''
利用item_train_pair.jsonl制作训练集/验证集
'''
blackitems = ["7d092da468a4ae3c2bca56661ea07bf6"]
with open("item_train_pair.jsonl") as file:
    lines = file.readlines()
data = [json.loads(line) for line in lines]
data = [line for line in data if line["src_item_id"] not in blackitems and line["tgt_item_id"] not in blackitems]

random.shuffle(data)
positive = [i for i in data if i['item_label']=='1' ]
negtive = [i for i in data if i['item_label']=='0']

# 正负样本各取1500
pidx = 1500
train = positive[:-pidx] + negtive[:-pidx]
test = positive[-pidx:] + negtive[-pidx:]

random.shuffle(train)
random.shuffle(test)
with open("item_train_pair_train.jsonl", "w", encoding="utf-8") as file:
    print(f"The number of train: {len(train)}")
    for line in train:
        file.write(json.dumps(line, ensure_ascii=False)+"\n")
with open("item_train_pair_test.jsonl", "w", encoding="utf-8") as file:
    print(f"The number of test: {len(test)}")
    for line in test:
        file.write(json.dumps(line, ensure_ascii=False)+"\n")
                       
'''
制作id输入文本
'''
with open('item_train_pair.jsonl', encoding="utf-8") as f:
    item_train_pair = f.readlines()
with open('item_train_info.jsonl', encoding="utf-8") as f:
    item_train_info = f.readlines()
with open('item_valid_info.jsonl', encoding="utf-8") as f:
    item_valid_info = f.readlines()
with open('item_test_info.jsonl', encoding="utf-8") as f:
    item_test_info = f.readlines()
    
with open('item_valid_pair.jsonl', encoding="utf-8") as f:
    item_valid_pair = f.readlines()
with open('item_test_pair.jsonl', encoding="utf-8") as f:
    item_test_pair = f.readlines()
    
'''
【买3组送一罐本品】
官方旗舰店限量款

'''
# 加入title 组合成文本，写入pkl
item_train = {}
item_valid = {}
item_test = {}
item_train_length = []
item_valid_length = []
item_test_length = []
for line in tqdm(item_train_info):
    line = line.strip()
    line = json.loads(line)
    item_train[line['item_id']] = line['title']
    item_train_length.append(len(line['title']))
for line in tqdm(item_valid_info):
    line = line.strip()
    line = json.loads(line)
    item_valid[line['item_id']] = line['title']
    item_valid_length.append(len(line['title']))
for line in tqdm(item_test_info):
    line = line.strip()
    line = json.loads(line)
    item_test[line['item_id']] = line['title']
    item_test_length.append(len(line['title']))
    # +train 为了知识蒸馏时用
    item_train[line['item_id']] = line['title']
    item_train_length.append(len(line['title']))

with open("item_train_text_title.pkl", "wb") as fp:   #Pickling
    pickle.dump(item_train, fp, protocol = pickle.HIGHEST_PROTOCOL)      
with open("item_valid_text_title.pkl", "wb") as fp:   #Pickling
    pickle.dump(item_valid, fp, protocol = pickle.HIGHEST_PROTOCOL) 
with open("item_test_text_title.pkl", "wb") as fp:   #Pickling
    pickle.dump(item_test, fp, protocol = pickle.HIGHEST_PROTOCOL) 
    
#
#紫色 绿色 官方旗舰店正品  官网 【详情领10元券】 
    
'''
制作pretrain文本数据
'''
# PRETRAIN

item_train_length = []
for line in tqdm(item_train_info+item_valid_info+item_test_info):
    line = line.strip()
    line = json.loads(line)
    item_train[line['item_id']] = line['title']
    item_train_length.append(len(line['title']))

with open("item_pretrain_text_title.pkl", "wb") as fp:   #Pickling
    pickle.dump(item_train, fp, protocol = pickle.HIGHEST_PROTOCOL)      

## 
print(np.mean(item_train_length))
print(len([i for i in item_train_length if i >256]) / len(item_train_length))


'''拟合样本构造'''
def read_data(filename: str) -> DataFrame:
    with open(filename) as file:
        lines = file.readlines()
    data = [json.loads(line) for line in lines]
    data = pd.DataFrame(data)
    return data

data=read_data("item_train_pair_train.jsonl")

# 关联增强
with open('item_train_info.jsonl', encoding="utf-8") as f:
    item_train_info = f.readlines()
    
item_dict = {}
for line in tqdm(item_train_info):
    line = line.strip()
    line = json.loads(line)
    item_dict[line['item_id']] = line['title']
data = data[data['tgt_item_id']!='7d092da468a4ae3c2bca56661ea07bf6'].reset_index(drop=True)
data_train = data.reset_index(drop=True)

######. 关联性数据增强

# 找相似id
sim = {}
# 不相似id
unsim = {}
for i in tqdm(range(len(data_train))):
    src_id = data_train.loc[i, 'src_item_id']
    tgt_id = data_train.loc[i, 'tgt_item_id']
    label = data_train.loc[i, 'item_label']
    if label=='1':
        if src_id in sim.keys():
            sim[src_id].append(tgt_id)
        else:
            sim[src_id] = [tgt_id]
    else:
        if src_id in unsim.keys():
            unsim[src_id].append(tgt_id)
        else:
            unsim[src_id] = [tgt_id]
            
for i in tqdm(range(len(data_train))):
    src_id = data_train.loc[i, 'src_item_id']
    tgt_id = data_train.loc[i, 'tgt_item_id']
    label = data_train.loc[i, 'item_label']
    if label=='1':
        if tgt_id in sim.keys():
            sim[tgt_id].append(src_id)
        else:
            sim[tgt_id] = [src_id]
    else:
        if tgt_id in unsim.keys():
            unsim[tgt_id].append(src_id)
        else:
            unsim[tgt_id] = [src_id]

from itertools import combinations,product 
from collections import defaultdict
# 记录一个id的样本数
sim_id =  defaultdict(int)
unsim_id =  defaultdict(int)
# 相似/不相似样本
sim_pair_new = []
unsim_pair_new = []
# 相似*相似->相似
'''
A:{[B,D]} 相似
-〉相似(B,D)
添加 (A,B)(A,D)(B,D) 并交换[0][1]位置检查sim_pair_new中有无重复
ps. 一个id的正样本数小于5
'''
for key, value in tqdm(sim.items()):
    for i in value:
        if [i,key,'1'] not in sim_pair_new:  # check
            sim_id[key] += 1
            sim_id[i] += 1
            sim_pair_new.append([key,i,'1']) # 软标签
    tmp = list(combinations(value, 2))
    for i in tmp:
        if [i[1],i[0],'1'] not in sim_pair_new:
            if sim_id[i[0]]>5:
                continue
            elif sim_id[i[1]]>5:
                continue
            else:
                sim_pair_new.append([i[0],i[1],'1']) 
                sim_id[i[0]] += 1
                sim_id[i[1]] += 1
print(len(sim_pair_new))

# 相似*不相似->不相似
'''
A:{[B,D]} 相似
A:{[C]} 不相似
-〉不相似(B,C)(C,D)
'''
for key, value in tqdm(unsim.items()):  
    for i in value:
        if [i,key,'0'] not in unsim_pair_new:  # check
            unsim_id[key] += 1
            unsim_id[i] += 1
            unsim_pair_new.append([key,i,'0']) # 标注数据对
            
keys = list(sim.keys())
for key in tqdm(keys):
    if key in unsim.keys():
        sim_pair = sim[key]
        unsim_pair = unsim[key]
        tmp = list(product(sim_pair, unsim_pair))
        for i in tmp:
            if [i[1],i[0],'0'] not in unsim_pair_new:
                if unsim_id[i[0]]>5:
                    continue
                elif unsim_id[i[1]]>5:
                    continue
                else:
                    unsim_pair_new.append([i[0],i[1],'0']) 
                    unsim_id[i[0]] += 1
                    unsim_id[i[1]] += 1
print(len(unsim_pair_new))

sim_df = pd.DataFrame(sim_pair_new, columns=['src_item_id', 'tgt_item_id', 'item_label'])
unsim_df = pd.DataFrame(unsim_pair_new, columns=['src_item_id', 'tgt_item_id', 'item_label'])

df_all_train = sim_df.append(unsim_df).reset_index(drop=True)
df_all_train = df_all_train.drop_duplicates()
print(len(df_all_train))
df_all_train = df_all_train.drop_duplicates(['src_item_id', 'tgt_item_id'],keep=False)
print(len(df_all_train))
df_all_train = df_all_train.reset_index(drop=True)

print(df_all_train.groupby(by=['item_label']).count())

# 用于蒸馏训练的文件
with open("item_train_pair_train_plus.jsonl", "w", encoding="utf-8") as file:     
    num = 0
    for i in range(len(df_all_train)):
        item = {}
        item['src_item_id'] = df_all_train.loc[i, 'src_item_id']
        item['tgt_item_id'] = df_all_train.loc[i, 'tgt_item_id']
        item['item_label'] = df_all_train.loc[i, 'item_label']
        file.write(json.dumps(item, ensure_ascii=False)+"\n")
        num += 1
        
    for line in tqdm(item_test_pair):
        line = line.strip()
        line = json.loads(line)
        src_item_id = line['src_item_id']
        tgt_item_id = line['tgt_item_id']
        item = {}
        item['src_item_id'] = src_item_id
        item['tgt_item_id'] = tgt_item_id
        item['item_label'] = '0'
        file.write(json.dumps(item, ensure_ascii=False)+"\n")
        num += 1
print(num)
