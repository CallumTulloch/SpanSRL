import pandas as pd


with open('../Data/common_data_v2_bert.json', 'r', encoding="utf-8_sig") as json_file:
    data = pd.read_json(json_file)
    print(len(data))
with open('../Data/common_data_v2_rinna2.json', 'r', encoding="utf-8_sig") as json_file:
    data = pd.read_json(json_file)
    print(len(data))


import collections


#with open('../Data/data_v3_del_argandpred_for_span.json', 'r',encoding='utf-8_sig') as json_file:
#    data = pd.read_json(json_file, orient='records', lines=True)
# （データのトークン数，predのトークン数，ラベル数） の平均，マックス，ミニマムのテーブル．また，データ数 を算出
data['sent_token'] = data['sentence'].map(lambda x: len(x.split(' ')))
data['pred_token'] = data['predicate'].map(lambda x: len(x['surface'].split(' ')))
data['labels_per_a_pred'] = data['args'].map(lambda x: len(x))
data['arg_len'] = data['args'].map(lambda x: max(len(arg['surface'].split(' ')) for arg in x))

data_list = [[data['sent_token'].max(), data['sent_token'].min(), data['sent_token'].mean()]]
data_list.append([data['pred_token'].max(), data['pred_token'].min(), data['pred_token'].mean()])
data_list.append([data['labels_per_a_pred'].max(), data['labels_per_a_pred'].min(), data['labels_per_a_pred'].mean()])

df = pd.DataFrame(data_list, index=['sent_token','pred_token','labels_per_a_pred'], columns=['max', 'min', 'mean'])

arg_list=[]
for args in data['args']:
    for arg in args: 
        arg_list.append(arg['argrole'])
s= set(arg_list)
print(len(s))
c = collections.Counter(arg_list)
c = sorted(list(c.items()), key=lambda x:x[1], reverse=True)

arg_list=[]
for args in data['args']:
    for arg in args: 
        arg_list.append(len(arg['surface'].split(' ')))
print('Ave of arg length', round( sum(arg_list)/len(arg_list), 3) )
print('max and min of arg length', max(arg_list), min(arg_list))
print('Num of data = ', len(data))
print('Num of data (arg <= 30) = ',len(data[data['arg_len'] <= 30]))
print(df.round(2))
print('max arg len = ', max(data['arg_len']))
print('max fid = ', max(data['predicate'].map(lambda x:x['frameID'])))
print('max fid = ', min(data['predicate'].map(lambda x:x['frameID'])))