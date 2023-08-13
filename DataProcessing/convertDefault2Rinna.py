import re
import json
import pandas as pd
import itertools
from transformers import AutoTokenizer


def convert2rinna(row, tokenizer):
    rinna_tokens = []
    offset_index_s = []   # index:前のワード開始位置, value:前のワード位置からのoffset数
    offset_index_e = []   # index:前のワード終了位置, value:前のワード位置からのoffset数
    for token in row['sentence'].split(' '):
        if token == ' ' or token =='':
            continue
        tokens = tokenizer.tokenize(token)
        assert len(tokens) > 0, print(token, '##########',row['sentence'])
        try:
            next_offset = offset_index_e[-1] + len(tokens) - 1
        except:
            next_offset = len(tokens) - 1
        try:
            offset_index_s.append(offset_index_e[-1])
        except:
            offset_index_s.append(0)
            
        offset_index_e.append(next_offset)
        rinna_tokens += tokens
        #print(token,tokens)
    #print(row['sentence'].split(' '))
    #print(rinna_tokens)
    #print(offset_index_s)
    #print(offset_index_e)
    #print()
        
    # 位置修正(辞書型なので参照)
    row['predicate']['word_start'] += offset_index_s[row['predicate']['word_start']]
    row['predicate']['word_end'] += offset_index_e[row['predicate']['word_end']]
    for arg in row['args']:
        arg['word_start'] += offset_index_s[arg['word_start']]
        arg['word_end'] += offset_index_e[arg['word_end']]
    rinna_sent = ' '.join(rinna_tokens)
    
    #print(arg['surface'], ' '.join(rinna_sent.split()[arg['word_start']:arg['word_end']+1]))
    #print(row['predicate']['surface'],' '.join(rinna_sent.split()[row['predicate']['word_start']:row['predicate']['word_end']+1]))
    #print(rinna_tokens)
    return rinna_sent   # sentは辞書型でないので，値渡し
    

if __name__ == '__main__':
    pd.set_option('display.max_colwidth',2000)
    tokenizer = AutoTokenizer.from_pretrained("rinna/japanese-gpt-neox-3.6b", use_fast=False)
    DATAPATH = '../Data/data_default.json'
    
    with open(DATAPATH, 'r', encoding="utf-8_sig") as json_file:
        data = pd.read_json(json_file).sample(frac=1)
        print(len(data))
        
    
    # start位置がend位置関係の確認
    dic = {}
    for idx,(index, row) in enumerate(data.iterrows()):
        assert int(row['predicate']['word_start']) <= int(row['predicate']['word_end'])
        for arg in row['args']:
            assert int(arg['word_start']) <= int(arg['word_end'])
    
    # エラーデータ数をカウント
    p_error, a_error = [],[]
    for idx,(index, row) in enumerate(data.iterrows()):
        if ''.join(row['predicate']['surface'].split()) != ''.join(row['sentence'].split()[row['predicate']['word_start']:row['predicate']['word_end']+1]):
            p_error.append(row['abs_id'])
            #print(row['sentence'])
            #print(row['abs_id'])
            #print(row['args'])
            #print()
        for arg in row['args']:
            if ''.join(arg['surface'].split()) != ''.join(row['sentence'].split()[arg['word_start']:arg['word_end']+1]):
                a_error.append(row['abs_id'])
    print(len(p_error), len(a_error))
    
    # Rinnaに変換
    for idx, (index, row) in enumerate(data.iterrows()):
        #print(data.iloc[idx, 2])
        data.iloc[idx, 2] = convert2rinna(row, tokenizer)
        #print(data.iloc[idx, 2])
        #print()
    
    # エラーデータ数をカウント. TODO:存在しない感じは<0x>で表されるため，正確な数を算出できていない
    p_error, a_error = [],[]
    for idx,(index, row) in enumerate(data.iterrows()):
        if ''.join(row['predicate']['surface'].split()) != ''.join(row['sentence'].split()[row['predicate']['word_start']:row['predicate']['word_end']+1]):
            p_error.append(row['abs_id'])
            #print(row['sentence'])
            #print(row['abs_id'])
            #print(row['args'])
            print(row['predicate']['surface'])
            print(' '.join(row['sentence'].split()[row['predicate']['word_start']:row['predicate']['word_end']+1]))
            print()
        for arg in row['args']:
            if ''.join(arg['surface'].split()) != ''.join(row['sentence'].split()[arg['word_start']:arg['word_end']+1]):
                a_error.append(row['abs_id'])
    print(len(p_error), len(a_error))
    data.to_json(f"temp.json",orient='records',force_ascii=False)
    with open(f"temp.json") as f:
        data = json.load(f)
    with open(f"temp.json", mode="w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)