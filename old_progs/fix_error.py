import re

import pandas as pd
import itertools
from transformers import AutoTokenizer

"""
BERTで壊れているデータは無視．
Rinnaに関しては，BERTで壊れているもの以外のエラーを修正する
腹が立ってきた

EX)
BERT：['腹', 'が', 'たっ', 'て', 'き', 'た']
RINNA：['腹', 'が', 'た', 'ってきた', '。']
-> RINNA : ['腹', 'が', 'た', 'って',きた', '。']

1. tokenizer.tokennize(って) + tokenizer.tokennize(きた)
2. ガチ正解のトークンごとに [tokenizer.tokenize(token) for token in ガチ正解tokens]

3. 本来は文字でスパンを考えるべき．文字レベルと単語レベルで組み合わせるようなモデルを考えるのが良い気がする．
 → いろんなトークナイザーで評価可能になる


正しい述語は「たって」であり，BERTの場合span(2, 3)で表せられる.
Rinnaはできないから，「ってきた」を「って」と「きた」に分割後，それぞれに対してtokenizeを行う．
この際，ワード位置のつじつま合わせを行う必要がある．
"""


def remove_prefix(s, a):
    # sの先頭がaで始まる場合のみ削除
    if s.startswith(a):
        return s[len(a):]
    else:
        return -1
    

if __name__ == '__main__':
    pd.set_option('display.max_colwidth',2000)
    tokenizer = AutoTokenizer.from_pretrained("rinna/japanese-gpt-neox-3.6b", use_fast=False)
    DATAPATH = '../Data/common_data_v2_bert.json'
    
    with open(DATAPATH, 'r', encoding="utf-8_sig") as json_file:
        data = pd.read_json(json_file).sample(frac=0.01)

    # BERTのエラーデータを抽出
    p_error, a_error = [],[]
    for idx,(index, row) in enumerate(data.iterrows()):
        if row['predicate']['surface'] != ' '.join(row['sentence'].split()[row['predicate']['word_start']:row['predicate']['word_end']+1]):
            p_error.append(row['abs_id'])
            #print(row['sentence'])
            #print(row['abs_id'])
            #print(row['args'])
            #print()
        for arg in row['args']:
            if arg['surface'] != ' '.join(row['sentence'].split()[arg['word_start']:arg['word_end']+1]):
                a_error.append(row['abs_id'])
    print(len(p_error), len(a_error))


    """
    Rinna のエラーを修正する
    """
    DATAPATH = '../Data/common_data_v2_rinna.json'
    with open(DATAPATH, 'r', encoding="utf-8_sig") as json_file:
        data = pd.read_json(json_file).sample(frac=1)

    # start位置がend位置より1多い問題の解決
    dic = {}
    for idx,(index, row) in enumerate(data.iterrows()):
        if int(row['predicate']['word_start']) - 1 == int(row['predicate']['word_end']):
            row['predicate']['word_start'] = int(row['predicate']['word_start']) - 1
        assert int(row['predicate']['word_start']) <= int(row['predicate']['word_end'])

        for arg in row['args']:
            if int(arg['word_start']) - 1 == int(arg['word_end']):
                arg['word_start'] = int(arg['word_start']) - 1
            assert int(arg['word_start']) <= int(arg['word_end'])

    fixed_abs_id = []
    # surface（正解の文字列）と，rinnaトークナイザーによる分割位置におけるsurface(間違い)の修正
    # とりあえず，先頭は一致していて，後ろのトークナイズを間違えているものとする．
    for idx, (index, row) in enumerate(data.iterrows()):
        # predicateの修正
        # if row['abs_id'] not in p_error:    # bertで壊れているものは無視
        offset_count_p = 0
        correct_surface = ''.join(row['predicate']['surface'].split())
        rinna_surface = ''.join(row['sentence'].split()[row['predicate']['word_start']:row['predicate']['word_end']+1])
        if correct_surface != rinna_surface:
            middle_head = [tokenizer.tokenize(token) for token in row['predicate']['surface'].split()]
            middle_head = list(itertools.chain(*middle_head))
            middle_tail_string = remove_prefix(rinna_surface, correct_surface)
            if middle_tail_string == -1:
                #print(row['abs_id'])
                #print(row['sentence'])
                #print(rinna_surface, correct_surface)
                #print()
                continue
            middle_tail = tokenizer.tokenize(middle_tail_string)
            
            head = row['sentence'].split()[:row['predicate']['word_start']]
            middle = middle_head + middle_tail
            tail = row['sentence'].split()[row['predicate']['word_end']+1:]
            token_list = head + middle + tail
            
            # 辻褄合わせ. 
            offset_count_p = (len(middle) - 1) - (row['predicate']['word_end'] - row['predicate']['word_start'])
            offset_itself = (len(middle_head) - 1) - (row['predicate']['word_end'] - row['predicate']['word_start'])
            row['sentence'] = ' '.join(token_list)
            data.iloc[idx, 2] = ' '.join(token_list)
            for arg in row['args']:
                if row['predicate']['word_start'] < arg['word_start']:
                    arg['word_start'] = arg['word_start'] + offset_count_p
                    arg['word_end'] = arg['word_end'] + offset_count_p
            #row['predicate']['word_start'] = row['predicate']['word_start'] + offset_count_p
            row['predicate']['word_end'] = row['predicate']['word_end'] + offset_itself
            print("## Prideccate ##")
            print(correct_surface, ''.join(row['sentence'].split()[row['predicate']['word_start']:row['predicate']['word_end']+1]))
                
                
        # argの修正
        # if row['abs_id'] not in a_error:    # bertで壊れているものは無視
        offset_count_a = 0
        for arg in row['args']:
            offset_itself = 0
            correct_surface = ''.join(arg['surface'].split())
            rinna_surface = ''.join(row['sentence'].split()[arg['word_start']:arg['word_end']+1])
            if correct_surface != rinna_surface:
                middle_head = [tokenizer.tokenize(token) for token in arg['surface'].split()]
                middle_head = list(itertools.chain(*middle_head))
                middle_tail_string = remove_prefix(rinna_surface, correct_surface)
                if middle_tail_string == -1:
                    middle_tail_string = rinna_surface.replace(correct_surface, f' {correct_surface} ', 1)
                    middle_tail = [tokenizer.tokenize(token) for token in middle_tail_string.split()]
                    middle_tail = list(itertools.chain(*middle_head))
                    continue
                else:
                    middle_tail = tokenizer.tokenize(middle_tail_string)
                
                head = row['sentence'].split()[:arg['word_start']]
                middle = middle_head + middle_tail
                tail = row['sentence'].split()[arg['word_end']+1:]
                token_list = head + middle + tail
                print(offset_count_a)
                print(arg['surface'].split(), row['sentence'].split()[arg['word_start']:arg['word_end']+1])
                print(middle)
                
                # 辻褄合わせ
                offset_count_a = (len(middle) - 1) - (arg['word_end'] - arg['word_start'])
                offset_itself += (len(middle_head) - 1) - (arg['word_end'] - arg['word_start'])
                row['sentence'] = ' '.join(token_list)
                data.iloc[idx, 2] = ' '.join(token_list)
                
                if row['predicate']['word_start'] > arg['word_start']:
                    row['predicate']['word_start'] = row['predicate']['word_start'] + offset_count_a
                    row['predicate']['word_end'] = row['predicate']['word_end'] + offset_count_a
                    
                for arg_temp in row['args']:
                    if arg_temp['word_start'] == arg['word_start']:
                        arg_temp['word_end'] = arg_temp['word_end'] + offset_itself
                    elif arg_temp['word_start'] < arg['word_start']:
                        arg_temp['word_start'] = arg_temp['word_start'] + offset_count_a
                        arg_temp['word_end'] = arg_temp['word_end'] + offset_count_a
                print(arg['surface'].split(), row['sentence'].split()[arg['word_start']:arg['word_end']+1])
                fixed_abs_id.append(row['abs_id'])
                print(offset_count_a)

    #for idx, (index, row) in enumerate(data.iterrows()):
    #    if row['abs_id'] in fixed_abs_id:
    #        print(row['sentence'])

    # bertとrinnaのエラーデータ数が一致するかどうかの確認
    p_error_rinna, a_error_rinna = [],[]
    for idx,(index, row) in enumerate(data.iterrows()):
        correct_surface = ''.join(row['predicate']['surface'].split())
        rinna_surface = ''.join(row['sentence'].split()[row['predicate']['word_start']:row['predicate']['word_end']+1])
        if correct_surface != rinna_surface:
            p_error_rinna.append(row['abs_id'])
            #print(row['sentence'])
            #print(row['abs_id'])
            #print(row['args'])
            #print(correct_surface, rinna_surface)
            #print()
        for arg in row['args']:
            correct_surface = ''.join(arg['surface'].split())
            rinna_surface = ''.join(row['sentence'].split()[arg['word_start']:arg['word_end']+1])
            if correct_surface != rinna_surface:
                a_error_rinna.append(row['abs_id'])
                #rint(correct_surface, rinna_surface)
                #rint()
    print(len(p_error), len(p_error_rinna), len(a_error), len(a_error_rinna))
