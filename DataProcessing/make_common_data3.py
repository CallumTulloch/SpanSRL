import json
import pandas as pd


pd.set_option('display.max_colwidth',2000)
with open('../Data/data_default.json', 'r', encoding="utf-8_sig") as json_file:
    ddf = pd.read_json(json_file)
    #ddf=ddf.sample(frac=0.01)
    print(len(ddf))
with open('../Data/common_data_v2_rinna.json', 'r', encoding="utf-8_sig") as json_file:
    rdf = pd.read_json(json_file)
    #rdf=rdf[rdf['abs_id'].isin(ddf['abs_id'].to_list())]
    print(len(rdf))

indication_lists = []
for idx, (index, row) in enumerate(ddf.iterrows()):
    # 述語情報の更新
    row['predicate']['char_start_without_blank'] = row['predicate']['char_start'] - row['predicate']['word_start']
    row['predicate']['char_end_without_blank'] = row['predicate']['char_end'] - row['predicate']['word_end']
    
    # arg情報の更新
    for arg in row['args']:
        arg['char_start_without_blank'] = arg['char_start'] - arg['word_start']
        arg['char_end_without_blank'] = arg['char_end'] - arg['word_end']

    # 切れる位置をしめすリスト作成
    indication_list = []
    for i, c in enumerate(row['sentence'].split()):
        indication_list += [i] * len(c)
    assert len(''.join(row['sentence'].split())) == len(indication_list), print('文字数が一致しなくてはならない')
    indication_lists.append(indication_list)
ddf['split_indication'] = indication_lists

indication_lists = []
for idx, (index, row) in enumerate(rdf.iterrows()):
    # 述語情報の追加
    row['predicate']['char_start_without_blank'] = row['predicate']['char_start'] - row['predicate']['word_start']
    row['predicate']['char_end_without_blank'] = row['predicate']['char_end'] - row['predicate']['word_end']
    
    # arg情報の追加
    for arg in row['args']:
        arg['char_start_without_blank'] = arg['char_start'] - arg['word_start']
        arg['char_end_without_blank'] = arg['char_end'] - arg['word_end']

    # 切れる位置をしめすリスト作成
    indication_list = []
    for i, c in enumerate(row['sentence'].split()):
        indication_list += [i] * len(c)
    assert len(''.join(row['sentence'].split())) == len(indication_list), print('文字数が一致しなくてはならない')
    indication_lists.append(indication_list)
rdf['split_indication'] = indication_lists
    

"""
rinnaの各sentenceについて（複数の述語データ），predicate, arg の切り方を以下の手順で修正する．
イメージとしては，切れるべきところで切れているかどうかのみを意識すればよい．
1. 開始位置とその前で値が切り替わっていない場合： 開始位置以降のsplit_indicationを全て＋1する．(がた，って，きた2[23344] -> が，た，って，きた2[34455])
2. 終了位置とその後ろで値が切り替わっていない場合： 終了位置＋１以降のsplit_indicationを全て＋1する．(た，ってきた23333 -> た，って，きた23344)
その後，修正されたsplit_indicationとchar_start_without_blankを基に，word_startとword_endを更新する．具体的には，その文字数位置におけるindicationの値がワード位置となる．
このデータセットのトークナイズは，token_to_ids が出来なかった場合に，tokenizer.tokenize を行うようにすべき
"""

# TODO:ddfとrdfの両方に共通して存在するデータに関してのみ，この操作を行うことができるため，修正しきれていないデータも存在する．いつか解決したい．
sentID2splitIndication = {}
grouped_rdf = rdf.groupby('sentenceID')

for sentenceID, rgroup in grouped_rdf:
    # print(rgroup['split_indication'].to_list()[0])
    sentID2splitIndication[sentenceID] = rgroup['split_indication'].to_list()[0]
    #print(len(rgroup))
    offset_digits = [0] * len(sentID2splitIndication[sentenceID])
    for index, row in rgroup.iterrows():
        if ddf[ddf['abs_id']==row['abs_id']].empty:
            #print(f'There is no such data. : abs_id = {row["abs_id"]}')
            continue
        # print(ddf[ddf['abs_id']==row['abs_id']]['predicate'])
        
        # predicateに関して修正
        p_gold_start = ddf[ddf['abs_id']==row['abs_id']]['predicate'].iloc[0]['char_start_without_blank']
        p_gold_end = ddf[ddf['abs_id']==row['abs_id']]['predicate'].iloc[0]['char_end_without_blank']
        #print(row['sentence'])
        #print(' ',row['split_indication'])
        #print('predicate : ', p_gold_start, p_gold_end)
        if p_gold_start != 0: # 開始位置0の場合，それより前は存在しない．
            if sentID2splitIndication[sentenceID][p_gold_start] - sentID2splitIndication[sentenceID][p_gold_start-1] == 0:
                offset = [0]*(p_gold_start) + [1]*(len(sentID2splitIndication[sentenceID]) - p_gold_start)
                offset_digits = [a + b for a, b in zip(offset, offset_digits)]
                #print('a',offset_digits)
                #print(row['sentence'])
                #print(ddf[ddf['abs_id']==row['abs_id']]['sentence'].iloc[0])
                #print(sentID2splitIndication[sentenceID])
                #print([a + b for a, b in zip(sentID2splitIndication[sentenceID], offset_digits)])
                #exit()
                assert len(offset) == len(sentID2splitIndication[sentenceID])
        
        if p_gold_end != len(sentID2splitIndication[sentenceID])-1: # 終了位置が文字列数の場合，それより後は存在しない．
            if sentID2splitIndication[sentenceID][p_gold_end] - sentID2splitIndication[sentenceID][p_gold_end+1] == 0:
                offset = [0]*(p_gold_end+1) + [1]*(len(sentID2splitIndication[sentenceID]) - p_gold_end - 1)
                offset_digits = [a + b for a, b in zip(offset, offset_digits)]
                #print('b',offset_digits)
                assert len(offset) == len(sentID2splitIndication[sentenceID])
        
        # argに関して修正
        for idx, arg in enumerate(row['args']):
            a_gold_start = ddf[ddf['abs_id']==row['abs_id']]['args'].iloc[0][idx]['char_start_without_blank']
            a_gold_end = ddf[ddf['abs_id']==row['abs_id']]['args'].iloc[0][idx]['char_end_without_blank']
            #print('arg : ', a_gold_start, a_gold_end)
            if a_gold_start != 0 and p_gold_end+1 != a_gold_start: # 開始位置0の場合，それより前は存在しない．2重に修正しないようにpredicateとargの位置関係を条件に
                if idx == 0:    # 前のargと今回のargで2重に修正しないようにするための条件
                    if sentID2splitIndication[sentenceID][a_gold_start] - sentID2splitIndication[sentenceID][a_gold_start-1] == 0:
                        offset = [0]*(a_gold_start) + [1]*(len(sentID2splitIndication[sentenceID]) - a_gold_start)
                        offset_digits = [a + b for a, b in zip(offset, offset_digits)]
                        #print('c',offset_digits)
                        assert len(offset) == len(sentID2splitIndication[sentenceID])
                else:
                    if ddf[ddf['abs_id']==row['abs_id']]['args'].iloc[0][idx-1]['char_end_without_blank'] + 1 != a_gold_start:
                        if sentID2splitIndication[sentenceID][a_gold_start] - sentID2splitIndication[sentenceID][a_gold_start-1] == 0:
                            offset = [0]*(a_gold_start) + [1]*(len(sentID2splitIndication[sentenceID]) - a_gold_start)
                            offset_digits = [a + b for a, b in zip(offset, offset_digits)]
                            #print('c',offset_digits)
                            assert len(offset) == len(sentID2splitIndication[sentenceID])
                    
            if a_gold_end != len(sentID2splitIndication[sentenceID])-1 and p_gold_start-1 != a_gold_end: # 終了位置が文字列数の場合，それより後は存在しない．
                if sentID2splitIndication[sentenceID][a_gold_end] - sentID2splitIndication[sentenceID][a_gold_end+1] == 0:
                    offset = [0]*(a_gold_end+1) + [1]*(len(sentID2splitIndication[sentenceID]) - a_gold_end - 1)
                    offset_digits = [a + b for a, b in zip(offset, offset_digits)]
                    #print('d',offset_digits)
                    assert len(offset) == len(sentID2splitIndication[sentenceID])
        
    # split_indication の更新
    #print(sentID2splitIndication[sentenceID])
    sentID2splitIndication[sentenceID] = [a + b for a, b in zip(sentID2splitIndication[sentenceID], offset_digits)]
    #print(sentID2splitIndication[sentenceID])
    #print()


def extract_words_from_indices(text, indices):
    #print(text, indices)
    word_list = []
    current_word = ''
    previous_index = indices[0]
    for char, index in zip(text, indices):
        if index == previous_index:
            current_word += char
        else:
            word_list.append(current_word)
            current_word = char
            previous_index = index
    word_list.append(current_word)  # 最後の単語をリストに追加
    #print(word_list)
    #print()
    return ' '.join(word_list)


# sentence, sentenceID, word位置の修正
rdf['split_indication'] = rdf['sentenceID'].map(sentID2splitIndication)
rdf['sentence'] = rdf.apply(lambda x:extract_words_from_indices(''.join(x['sentence'].split()), x['split_indication']), axis=1)

for index, row in rdf.iterrows():
    # 述語情報の更新
    #print(rdf.loc[index, 'sentence'])
    #print(row['split_indication'])
    #print(rdf.loc[index, 'predicate']['word_start'], row['predicate']['char_start_without_blank'], row['split_indication'][row['predicate']['char_start_without_blank']])
    #print()
    rdf.loc[index, 'predicate']['word_start'] = row['split_indication'][row['predicate']['char_start_without_blank']]
    rdf.loc[index, 'predicate']['word_end'] = row['split_indication'][row['predicate']['char_end_without_blank']]
    
    # arg情報の更新
    for i, arg in enumerate(row['args']):
        rdf.loc[index, 'args'][i]['word_start'] = row['split_indication'][arg['char_start_without_blank']]
        rdf.loc[index, 'args'][i]['word_end'] = row['split_indication'][arg['char_end_without_blank']]

    
# エラーデータ数をカウント. TODO:存在しない漢字は<0x>で表されるため，正確な数を算出できていない
p_error, a_error = [],[]
for idx,(index, row) in enumerate(rdf.iterrows()):
    if ''.join(row['predicate']['surface'].split()) != ''.join(row['sentence'].split()[row['predicate']['word_start']:row['predicate']['word_end']+1]):
        p_error.append(row['abs_id'])
        # print(row['sentence'])
        # print(row['abs_id'])
        # print(row['args'])
        print(row['predicate']['surface'])
        print(' '.join(row['sentence'].split()[row['predicate']['word_start']:row['predicate']['word_end']+1]))
        print()
    for arg in row['args']:
        if ''.join(arg['surface'].split()) != ''.join(row['sentence'].split()[arg['word_start']:arg['word_end']+1]):
            a_error.append(row['abs_id'])
print(len(p_error), len(a_error))

# 保存と変換
rdf.to_json("../Data/common_data_v2_rinna3.json", orient='records',force_ascii=False)
print(len(rdf))

with open("../Data/common_data_v2_rinna3.json") as f:
    rdf = json.load(f)
with open("../Data/common_data_v2_rinna3.json", mode="w") as f:
    json.dump(rdf, f, indent=2, ensure_ascii=False)
