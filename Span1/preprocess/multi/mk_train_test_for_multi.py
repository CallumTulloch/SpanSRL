import pandas as pd
import numpy as np
import itertools
from sklearn.model_selection import train_test_split

def get_o_label_span(role_start_end, max_length):
    o_spans = []
    role_start_end = sorted(role_start_end, key=lambda x:x[1])
    
    role, first_start, first_end = role_start_end.pop(0)
    if (first_start != 0) and (first_start < max_length):
        o_spans.append(('O', 0, first_start-1))
    
    prev_end = first_end
    for role, start, end in role_start_end:
        if start > max_length-1:  # 順番に並んでおり，そのスタートがマックスを超えていいたら終わり
            break
        if prev_end +1 != start: # 連続でなく，始まりがマックスを超えてない
            o_spans.append(('O', prev_end+1, start-1))
        prev_end = end
    
    if (prev_end < max_length-1):
        o_spans.append((('O', prev_end+1, max_length-1)))
    return o_spans

# GCN のデータはメインのプログラムにおいてPadding する．
# 対応するモデル．lesspadding, lesspadding_gcn, multi_task, multi_task_gcn
def get_train_test(MAX_LENGTH, MAX_ARGUMENT_SEQUENCE_LENGTH, data, lab2id, fid2id):
    # 正解ラベル作成
    gold_standerd_labels = []
    role_start_end_lists=[]
    for args, pred in zip(data['args'], data['predicate']):
        role_start_end = [(arg['argrole'], arg['word_start'], arg['word_end']) for arg in args]
        role_start_end.append(('V', pred['word_start'], pred['word_end']))
        role_start_end += get_o_label_span(role_start_end, MAX_LENGTH)
        role_start_end_lists.append(role_start_end)

    for role_start_end in role_start_end_lists:
        #print(role_start_end)
        label_matrix = np.zeros([MAX_LENGTH, MAX_LENGTH])
        label_matrix[:,:] = -1
        for span in list(itertools.combinations_with_replacement(np.arange(MAX_LENGTH), 2)):
            if (span[1] - span[0]) < MAX_ARGUMENT_SEQUENCE_LENGTH:
                label_matrix[span[0], span[1]] = lab2id['N']
        
        # Fragment Label or O Label. Arg Label のスパン とかぶっても後から上書きされる．
        for c,(role, start, end) in enumerate(role_start_end):  # 文章のトークン数は256以下
            inside_spans = list(itertools.combinations_with_replacement(np.arange(start, end+1), 2))
            for i,j in inside_spans:
                if(i < MAX_LENGTH) and (j < MAX_LENGTH):    # MAX_LENGTH が252未満の場合範囲を越さないための条件
                    if (label_matrix[i, j] != -1):    # 行列の-1の部分には正解ラベルを割り当てない．
                        if role == 'O':
                            role2 = 'O'
                        elif role == 'V':
                            role2 = 'F-P'
                        else:
                            role2 = 'F-A'
                        label_matrix[i,j] = lab2id[role2]

        # Arg Label
        for c,(role, start, end) in enumerate(role_start_end):
            if ((end - start) <= MAX_ARGUMENT_SEQUENCE_LENGTH) and (end < MAX_LENGTH) and (start < MAX_LENGTH):# MAX_LENGTH が243未満の場合範囲を越さないための条件
                label_matrix[start, end] = lab2id[role]
        
        gold_standerd_labels.append(label_matrix)

    # カテゴリーID列をDFに追加
    data['label_id'] = gold_standerd_labels
    data['span_answers'] = role_start_end_lists

    # トークン数情報
    data['num_of_tokens'] = data['sentence'].map(lambda x: len(x.split(' ')))
    
    # fid情報
    data['fid_candidates'] = data['predicate'].map(lambda x: [fid2id[ele] for ele in x['candidate']])
    data['frameID'] = data['predicate'].map(lambda x:fid2id[x['frameID']])

    # 学習データ，テストデータ作成．
    train_df, test_valid_df = train_test_split(data, test_size=0.2, random_state=0)
    test_df, valid_df = train_test_split(test_valid_df, test_size=0.5, random_state=0)

    train_df.reset_index(drop=False, inplace=True)
    test_df.reset_index(drop=False, inplace=True)
    valid_df.reset_index(drop=False, inplace=True)
        
    # form data 
    try :
        train_df = train_df.loc[:, ['sentence', 'predicate', 'label_id', 'directed_graph_matrix', 'span_answers', 'num_of_tokens', 'args', 'fid_candidates', 'frameID']]
        test_df = test_df.loc[:, ['sentence', 'predicate', 'label_id', 'directed_graph_matrix', 'span_answers', 'num_of_tokens', 'args', 'fid_candidates', 'frameID']]
        valid_df = valid_df.loc[:, ['sentence', 'predicate', 'label_id', 'directed_graph_matrix', 'span_answers', 'num_of_tokens', 'args', 'fid_candidates', 'frameID']]
    except:
        train_df = train_df.loc[:, ['sentence', 'predicate', 'label_id', 'span_answers', 'num_of_tokens','args', 'fid_candidates', 'frameID']]
        test_df = test_df.loc[:, ['sentence', 'predicate', 'label_id', 'span_answers', 'num_of_tokens', 'args', 'fid_candidates', 'frameID']]
        valid_df = valid_df.loc[:, ['sentence', 'predicate', 'label_id', 'span_answers', 'num_of_tokens', 'args', 'fid_candidates', 'frameID']]

    return train_df, test_df, valid_df