import pandas as pd
import numpy as np
import itertools
from sklearn.model_selection import train_test_split

def find_combination_index(MAX_LEN, target):
    assert target[0] <= target[1], print('start should be lower than end')
    #span_available_indication = np.zeros([MAX_LENGTH, MAX_LENGTH])
    #span_available_indication[:,:] = -1
    #for span in list(itertools.combinations_with_replacement(np.arange(n), 2)):
    #    if (span[1] - span[0]) <= 30:
    #        span_available_indication[span[0], span[1]] = 1
    for index, combination in enumerate(itertools.product(range(MAX_LEN), repeat=2)):
        if combination == target:
            return index
    return -1
    
def get_train_test(MAX_LENGTH, data, lab2id):
    # 正解ラベル作成
    features = []
    span_answers=[]
    for args, pred, sentence in zip(data['args'].tolist(), data['predicate'].tolist(), data['sentence'].tolist()):
        arg_info=[[role, pred['word_start'], pred['word_end'], find_combination_index(MAX_LENGTH, (int(pred['word_start']), int(pred['word_end'])))] for role in lab2id.values()]
        span_answer=[]
        for arg in args:
            arg_info[lab2id[arg['argrole']]] = [lab2id[arg['argrole']], int(arg['word_start']), int(arg['word_end']), find_combination_index(MAX_LENGTH, (int(arg['word_start']), int(arg['word_end']))) ]
            span_answer.append([arg['argrole'], arg['word_start'], arg['word_end'], find_combination_index(MAX_LENGTH, (int(arg['word_start']), int(arg['word_end']))) ])
        features.append([arg_info, pred, sentence, len(sentence.split())])
        span_answers.append(span_answer)
    # 学習データ，テストデータ作成．
    data2 = pd.DataFrame(features, columns=['args', 'predicate', 'sentence', 'num_of_tokens'])
    data2['span_answers'] = span_answers
    #print(data2['span_answers'])
    #print(data2['args'])
    train_df, test_valid_df = train_test_split(data2, test_size=0.2, random_state=0)
    test_df, valid_df = train_test_split(test_valid_df, test_size=0.5, random_state=0)
    print('train, test, valid = ', len(train_df),len(test_df),len(valid_df))
    
    return train_df, test_df, valid_df