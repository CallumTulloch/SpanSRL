import pandas as pd
import numpy as np
import itertools
from sklearn.model_selection import train_test_split
import collections
from itertools import combinations, product
from functools import reduce

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

def multiply_list_elements(my_list):
    result = reduce((lambda x, y: x * y), my_list)
    return result

def generate_combinations(data):
    # Transform data into a DataFrame for easy manipulation
    df = pd.DataFrame(data)
    
    # Group data by 'argrole'
    grouped = df.groupby('argrole')
    
    # Create combinations for each 'argrole' group
    combination_dict = {}
    for argrole, group in grouped:
        combination_dict[argrole] = []
        for r in range(1, len(group)+1):
            combination_dict[argrole].extend(list(combinations(group.to_dict('records'), r)))
    
    # Combine the combinations from different 'argrole' groups
    all_combinations = list(product(*combination_dict.values()))

    # Flatten the nested tuples and convert dict to tuple to make it hashable, then remove duplicates
    all_combinations = [
        sorted(set([tuple(arg.items()) for arg in sum(comb, ())]), key=lambda x: dict(x)['word_start']) 
        for comb in all_combinations
    ]

    # Convert back to list of dicts
    all_combinations = [[dict(arg) for arg in comb] for comb in all_combinations]
    
    return all_combinations

# valid と test はデコードを使用するため，データ拡張は必要ない．（逆に行うと不正確）
def get_train_test(MAX_LENGTH, data, lab2id):
    # 正解ラベル作成
    train_df, test_valid_df = train_test_split(data, test_size=0.2, random_state=0)
    test_df, valid_df = train_test_split(test_valid_df, test_size=0.5, random_state=0)
    """
    train
    """
    features = []
    span_answers=[]
    for args, pred, sentence in zip(train_df['args'].tolist(), train_df['predicate'].tolist(), train_df['sentence'].tolist()):
        arg_info=[[role, pred['word_start'], pred['word_end'], find_combination_index(MAX_LENGTH, (int(pred['word_start']), int(pred['word_end'])))] for role in lab2id.values()]
        span_answer=[]
        arg_list=[]
        for arg in args: 
            arg_list.append(arg['argrole'])
        args_combi = generate_combinations(args)
        for args in args_combi:
            for arg in args:
                arg_info[lab2id[arg['argrole']]] = [lab2id[arg['argrole']], int(arg['word_start']), int(arg['word_end']), find_combination_index(MAX_LENGTH, (int(arg['word_start']), int(arg['word_end']))) ]
                span_answer.append([arg['argrole'], arg['word_start'], arg['word_end'], find_combination_index(MAX_LENGTH, (int(arg['word_start']), int(arg['word_end']))) ])
            features.append([arg_info, pred, sentence, len(sentence.split()), train_df['args'].tolist()[0]])
            span_answers.append(span_answer)
    # 学習データ，テストデータ作成．
    train_df2 = pd.DataFrame(features, columns=['args', 'predicate', 'sentence', 'num_of_tokens', 'original_args_info'])
    train_df2['span_answers'] = span_answers
    """
    test
    """
    features = []
    span_answers=[]
    for args, pred, sentence, sentenceid in zip(test_df['args'].tolist(), test_df['predicate'].tolist(), test_df['sentence'].tolist(), test_df['sentenceID'].tolist()):
        arg_info=[[role, pred['word_start'], pred['word_end'], find_combination_index(MAX_LENGTH, (int(pred['word_start']), int(pred['word_end'])))] for role in lab2id.values()]
        span_answer=[]
        for arg in args:
            arg_info[lab2id[arg['argrole']]] = [lab2id[arg['argrole']], int(arg['word_start']), int(arg['word_end']), find_combination_index(MAX_LENGTH, (int(arg['word_start']), int(arg['word_end']))) ]
            span_answer.append([arg['argrole'], arg['word_start'], arg['word_end'], find_combination_index(MAX_LENGTH, (int(arg['word_start']), int(arg['word_end']))) ])
        features.append([arg_info, pred, sentence, len(sentence.split()), sentenceid])
        span_answers.append(span_answer)
    # 学習データ，テストデータ作成．
    test_df2 = pd.DataFrame(features, columns=['args', 'predicate', 'sentence', 'num_of_tokens', 'sentenceID'])
    test_df2['span_answers'] = span_answers
    
    """
    valid
    """
    features = []
    span_answers=[]
    for args, pred, sentence, sentenceid in zip(valid_df['args'].tolist(), valid_df['predicate'].tolist(), valid_df['sentence'].tolist(), valid_df['sentenceID'].tolist()):
        arg_info=[[role, pred['word_start'], pred['word_end'], find_combination_index(MAX_LENGTH, (int(pred['word_start']), int(pred['word_end'])))] for role in lab2id.values()]
        span_answer=[]
        for arg in args:
            arg_info[lab2id[arg['argrole']]] = [lab2id[arg['argrole']], int(arg['word_start']), int(arg['word_end']), find_combination_index(MAX_LENGTH, (int(arg['word_start']), int(arg['word_end']))) ]
            span_answer.append([arg['argrole'], arg['word_start'], arg['word_end'], find_combination_index(MAX_LENGTH, (int(arg['word_start']), int(arg['word_end']))) ])
        features.append([arg_info, pred, sentence, len(sentence.split()), sentenceid])
        span_answers.append(span_answer)
    # 学習データ，テストデータ作成．
    valid_df2 = pd.DataFrame(features, columns=['args', 'predicate', 'sentence', 'num_of_tokens', 'sentenceID'])
    valid_df2['span_answers'] = span_answers
    print('train, test, valid = ', len(train_df),len(test_df),len(valid_df))
    print('train, test, valid = ', len(train_df2),len(test_df2),len(valid_df2))
    
    return train_df2, test_df2, valid_df2


# valid と test はデコードを使用するため，データ拡張は必要ない．（逆に行うと不正確）
def get_train_test_decode(MAX_LENGTH, data, lab2id):
    # 正解ラベル作成
    train_df, test_valid_df = train_test_split(data, test_size=0.2, random_state=0)
    test_df, valid_df = train_test_split(test_valid_df, test_size=0.5, random_state=0)

    """
    test
    """
    #features = []
    #span_answers=[]
    #for args, pred, sentence, sentenceid in zip(test_df['args'].tolist(), test_df['predicate'].tolist(), test_df['sentence'].tolist(), test_df['sentenceID'].tolist()):
    #    arg_info=[[role, pred['word_start'], pred['word_end'], find_combination_index(MAX_LENGTH, (int(pred['word_start']), int(pred['word_end'])))] for role in lab2id.values()]
    #    span_answer=[]
    #    for arg in args:
    #        arg_info[lab2id[arg['argrole']]] = [lab2id[arg['argrole']], int(arg['word_start']), int(arg['word_end']), find_combination_index(MAX_LENGTH, (int(arg['word_start']), int(arg['word_end']))) ]
    #        span_answer.append([arg['argrole'], arg['word_start'], arg['word_end'], find_combination_index(MAX_LENGTH, (int(arg['word_start']), int(arg['word_end']))) ])
    #    features.append([arg_info, pred, sentence, len(sentence.split()), sentenceid, args])
    #    span_answers.append(span_answer)
    ## 学習データ，テストデータ作成．
    #test_df2 = pd.DataFrame(features, columns=['args', 'predicate', 'sentence', 'num_of_tokens', 'sentenceID', 'args2'])
    #test_df2['span_answers'] = span_answers
    
    """
    valid
    """
    features = []
    span_answers=[]
    for args, pred, sentence, sentenceid in zip(valid_df['args'].tolist(), valid_df['predicate'].tolist(), valid_df['sentence'].tolist(), valid_df['sentenceID'].tolist()):
        arg_info=[[role, pred['word_start'], pred['word_end'], find_combination_index(MAX_LENGTH, (int(pred['word_start']), int(pred['word_end'])))] for role in lab2id.values()]
        span_answer=[]
        for arg in args:
            arg_info[lab2id[arg['argrole']]] = [lab2id[arg['argrole']], int(arg['word_start']), int(arg['word_end']), find_combination_index(MAX_LENGTH, (int(arg['word_start']), int(arg['word_end']))) ]
            span_answer.append([arg['argrole'], arg['word_start'], arg['word_end'], find_combination_index(MAX_LENGTH, (int(arg['word_start']), int(arg['word_end']))) ])
        features.append([arg_info, pred, sentence, len(sentence.split()), sentenceid, args])
        span_answers.append(span_answer)
    # 学習データ，テストデータ作成．
    valid_df2 = pd.DataFrame(features, columns=['args', 'predicate', 'sentence', 'num_of_tokens', 'sentenceID', 'args2'])
    valid_df2['span_answers'] = span_answers

    return valid_df2