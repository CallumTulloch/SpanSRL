import sys
import datetime
import itertools
import subprocess
sys.path.append('../')
sys.path.append('../../')
sys.path.append('../../../')

import time
import pandas as pd
import numpy as np
pd.set_option('display.max_rows', 990)
pd.set_option('display.max_columns', 990)
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from transformers import AutoTokenizer

from preprocess.base.mk_dataset import *
from preprocess.base.netwroks import BertClassifierDecode
from preprocess.base.prepare_data import *
from Evaluation.evaluate import cal_accuracy
from Evaluation.evaluate import cal_label_f1
from Evaluation.evaluate import cal_span_f1
from Evaluation.evaluate import span2seq
from Evaluation.evaluate import get_pred_dic_lists
from utils.tools import filter_span_score_for_batch_1

# NLPIR
# DATAPATH = '../../../Data/data_v2.json'
# 
# with open(DATAPATH, 'r', encoding="utf-8_sig") as json_file:
#     DATA = pd.read_json(json_file)

# EMNLP
DATAPATH = '../../../../Ohuchi_old/Data/data_v2_under53.json'
EXDATAPATH = '../../../../Ohuchi_old/Data/data_v2_extra2.json'

with open(DATAPATH, 'r', encoding="utf-8_sig") as json_file:
    DATA = pd.read_json(json_file)
    
with open(EXDATAPATH, 'r', encoding="utf-8_sig") as json_file:
    EXDATA = pd.read_json(json_file)

    
# 正解ラベル
labels = []
for args in DATA['args']:
    labels += [ arg['argrole'] for arg in args]
labels = set(labels)
LABELS = sorted(list(labels)) + ['F-A', 'F-P', 'V', 'O', 'N']

# カテゴリーのID辞書を作成
ID2LAB = dict(zip(list(range(len(LABELS))), LABELS))
LAB2ID = dict(zip(LABELS, list(range(len(LABELS)))))
print(LAB2ID, '\n')

# 各種定義
OUTPUT_LAYER_DIM = len(LABELS)                              # 全ラベルの数
PRED_SEP_NUM = 10 + 2                              # 述語情報のためのトークン数．sep:2, pred:8(最長)
MAX_LENGTH = 252                           # ラベル割り当てするトークン数. 文章の最長は 243 token
MAX_ARGUMENT_SEQUENCE_LENGTH = 30         # 項の最高トークン数．（これより大きいものは予測不可能）
MAX_TOKEN = MAX_LENGTH + PRED_SEP_NUM + 1         # BERT に入力するトークン数．+1 は cls 分のトークン． 
BATCH_SIZE = 1

if __name__ == "__main__":
    DATA = DATA.sample(frac=1, random_state=0).reset_index(drop=True)
    tokenizer = AutoTokenizer.from_pretrained("cl-tohoku/bert-base-japanese-v2")
    train_df, test_df, valid_df = get_train_test_decode(MAX_LENGTH, MAX_ARGUMENT_SEQUENCE_LENGTH, DATA, LAB2ID)
    
    # EMNLP
    EXDATA = EXDATA.sample(frac=1, random_state=0).reset_index(drop=True)
    extrain_df, extest_df, exvalid_df = get_train_test_decode(MAX_LENGTH, MAX_ARGUMENT_SEQUENCE_LENGTH, EXDATA, LAB2ID)
    test_df=pd.concat([test_df,extest_df],axis=0)
    print(len(test_df))
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # GPUの設定
    classifier = BertClassifierDecode(OUTPUT_LAYER_DIM, MAX_LENGTH, device).to(device)
    
    # dataset, dateloader
    dataset = mk_dataset_decode(test_df, BATCH_SIZE, MAX_LENGTH, MAX_TOKEN, PRED_SEP_NUM, tokenizer)
    classifier.load_state_dict(torch.load(f'../../models/srl_base_252_enc4_best.pth'))

    """
    Decode Area
    """
    index_for_correspondence = 0
    predictions = []
    core_labels = ['Arg','Arg0','Arg1', 'Arg2', 'Arg3', 'Arg4', 'Arg5']
    start = time.time()

    with torch.no_grad():
        for i, (batch_features, batch_preds, batch_token) in enumerate(dataset):
            # BATCH サイズは１
            token_num = batch_token.item()
            preds_span = [batch_preds[0][0].item(), batch_preds[0][1].item()]

            # 使用するスパンの範囲を教える変数（使用可：１， 使用不可：－１）
            span_available_indication = np.zeros([token_num, token_num])
            span_available_indication[:,:] = -1
            num_of_span_vecs = torch.tensor(0)
            for span in list(itertools.combinations_with_replacement(np.arange(token_num), 2)):
                if (span[1] - span[0]) <= MAX_ARGUMENT_SEQUENCE_LENGTH:
                    span_available_indication[span[0], span[1]] = 1
                    num_of_span_vecs += 1

            # スパンの範囲が網羅されているか確認するための変数
            span_element = set()                                                     # 推測しているスパンの要素
            input_span_element = set( torch.arange(0, token_num+1) )                 # 入力のスパン要素
            pred_span_element = set( torch.arange(preds_span[0], preds_span[1]+1) )  # 述語のスパン要素
            span_element_for_comaparison = input_span_element - pred_span_element    # 予測と入力の比較するための要素

            # 各特徴，ラベルをデバイスへ送る.
            input_ids = batch_features.to(device)         # input token ids
            pred_span = batch_preds.to(device)            # predicate span

            # get output
            out = classifier(input_ids, pred_span, token_num, span_available_indication) # out[MAX*MAX][batch][labels]
            out = [tmp.to('cpu').detach().numpy().copy() for tmp in out] # cpu に移し，numpyに変換するためにdetach. 
            out = np.array(out).transpose(1, 0, 2) # out[batch][MAX*MAX][labels]

            # prepare input for decode
            for scores, pred_span, token_num in zip(out, batch_preds, batch_token):   # batchから取り出し， 各文章毎に行う．
                span_score_lists, used_cores, confirmed_spans = [], [], []
                # 各スパンの各ラベルごとのscoreを入れていく
                token_num = MAX_LENGTH if token_num > MAX_LENGTH else token_num
                span_possible_tuples = [(i, j) for i in range(token_num) for j in range(token_num)]
                null_label_scores = scores[:, -1].reshape(token_num, token_num)
                reshaped_scores = scores.reshape(token_num, token_num, -1)
                pred_span_scores = reshaped_scores[pred_span[0], pred_span[1], :]
                for span_idx, scores_of_a_span in zip(span_possible_tuples, scores):
                    if span_available_indication[span_idx[0], span_idx[1]] != -1:
                        for label, score in zip( list(ID2LAB.values()), scores_of_a_span ):
                            span_score_lists.append([span_idx[0], span_idx[1], label, score])

                # filterにかけた後，貪欲法で値の大きいものから取っていく
                prev_num = len(span_score_lists)
                span_score_lists = filter_span_score_for_batch_1(span_score_lists, pred_span, token_num)
                #span_score_lists = filter_span_score_for_batch_1_addnullspan(span_score_lists, pred_span, token_num, null_label_scores, pred_span_scores, lab2id)
                print(f'{i} : ',prev_num, ' -> ', len(span_score_lists), f'token_num = {token_num}')

                for span_score in sorted(span_score_lists, key=lambda x:x[3], reverse=True):
                    target_span_element = set( np.arange(span_score[0], span_score[1]+1) )
                    overlap_flag = not target_span_element.isdisjoint(span_element)
                    # spanを追加する条件 
                    if (overlap_flag == False) and span_score[2] not in used_cores:
                        confirmed_spans.append(span_score)
                        span_element = span_element | target_span_element
                        if span_score[2] in core_labels:  # core_label は一度しか使ってはならない
                            used_cores.append(span_score[2])
                    
                    # span が入力文のスパンを網羅しているか
                    if span_element_for_comaparison == span_element:
                        break
                #print('confirmed_spans = ', sorted(confirmed_spans, key=lambda x:x[0]), '\n')
                predictions.append(sorted(confirmed_spans, key=lambda x:x[0]))
    print('----------------------------------------')
    #print(predictions)
    span_answers = test_df['span_answers'].to_list()
    lf1 = cal_label_f1(predictions, span_answers, LAB2ID)
    print(lf1)
    lf1.to_csv('span1_label.csv')
    sf1 = cal_span_f1(predictions, span_answers, LAB2ID, MAX_LENGTH)
    print(sf1)
    sf1.to_csv('span1_span.csv')
    
    # 解析用データ作成
    pred_seq = [span2seq(p, int(num_of_tokens)) for p, num_of_tokens in zip(predictions, test_df['num_of_tokens'].to_list())]
    test_df['BIOtag'] = pred_seq
    
    pred_dic_lists, match_count_list, args_count_list = get_pred_dic_lists(predictions, span_answers, LAB2ID)
    test_df['pred_arg'] = pred_dic_lists
    test_df['match_count'] = match_count_list
    test_df['args_num'] = args_count_list
    test_df['predict_num'] = [len(dic) for dic in pred_dic_lists]
    test_df = test_df[['sentence', 'sentenceID', 'predicate','args', 'BIOtag', 'pred_arg', 'match_count', 'args_num', 'predict_num']]
    test_df.to_json('data_for_analy.json',orient='records', lines=True,force_ascii=False)
    
    print(f"Time : {time.time() - start}")
