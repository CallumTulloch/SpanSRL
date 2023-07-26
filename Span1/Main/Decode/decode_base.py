import sys
import datetime
import itertools
import subprocess
sys.path.append('../')
sys.path.append('../../')

import time
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from transformers import AutoTokenizer
from preprocess.base.mk_dataset import *
from preprocess.base.netwroks import BertClassifierDecode
from preprocess.base.prepare_data import *

from utils.evaluate import cal_accuracy
from utils.evaluate import cal_label_f1
from utils.evaluate import cal_span_f1
from utils.tools import filter_span_score_for_batch_1


print(sys.argv)
DATAPATH = sys.argv[4]
# read data
with open(DATAPATH, 'r', encoding="utf-8_sig") as json_file:
    DATA = pd.read_json(json_file)
    
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
MAX_LENGTH = int(sys.argv[1])                           # ラベル割り当てするトークン数. 文章の最長は 243 token
MAX_ARGUMENT_SEQUENCE_LENGTH = int(sys.argv[2])         # 項の最高トークン数．（これより大きいものは予測不可能）
MAX_TOKEN = MAX_LENGTH + PRED_SEP_NUM + 1         # BERT に入力するトークン数．+1 は cls 分のトークン． 
BATCH_SIZE = 1
MODEL_NAME = sys.argv[6]


if __name__ == "__main__":
    print(sys.argv)
    DATA = DATA.sample(frac=1, random_state=0).reset_index(drop=True)
    tokenizer = AutoTokenizer.from_pretrained("cl-tohoku/bert-base-japanese-v2")
    train_df, test_df, valid_df = get_train_test(MAX_LENGTH, MAX_ARGUMENT_SEQUENCE_LENGTH, DATA, LAB2ID)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # GPUの設定
    test_df = test_df.sample(n=100, random_state=0).reset_index(drop=True)
    classifier = BertClassifierDecode(OUTPUT_LAYER_DIM, MAX_LENGTH, device).to(device)
    # dataset, dateloader
    if sys.argv[5] == 'valid':
        dataset = mk_dataset_decode(valid_df, BATCH_SIZE, MAX_LENGTH, MAX_TOKEN, PRED_SEP_NUM, tokenizer)
        print(len(valid_df))
        # モデルをロード
        ENC_NUM = int(sys.argv[3])
        classifier.load_state_dict(torch.load(f'../../models/srl_{MODEL_NAME}_eachEP.pth'))
        
    if sys.argv[5] == 'test':
        dataset = mk_dataset_decode(test_df, BATCH_SIZE, MAX_LENGTH, MAX_TOKEN, PRED_SEP_NUM, tokenizer)
        print(len(test_df))
        # モデルをロード
        ENC_NUM = int(sys.argv[3])
        classifier.load_state_dict(torch.load(f'../../models/srl_{MODEL_NAME}_best.pth'))

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
    if sys.argv[5] == 'valid':
        span_answers = valid_df['span_answers'].to_list()
        lf1 = cal_label_f1(predictions, span_answers, LAB2ID)
        lf1.to_csv(f"../../report/Validation/results_{MODEL_NAME}.csv", encoding='utf-8_sig')
        #fid_report.to_csv(f"../../report/Validation/results_fid_{MAX_LENGTH}_enc{ENC_NUM}_multi.csv", encoding='utf-8_sig')
    if sys.argv[5] == 'test':
        span_answers = test_df['span_answers'].to_list()
        lf1 = cal_label_f1(predictions, span_answers, LAB2ID)
        lf1.to_csv(f"../../report/Test/results_{MODEL_NAME}.csv", encoding='utf-8_sig')
        #fid_report.to_csv(f"../../report/Test/results_fid_{MAX_LENGTH}_enc{ENC_NUM}_multi.csv", encoding='utf-8_sig')
    print(lf1)
    print(f"Time : {time.time() - start}")
    #report.to_csv("../../report/enc4_incO_soft_soft_243.csv", encoding='utf-8_sig')
