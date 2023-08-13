import sys
import datetime
import itertools
import subprocess
sys.path.append('../')
sys.path.append('../../')
sys.path.append('../../../')

import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR

from transformers import AutoTokenizer
from preprocess.base.mk_dataset import *
from preprocess.base.netwroks import BertClassifierDecodeIntegrated2
from preprocess.base.prepare_data import *
from utils.tools import filter_span_score_for_batch_1
from Evaluation.evaluate import cal_label_f1
from Evaluation.evaluate import cal_label_f1


DATAPATH = '../../../Data/common_data_v2_bert.json'
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
MAX_LENGTH = 254
MAX_ARGUMENT_SEQUENCE_LENGTH = 30                       # 項の最高トークン数．（これより大きいものは予測不可能）
MAX_TOKEN = MAX_LENGTH + PRED_SEP_NUM + 2         # BERT に入力するトークン数．+2 は BOSとEOS 分のトークン． 
BATCH_SIZE = 16
MODEL_NAME = 'bert_dif'
print(f'MAX_TOKEN = {MAX_TOKEN}, MAX_LENGTH = {MAX_LENGTH}, MAX_ARGUMENT_SEQUENCE_LENGTH = {MAX_ARGUMENT_SEQUENCE_LENGTH}\n\n')

def decode(dataset, data_df):
    predictions = []
    core_labels = ['Arg','Arg0','Arg1', 'Arg2', 'Arg3', 'Arg4', 'Arg5']
    with torch.no_grad():
        classifier.eval()
        for i, (batch_features, batch_preds, batch_token) in enumerate(dataset):
            # BATCH サイズは１
            token_num = batch_token.item()
            preds_span = [batch_preds[0][0].item(), batch_preds[0][1].item()]

            # 使用するスパンの範囲を教える変数（使用可：１， 使用不可：－１）
            span_available_indication = np.zeros([token_num, token_num])
            span_available_indication[:,:] = 0
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
            out = classifier(input_ids, pred_span, token_num, span_available_indication, 'decode') # out[MAX*MAX][batch][labels]
            out = [tmp.to('cpu').detach().numpy().copy() for tmp in out] # cpu に移し，numpyに変換するためにdetach. 
            out = np.array(out).transpose(1, 0, 2) # out[batch][MAX*MAX][labels]

            # prepare input for decode
            for scores, pred_span, token_num in zip(out, batch_preds, batch_token):   # batchから取り出し， 各文章毎に行う．
                span_score_lists, used_cores, confirmed_spans = [], [], []
                # 各スパンの各ラベルごとのscoreを入れていく
                token_num = MAX_LENGTH if token_num > MAX_LENGTH else token_num
                span_possible_tuples = [(i, j) for i in range(token_num) for j in range(token_num)]
                assert len(span_possible_tuples) == scores.shape[0], print(len(span_possible_tuples), scores.shape[0])
                for span_idx, scores_of_a_span in zip(span_possible_tuples, scores):
                    if span_available_indication[span_idx[0], span_idx[1]] != 0:
                        for label, score in zip( list(ID2LAB.values()), scores_of_a_span ):
                            span_score_lists.append([span_idx[0], span_idx[1], label, score])

                # filterにかけた後，貪欲法で値の大きいものから取っていく
                prev_num = len(span_score_lists)
                span_score_lists = filter_span_score_for_batch_1(span_score_lists, pred_span, token_num)
                #print(f'{i} : ',prev_num, ' -> ', len(span_score_lists), f'token_num = {token_num}')

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
    span_answers = data_df['span_answers'].to_list()
    #for a,b in zip(span_answers, predictions):
    #    print(a)
    #    print(b)
    #    print()
    report = cal_label_f1(predictions, span_answers, LAB2ID)
    f1 = report.loc['f1','correct_num'] if not np.isnan(report.loc['f1','correct_num']) else 0
    return f1, report



if __name__ == "__main__":
    # 各種データ作成（学習，テスト，検証）
    DATA = DATA.sample(frac=1, random_state=0).reset_index(drop=True)
    tokenizer = AutoTokenizer.from_pretrained("rinna/japanese-gpt-neox-3.6b", use_fast=False)
    train_df, test_df, valid_df = get_train_test(MAX_LENGTH, MAX_ARGUMENT_SEQUENCE_LENGTH, DATA, LAB2ID)
    #test_df = test_df.sample(frac=1)

    print(len(train_df))
    loss_function = nn.NLLLoss()    # 損失関数の設定
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # GPUの設定


    classifier = BertClassifierDecodeIntegrated2(OUTPUT_LAYER_DIM, MAX_LENGTH, device).to(device)
    # まずは全部OFF
    for param in classifier.parameters():
        param.requires_grad = False

    # BERTの最終4層分をON
    for param in classifier.bert.encoder.layer[-1].parameters():
        param.requires_grad = True
    for param in classifier.bert.encoder.layer[-2].parameters():
        param.requires_grad = True
    for param in classifier.bert.encoder.layer[-3].parameters():
        param.requires_grad = True
    for param in classifier.bert.encoder.layer[-4].parameters():
        param.requires_grad = True

    # 追加した層のところもON
    for param in classifier.linear.parameters():
        param.requires_grad = True
    for param in classifier.linear2.parameters():
        param.requires_grad = True

    # 事前学習済の箇所は学習率小さめ、最後の全結合層は大きめにする。
    ENC_NUM = 4
    optimizer = optim.Adam([
        {'params': classifier.bert.encoder.layer[-1].parameters(), 'lr': 5e-5},
        {'params': classifier.bert.encoder.layer[-2].parameters(), 'lr': 5e-5},
        {'params': classifier.bert.encoder.layer[-3].parameters(), 'lr': 5e-5},
        {'params': classifier.bert.encoder.layer[-4].parameters(), 'lr': 5e-5},
        {'params': classifier.linear.parameters(), 'lr': 1e-4},
        {'params': classifier.linear2.parameters(), 'lr': 1e-4}
    ])

    """
    Train
    """
    # dataset, dateloader
    prev_f1, patience_counter = 0, 0
    train_dataset = mk_dataset(train_df, BATCH_SIZE, MAX_LENGTH, MAX_TOKEN, PRED_SEP_NUM, tokenizer, sort=True)
    valid_dataset = mk_dataset_decode(valid_df, 1, MAX_LENGTH, MAX_TOKEN, PRED_SEP_NUM, tokenizer, sort=True)
    test_dataset = mk_dataset_decode(test_df, 1, MAX_LENGTH, MAX_TOKEN, PRED_SEP_NUM, tokenizer, sort=True)
    time_start = datetime.datetime.now()
    # エポック数は5で
    for epoch in range(30):
        all_loss = 0
        classifier.train()
        for i, (batch_features, batch_labels, batch_preds, batch_token) in enumerate(train_dataset):   # labelsはバッチ16こに対し128トークンのラベル
            # 使用するスパンの範囲を教える変数（使用可：１， 使用不可：－１）
            if i%100 == 0:
                print(f'Progress {i*BATCH_SIZE} / {len(train_dataset)*BATCH_SIZE}')
                sys.stdout.flush() # 明示的にflush
            token_num = int(batch_token[-1]) if int(batch_token[-1]) < MAX_LENGTH else MAX_LENGTH   # batch内最大トークン数．
            span_available_indication = torch.ones([MAX_LENGTH, MAX_LENGTH]) * -1
            span_idx=[]
            for span in list(itertools.combinations_with_replacement(np.arange(token_num), 2)):
                if (span[1] - span[0]) <= MAX_ARGUMENT_SEQUENCE_LENGTH:
                    span_available_indication[span[0], span[1]] = 1
            for idx, span in enumerate(list(itertools.product(range(MAX_LENGTH), repeat=2))):
                if span_available_indication[span[0]][span[1]] == 1:
                    span_idx.append(idx)
            # 各特徴，ラベルをデバイスへ送る
            input_ids = batch_features.to(device)         # input token ids
            pred_span = batch_preds.to(device)            # predicate span
            token_num = torch.tensor(token_num,dtype=torch.long).to(device)
            label_ids = batch_labels.view(len(batch_labels), -1)            # batch_labels[batch][MAX_LENGTH][MAX_LENGTH] -> batch_labels[batch][MAX_LENGTH*MAX_LENGTH]
            label_ids = label_ids[:,span_idx].to(device)
            span_available_indication = span_available_indication.to(device)
            
            outs = classifier(input_ids, pred_span, token_num, span_available_indication, 'train') # [batch][max][label]
            batch_loss = loss_function(outs.reshape(-1, len(LABELS)), label_ids.reshape(-1))    # (batch*max,label)  : (batch*max), 各スパンに対して，正解ラベルを用意
            batch_loss.backward()

            all_loss += batch_loss.item()

            optimizer.step()
            classifier.zero_grad()  # 累積されるので，ここで初期化しなくてはならない．
        #scheduler.step()
        print("epoch", epoch, "\t" , "loss", all_loss)
        """
        Validation
        """
        f1, report = decode(valid_dataset, valid_df)
        print(report)
        if prev_f1 < f1:
            patience_counter = 0
            prev_f1 = f1
            print('Valid f1 = ',f1,'\n')
            classifier.save_pretrained(f"../../models/{MODEL_NAME}/")
            continue

        elif (prev_f1 >= f1) and (patience_counter < 2):   # 10回連続でf1が下がらなければ終了
            print('No change in valid f1\n') 
            if prev_f1 != 0:
                patience_counter += 1     
            continue
        else: 
            print('Stop. No change in valid f1\n') 
            break

    # 時間計測
    time_end = datetime.datetime.now()
    print('\n', time_end - time_start, '\n')

    """
    Test
    """
    classifier = BertClassifierDecodeIntegrated2(OUTPUT_LAYER_DIM, MAX_LENGTH, device)
    classifier.load_state_dict(torch.load(f"../../models/{MODEL_NAME}/")).to(device)

    print(decode(test_dataset, test_df))