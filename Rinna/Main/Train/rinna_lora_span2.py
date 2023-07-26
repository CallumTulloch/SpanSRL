import sys
import datetime
import itertools
import subprocess
sys.path.append('../')
sys.path.append('../../')

import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from peft import PeftModel
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
)

from transformers import AutoTokenizer
from preprocess.span2.mk_dataset import *
from preprocess.span2.netwroks import RinnaClassifierSpan2DecodeIntegrated
from preprocess.span2.prepare_data import *

from utils.evaluate import cal_label_f1
from utils.tools_span2 import filter_span_score


DATAPATH = '../../common_data_v2_rinna.json'

# read data
with open(DATAPATH, 'r', encoding="utf-8_sig") as json_file:
    DATA = pd.read_json(json_file)
    
# 正解ラベル（カテゴリー）をデータセットから取得
labels = []
for args in DATA['args']:
    labels += [ arg['argrole'] for arg in args]
labels = set(labels)
LABELS = sorted(list(labels))

# カテゴリーのID辞書を作成，出力層の数定義
ID2LAB = dict(zip(list(range(len(LABELS))), LABELS))
LAB2ID = dict(zip(LABELS, list(range(len(LABELS)))))
print(LAB2ID, '\n')

# 各種定義
OUTPUT_LAYER_DIM = len(LABELS)                              # 全ラベルの数
PRED_SEP_NUM = 10 + 2                                       # 述語情報のためのトークン数．sep:2, pred:8(最長)
MAX_LENGTH = 192
MAX_ARGUMENT_SEQUENCE_LENGTH = 30                           # 項の最高トークン数．（これより大きいものは予測不可能）
MAX_TOKEN = MAX_LENGTH + PRED_SEP_NUM + 2                   # BERT に入力するトークン数．+1 は cls 分のトークン． 
BATCH_SIZE = 32
SPAN_AVAILABLE_INDICATION = np.zeros([MAX_LENGTH, MAX_LENGTH])
SPAN_AVAILABLE_INDICATION[:,:] = -1
num_of_span_vec = 0
for span in list(itertools.combinations_with_replacement(np.arange(MAX_LENGTH), 2)):
    if (span[1] - span[0]) <= MAX_ARGUMENT_SEQUENCE_LENGTH:
        SPAN_AVAILABLE_INDICATION[span[0], span[1]] = 1
        num_of_span_vec += 1
NUM_OF_SPAN_VEC = num_of_span_vec
MODEL_NAME = 'rinna_lora_span2'
print(f'MAX_TOKEN = {MAX_TOKEN}, MAX_LENGTH = {MAX_LENGTH}, MAX_ARGUMENT_SEQUENCE_LENGTH = {MAX_ARGUMENT_SEQUENCE_LENGTH}\n\n')

def decode(dataset, data_df):
    predictions = []
    core_labels = ['Arg','Arg0','Arg1', 'Arg2', 'Arg3', 'Arg4', 'Arg5']
    with torch.no_grad():
        classifier.eval()
        for c, (batch_features, batch_preds, batch_token_num, batch_args) in enumerate(dataset):   # labelsはバッチ16こに対し128トークンのラベル
            # BATCH サイズは１
            token_num = batch_token_num[0].item()
            preds_span = [batch_preds[0][0].item(), batch_preds[0][1].item()]

            # 使用するスパンの範囲を教える変数（使用可：１， 使用不可：－１）
            span_available_indication = np.zeros([MAX_LENGTH, MAX_LENGTH])
            span_available_indication[:,:] = -1
            token_num = token_num if MAX_LENGTH>=token_num else MAX_LENGTH
            for span in list(itertools.combinations_with_replacement(np.arange(token_num), 2)):
                if (span[1] - span[0]) <= MAX_ARGUMENT_SEQUENCE_LENGTH:
                    span_available_indication[span[0]][span[1]] = 1
            #print(np.count_nonzero((span_available_indication >= 1)))

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
            #out = [tmp.to('cpu').detach().numpy().copy() for tmp in out] # cpu に移し，numpyに変換するためにdetach. 
            #out = np.array(out).transpose(1, 0, 2) # out[batch][MAX*MAX][labels]

            # prepare input for decode
            for scores, pred_span, token_num in zip(out, batch_preds, batch_token_num):   # batchから取り出し， 各文章毎に行う．
                span_score_lists, used_cores, confirmed_spans = [], [], []
                # 各スパンの各ラベルごとのscoreを入れていく
                token_num = MAX_LENGTH if token_num > MAX_LENGTH else token_num
                span_possible_tuples = [(i, j) for i in range(token_num) for j in range(token_num)]
                reshaped_scores = scores.reshape(MAX_LENGTH, MAX_LENGTH, -1)
                pred_span_scores = reshaped_scores[pred_span[0], pred_span[1], :]
                for i,j in span_possible_tuples:
                    if span_available_indication[i][j] != -1:
                        for label, score in zip( list(ID2LAB.values()), reshaped_scores[i][j] ):
                            span_score_lists.append([i, j, label, score])

                # filterにかけた後，貪欲法で値の大きいものから取っていく
                prev_num = len(span_score_lists)
                #span_score_lists = filter_span_score_for_batch_1(span_score_lists, pred_span, token_num)
                span_score_lists = filter_span_score(span_score_lists, pred_span, token_num, pred_span_scores, LAB2ID)
                print(f'{c} : ',prev_num, ' -> ', len(span_score_lists), f'token_num = {token_num}')

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
    report = cal_label_f1(predictions, span_answers, LAB2ID)
    f1 = report.loc['f1','correct_num'] if not np.isnan(report.loc['f1','correct_num']) else 0
    return f1, report



if __name__ == "__main__":
    # 各種データ作成（学習，テスト，検証）
    DATA = DATA.sample(frac=0.02, random_state=0).reset_index(drop=True)
    tokenizer = AutoTokenizer.from_pretrained("rinna/japanese-gpt-neox-3.6b", use_fast=False)
    train_df, test_df, valid_df = get_train_test(MAX_LENGTH, DATA, LAB2ID)

    print(len(train_df))
    loss_function = nn.NLLLoss()    # 損失関数の設定
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # GPUの設定

    config = LoraConfig(
        r=8,
        lora_alpha=16,
        # target_modules=["q_proj", "v_proj"], # Llamaを用いる場合
        target_modules=['query_key_value'], # GPT-NeoXを用いる場合
        #target_modules=['35.attention.query_key_value','34.attention.query_key_value',
        #                '33.attention.query_key_value','32.attention.query_key_value'], # GPT-NeoXを用いる場合
        lora_dropout=0.05,
        bias="none",
        #task_type="SEQ_CLS", # テキスト分類の場合は"SEQ_CLS"となる.modules_to_saveにスコアとくらしフィカーションを追加．自前の場合は設定しない，~~classifierのモデルとかで使う．
        modules_to_save=['my_linear', 'my_linear2']   # LoRA層以外の学習させたい層.LoRaは適用されない．LoRAで置き換えた部分以外は学習させていない->自分でoptimizer指定
    )
    classifier = RinnaClassifierSpan2DecodeIntegrated(OUTPUT_LAYER_DIM, MAX_LENGTH, device).to(device)
    classifier.rinna.pad_token_id = 0 # tokenizer の仕様
    classifier = get_peft_model(classifier, config)

    #ここでは全部ちゃんと設定しなくてはならない．
    base_params = [p for n, p in classifier.named_parameters() if "my_linear" not in n]
    #optimizer = optim.Adam([
    #    {'params': base_params, 'lr': 5e-4},
    #    {'params': classifier.my_linear.parameters(), 'lr': 1e-4},
    #    {'params': classifier.my_linear2.parameters(), 'lr': 1e-4},
    #])
    optimizer = optim.Adam([
        {'params': classifier.parameters()}
    ])
    
    """
    Train
    """
    # dataset, dateloader
    prev_f1, patience_counter = 0, 0
    train_dataset = mk_dataset(train_df, BATCH_SIZE, MAX_LENGTH, MAX_TOKEN, PRED_SEP_NUM, tokenizer, sort=True)
    valid_dataset = mk_dataset(valid_df, 1, MAX_LENGTH, MAX_TOKEN, PRED_SEP_NUM, tokenizer, sort=True)
    test_dataset = mk_dataset(test_df, 1, MAX_LENGTH, MAX_TOKEN, PRED_SEP_NUM, tokenizer, sort=True)
    time_start = datetime.datetime.now()
    # エポック数は5で
    for epoch in range(25):
        all_loss = 0
        classifier.train()
        for i, (batch_features, batch_preds, batch_token, batch_args) in enumerate(train_dataset):   # labelsはバッチ16こに対し128トークンのラベル
            # 使用するスパンの範囲を教える変数（使用可：１， 使用不可：－１）
            if i%100 == 0:
                print(f'Progress {i*BATCH_SIZE} / {len(train_dataset)*BATCH_SIZE}', batch_token[-1])
            sys.stdout.flush() # 明示的にflush
            ## 各特徴，ラベルをデバイスへ送る
            token_num = batch_token[-1] if MAX_LENGTH>=batch_token[-1] else MAX_LENGTH
            span_available_indication = np.ones([MAX_LENGTH, MAX_LENGTH]) * -1
            for span in list(itertools.combinations_with_replacement(np.arange(token_num), 2)):
                if (span[1] - span[0]) <= MAX_ARGUMENT_SEQUENCE_LENGTH:
                    span_available_indication[span[0]][span[1]] = 1
            #print(np.count_nonzero((span_available_indication >= 1)))
            input_ids = batch_features.to(device)         # input token ids
            pred_span = batch_preds.to(device)            # predicate span
            token_num = torch.tensor(batch_token,dtype=torch.long).to(device)
            #print(batch_args)
            target_ids_lists = [args[:,3] for args in batch_args]
            #print(target_ids_lists,batch_preds)
            target_ids = torch.cat([target_ids.unsqueeze(0) for target_ids in target_ids_lists], dim=0).to(device)   # target_ids[num_of_args] ... [14,55,67,33...] span_idx
            #my_dict_gpu = {key: value.to('cuda') for key, value in enumerate(batch_args)}
            
            outs = classifier(input_ids, pred_span, token_num, span_available_indication, 'train')
            #print(pred_span)
            #print(target_ids)
            #print(outs.shape)
            
            batch_loss = loss_function(outs.reshape(-1, MAX_LENGTH*MAX_LENGTH), target_ids.reshape(-1)) # outs:(batch, label, max) -> (batch*label,max)  target:(16, 31) -> (16*31)  one-hot: (16*31, max)
            batch_loss.backward() 
            all_loss += batch_loss.item()
            optimizer.step()
            classifier.zero_grad()  # 累積されるので，ここで初期化しなくてはならない．
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

        elif (prev_f1 >= f1) and (patience_counter < 4):   # 10回連続でf1が下がらなければ終了
            print('No change in valid f1\n') 
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
    classifier = RinnaClassifierSpan2DecodeIntegrated(OUTPUT_LAYER_DIM, MAX_LENGTH, device).to(device)
    classifier = PeftModel.from_pretrained(classifier, f"../../models/{MODEL_NAME}/")
    print(decode(test_dataset, test_df))