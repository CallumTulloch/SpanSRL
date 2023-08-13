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

from transformers import AutoTokenizer
from preprocess.multiDire.mk_dataset import *
from preprocess.multiDire.netwroks import BertClassifier
from preprocess.multiDire.prepare_data import *

DATAPATH = '../../Data/data_v2_under53_adddep2.json'

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
MAX_LENGTH = 53
MAX_ARGUMENT_SEQUENCE_LENGTH = 30                       # 項の最高トークン数．（これより大きいものは予測不可能）
MAX_TOKEN = MAX_LENGTH + PRED_SEP_NUM + 1         # BERT に入力するトークン数．+1 は cls 分のトークン． 
BATCH_SIZE = 16
print(f'MAX_TOKEN = {MAX_TOKEN}, MAX_LENGTH = {MAX_LENGTH}, MAX_ARGUMENT_SEQUENCE_LENGTH = {MAX_ARGUMENT_SEQUENCE_LENGTH}\n\n')

prev_f1, patience_counter = 0, 0
def whether_to_stop(epoch):
    global prev_f1
    global patience_counter
    
    print('Validation Start\n')
    torch.save(classifier.state_dict(), f"../../models/srl_base_{MAX_LENGTH}_enc{ENC_NUM}_eachEP.pth")
    # 標準エラーを標準出力に統合して取得
    try:
        proc = subprocess.run(['python3', '../Decode/decode_base.py', f'{MAX_LENGTH}', f'{MAX_ARGUMENT_SEQUENCE_LENGTH}', f'{ENC_NUM}', f'{DATAPATH}', 'valid'], stdout=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        # e.stderrでなくe.stdoutなのに注意．なお，e.outputでも可．
        print('ERROR:', e.stdout) # ERROR: b'cat: undefined.py: No such file or directory\n'
    report = pd.read_csv(f"../../report/Validation/base_{MAX_LENGTH}_enc{ENC_NUM}.csv", index_col=0, encoding='utf-8_sig')
    f1 = report.loc['f1','correct_num'] if not np.isnan(report.loc['f1','correct_num']) else 0
    print(report)
    sys.stdout.flush() # 明示的にflush

    if prev_f1 < f1:
        patience_counter = 0
        prev_f1 = f1
        print('Valid f1 = ',f1,'\n')
        torch.save(classifier.state_dict(), f"../../models/srl_base_{MAX_LENGTH}_enc{ENC_NUM}_best.pth")
        return False

    elif (prev_f1 >= f1) and (patience_counter < 4):   # 10回連続でf1が下がらなければ終了
        print('No change in valid f1\n') 
        patience_counter += 1     
        return False
    else: 
        print('Stop. No change in valid f1\n') 
        return True

import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

if __name__ == "__main__":
    # 各種データ作成（学習，テスト，検証）
    DATA = DATA.sample(frac=0.01, random_state=0).reset_index(drop=True)
    tokenizer = AutoTokenizer.from_pretrained("cl-tohoku/bert-base-japanese-v2")
    train_df, test_df, valid_df = get_train_test(MAX_LENGTH, MAX_ARGUMENT_SEQUENCE_LENGTH, DATA, LAB2ID)
    loss_function = nn.NLLLoss()    # 損失関数の設定
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # GPUの設定

    classifier = BertClassifier(OUTPUT_LAYER_DIM, MAX_LENGTH, device).to(device)
    # まずは全部OFF
    for param in classifier.parameters():
        param.requires_grad = False

    # BERTの最終4層分をON
    for param in classifier.bert.encoder.layer[-1].parameters():
        param.requires_grad = True
#    for param in classifier.bert.encoder.layer[-2].parameters():
#        param.requires_grad = True
#    for param in classifier.bert.encoder.layer[-3].parameters():
#        param.requires_grad = True
#    for param in classifier.bert.encoder.layer[-4].parameters():
#        param.requires_grad = True

    # 追加した層のところもON
    for param in classifier.linear.parameters():
        param.requires_grad = True
    for param in classifier.linear2.parameters():
        param.requires_grad = True

    # 事前学習済の箇所は学習率小さめ、最後の全結合層は大きめにする。
    ENC_NUM = 1
    optimizer = optim.Adam([
        {'params': classifier.bert.encoder.layer[-1].parameters(), 'lr': 5e-5},
        #{'params': classifier.bert.encoder.layer[-2].parameters(), 'lr': 5e-5},
        #{'params': classifier.bert.encoder.layer[-3].parameters(), 'lr': 5e-5},
        #{'params': classifier.bert.encoder.layer[-4].parameters(), 'lr': 5e-5},
        {'params': classifier.linear.parameters(), 'lr': 1e-4},
        {'params': classifier.linear2.parameters(), 'lr': 1e-4}
    ])
    """
    Train
    """
    # dataset, dateloader
    train_dataset = mk_dataset(train_df, BATCH_SIZE, MAX_LENGTH, MAX_TOKEN, PRED_SEP_NUM, tokenizer, sort=True)
    time_start = datetime.datetime.now()
    # エポック数は5で
    for epoch in range(50):
        all_loss = 0
        for i, (batch_features, batch_args, batch_labels, batch_preds, batch_token) in enumerate(train_dataset):   # labelsはバッチ16こに対し128トークンのラベル
            # 使用するスパンの範囲を教える変数（使用可：１， 使用不可：－１）
            if i%100 == 0:
                print(f'Progress {i*BATCH_SIZE} / {len(train_dataset)*BATCH_SIZE}')
                sys.stdout.flush() # 明示的にflush
            token_num = int(batch_token[-1]) if int(batch_token[-1]) < MAX_LENGTH else MAX_LENGTH   # batch内最大トークン数．
            span_available_indication = np.ones([MAX_LENGTH, MAX_LENGTH]) * -1
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
            
            target_ids_lists = [args[:,3] for args in batch_args]
            target_ids = torch.cat([target_ids.unsqueeze(0) for target_ids in target_ids_lists], dim=0).to(device)   # target_ids[num_of_args] ... [14,55,67,33...] span_idx
            my_dict_gpu = {key: value.to('cuda') for key, value in enumerate(batch_args)}
            
            outs_vertical, outs_horizion = classifier(input_ids, pred_span, token_num, span_available_indication) # [batch][max][label]
            batch_loss1 = loss_function(outs_vertical.reshape(-1, len(LABELS)), label_ids.reshape(-1))    # (batch*max,label)  : (batch*max), 各スパンに対して，正解ラベルを用意
            batch_loss2 = loss_function(outs_horizion.reshape(-1, 2809), target_ids.reshape(-1)) # outs:(batch, label, max) -> (batch*label,max)  target:(16, 31) -> (16*31)  one-hot: (16*31, max)
            batch_loss = batch_loss1 + batch_loss2
            batch_loss.backward()

            all_loss += batch_loss.item()

            optimizer.step()
            classifier.zero_grad()  # 累積されるので，ここで初期化しなくてはならない．
        print("epoch", epoch, "\t" , "loss", all_loss)
        """
        Validation
        """
        if whether_to_stop(epoch):
            break
    # 時間計測
    time_end = datetime.datetime.now()
    print('\n', time_end - time_start, '\n')

    """
    Test
    """
    try:
        proc = subprocess.run(['python3', '../Decode/decode_base.py', f'{MAX_LENGTH}', f'{MAX_ARGUMENT_SEQUENCE_LENGTH}', f'{ENC_NUM}', f'{DATAPATH}', 'test'], stdout=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        # e.stderrでなくe.stdoutなのに注意．なお，e.outputでも可．
        print('ERROR:', e.stdout) # ERROR: b'cat: undefined.py: No such file or directory\n'