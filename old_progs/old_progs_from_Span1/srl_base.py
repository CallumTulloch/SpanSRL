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
from preprocess.base.mk_dataset import *
from preprocess.base.netwroks import BertClassifier
from preprocess.base.prepare_data import *

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


if __name__ == "__main__":
    # 各種データ作成（学習，テスト，検証）
    DATA = DATA.sample(frac=1, random_state=0).reset_index(drop=True)
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
    train_dataset = mk_dataset(train_df, BATCH_SIZE, MAX_LENGTH, MAX_TOKEN, PRED_SEP_NUM, tokenizer, sort=False)
    time_start = datetime.datetime.now()
    # エポック数は5で
    for epoch in range(50):
        all_loss = 0
        for i, (batch_features, batch_labels, batch_preds, batch_token) in enumerate(train_dataset):   # labelsはバッチ16こに対し128トークンのラベル
            # 使用するスパンの範囲を教える変数（使用可：１， 使用不可：－１）
            if i%100 == 0:
                print(f'Progress {i*BATCH_SIZE} / {len(train_dataset)*BATCH_SIZE}')
                sys.stdout.flush() # 明示的にflush
            token_num = int(batch_token[-1]) if int(batch_token[-1]) < MAX_LENGTH else MAX_LENGTH   # batch内最大トークン数．
            span_available_indication = np.ones([MAX_LENGTH, MAX_LENGTH]) * -1
            for span in list(itertools.combinations_with_replacement(np.arange(token_num), 2)):
                if (span[1] - span[0]) <= MAX_ARGUMENT_SEQUENCE_LENGTH:
                    span_available_indication[span[0], span[1]] = 1
            # 各特徴，ラベルをデバイスへ送る
            input_ids = batch_features.to(device)         # input token ids
            pred_span = batch_preds.to(device)            # predicate span
            token_num = torch.tensor(token_num,dtype=torch.long).to(device)
            
            temp = batch_labels.permute(1,2,0)            # batch_labels[batch][MAX_LENGTH][MAX_LENGTH] -> temp[MAX_LENGTH][MAX_LENGTH][batch]
            batch_labels_per_span = temp.view(MAX_LENGTH*MAX_LENGTH, len(batch_labels))
            label_ids = batch_labels_per_span.to(device)  # label_ids[MAX_LENGTH*MAX_LENGTH][batch] .各スパンごとの正解ラベル
            
            outs = classifier(input_ids, pred_span, token_num, span_available_indication)
            count = 0
            for j,(input, target, indication) in enumerate(zip(outs, label_ids, span_available_indication.reshape(span_available_indication.size))):   # トークンの表現毎に学習を行う(inputはある表現に対する16（バッチ）のlogit)
                if indication == -1:
                    continue
                #print(j, count)
                #print(input)
                #print(target)
                batch_loss = loss_function(input, target)    # 各データの正解ラベルを渡せば勝手にone-hotになる. input は softmax からの値． target は正解ラベル．
                batch_loss.backward(retain_graph=True)       # retain_graph をすることで各トークンのloss 毎に微分できる（勾配を累積できる）
                all_loss += batch_loss.item()
                count += 1
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