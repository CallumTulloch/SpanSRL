import numpy as np
import pandas as pd
import sys
import os
pd.set_option('display.max_rows', 990)
pd.set_option('display.max_columns', 990)
np.set_printoptions(threshold=np.inf)
sys.path.append('../../')
sys.path.append('../')

import itertools
import subprocess
from tqdm import tqdm
from preprocess.mk_train_test_span2 import get_train_test

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# NLPIR
DATAPATH = '../../../Data/common_data_v2_bert.json'
FORM = 'bert'
with open(DATAPATH, 'r', encoding="utf-8_sig") as json_file:
    DATA = pd.read_json(json_file)


# EMNLP
# DATAPATH = '../../../../Ohuchi_old/Data/data_v2_under53.json'
# EXDATAPATH = '../../../../Ohuchi_old/Data/data_v2_extra2.json'
# FORM = 'bert'
# with open(DATAPATH, 'r', encoding="utf-8_sig") as json_file:
#     DATA = pd.read_json(json_file)
#     
# with open(EXDATAPATH, 'r', encoding="utf-8_sig") as json_file:
#     EXDATA = pd.read_json(json_file)


# 正解ラベル（カテゴリー）をデータセットから取得
labels = []
for args in DATA['args']:
    labels += [ arg['argrole'] for arg in args]
labels = set(labels)
labels = sorted(list(labels))

# カテゴリーのID辞書を作成，出力層の数定義
id2lab = dict(zip(list(range(len(labels))), labels))
lab2id = dict(zip(labels, list(range(len(labels)))))
print(lab2id, '\n')

# 各種定義
OUTPUT_LAYER = len(labels)                              # 全ラベルの数
PRED_SEP_CRITERION = 10 + 2                              # 述語情報のためのトークン数．sep:2, pred:8(最長)
MAX_LENGTH = 252
MAX_ARGUMENT_SEQUENCE_LENGTH = 30                       # 項の最高トークン数．（これより大きいものは予測不可能）
MAX_TOKEN = MAX_LENGTH + PRED_SEP_CRITERION + 1         # BERT に入力するトークン数．+1 は cls 分のトークン． 
FLAG_USE_ALL_DATA = True
num_of_span_vec = 0
SPAN_AVAILABLE_INDICATION = np.zeros([MAX_LENGTH, MAX_LENGTH])
SPAN_AVAILABLE_INDICATION[:,:] = -1
for span in list(itertools.combinations_with_replacement(np.arange(MAX_LENGTH), 2)):
    if (span[1] - span[0]) <= MAX_ARGUMENT_SEQUENCE_LENGTH:
        SPAN_AVAILABLE_INDICATION[span[0], span[1]] = 1
        num_of_span_vec += 1
NUM_OF_SPAN_VEC = num_of_span_vec
MODEL_NAME = 'span2_common_data'
print(f'MAX_TOKEN = {MAX_TOKEN}, MAX_LENGTH = {MAX_LENGTH}, MAX_ARGUMENT_SEQUENCE_LENGTH = {MAX_ARGUMENT_SEQUENCE_LENGTH}\n\n')

# 各種データ作成（学習，テスト，検証）
DATA = DATA.sample(frac=1, random_state=0).reset_index(drop=True)
train_df, test_df, valid_df = get_train_test(MAX_LENGTH, DATA, lab2id)

# EMNLP
# EXDATA = EXDATA.sample(frac=1, random_state=0).reset_index(drop=True)
# extrain_df, extest_df, exvalid_df = get_train_test(MAX_LENGTH, EXDATA, lab2id)
# train_df=pd.concat([train_df,extrain_df],axis=0)
# test_df=pd.concat([test_df,extest_df],axis=0)
# valid_df=pd.concat([valid_df,exvalid_df],axis=0)
# print(len(test_df))

# span2 はバックワードの時に30以上の場合学習に影響が出る．複数あるargの内，全てが30以上の場合はデータ削除
#train_df['arg_len'] = train_df['original_args_info'].map(lambda x: min(len(arg['surface'].split(' ')) for arg in x))
#train_df = train_df[train_df['arg_len'] <= 30]

from torch.utils.data import Dataset
from transformers import BertModel
from transformers import AutoTokenizer


BATCH_SIZE = 16
tokenizer = AutoTokenizer.from_pretrained("cl-tohoku/bert-base-japanese-v2")
def bert_tokenizer_unidic(wakati, pred): 
    #wakati = normalize(wakati)
    token_num = len(wakati.split(' '))
    pred_token_num = len(pred['surface'].split(' '))
    pred_sep_num =  pred_token_num + 2 if pred_token_num <= PRED_SEP_CRITERION-2 else PRED_SEP_CRITERION

    if token_num <= MAX_LENGTH:
        front_padding_num = MAX_LENGTH - token_num
        back_padding_num = MAX_TOKEN - MAX_LENGTH - pred_sep_num - 1 # -1 は cls
        tokens = ['[CLS]'] + wakati.split(' ') + ['[PAD]']*front_padding_num + ['[SEP]'] + pred['surface'].split(' ')[:pred_token_num] + ['[SEP]'] + ['[PAD]']*back_padding_num
    else:
        padding_num = MAX_TOKEN - MAX_LENGTH - pred_sep_num - 1 # padding >= 0 は保証
        tokens = ['[CLS]'] + wakati.split(' ')[:MAX_LENGTH] + ['[SEP]'] + pred['surface'].split(' ')[:pred_token_num] + ['[SEP]'] + ['[PAD]']*padding_num
    ids = torch.tensor( [tokenizer.encode(token)[1] for token in tokens], dtype=torch.long)
    #print(tokenizer.convert_ids_to_tokens(ids))
    #print(len(ids))
    return ids

def bert_tokenizer_bert(wakati, pred): 
    #wakati = normalize(wakati)
    token_num = len(wakati.split(' '))
    pred_token_num = len(pred['surface'].split(' '))
    pred_sep_num =  pred_token_num + 2 if pred_token_num <= PRED_SEP_CRITERION-2 else PRED_SEP_CRITERION

    if token_num <= MAX_LENGTH:
        front_padding_num = MAX_LENGTH - token_num
        back_padding_num = MAX_TOKEN - MAX_LENGTH - pred_sep_num - 1 # -1 は cls
        tokens = ['[CLS]'] + wakati.split(' ') + ['[PAD]']*front_padding_num + ['[SEP]'] + pred['surface'].split(' ')[:pred_token_num] + ['[SEP]'] + ['[PAD]']*back_padding_num
    else:
        padding_num = MAX_TOKEN - MAX_LENGTH - pred_sep_num - 1 # padding >= 0 は保証
        tokens = ['[CLS]'] + wakati.split(' ')[:MAX_LENGTH] + ['[SEP]'] + pred['surface'].split(' ')[:pred_token_num] + ['[SEP]'] + ['[PAD]']*padding_num
    ids = np.array(tokenizer.convert_tokens_to_ids(tokens))
    #print(tokenizer.convert_ids_to_tokens(ids))
    #print(len(ids))
    return ids

class BertClassifier(nn.Module):
    def __init__(self):
        super(BertClassifier, self).__init__()
        # BERT
        self.bert = BertModel.from_pretrained('cl-tohoku/bert-base-japanese-v2', output_attentions=True, output_hidden_states=True) # attention 受け取り，隠れ層取得
        # Final Layers
        self.linear = nn.Linear(768*2+1, 770)    # input, output
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(770, OUTPUT_LAYER)    # input, output

        # 重み初期化処理
        nn.init.normal_(self.linear.weight, std=0.02)   # std 正規分布の標準偏差
        nn.init.normal_(self.linear2.weight, std=0.02)
        nn.init.normal_(self.linear.bias, 0)
        nn.init.normal_(self.linear2.bias, 0)

    # ベクトルを取得する用の関数
    def _get_final_vecs(self, vec):
        vecs = [vec[:,i+1,:].view(-1, 768) for i in range(MAX_LENGTH)]    # cls ベクトルはいらないので+1 range (0~48) -> (1~49)
        return vecs # vecs[token_num][batch][768]

    def forward(self, input_ids, args_dic, pred_spans, token_num, span_available_indication_matrix):
        # 順伝播の出力結果は辞書形式なので、必要な値のkeyを指定して取得する
        output = self.bert(input_ids)
        hidden_states = output['hidden_states']

        # 隠れ層からそれぞれ トークンのベクトルを取得する
        vecs = self._get_final_vecs(hidden_states[-1])  # vecs[MAX_LENGTH][batch][768]

        # make span vec
        span_possible_tuples = [(i, j) for i in range(MAX_LENGTH) for j in range(MAX_LENGTH)]
        span_vec_list, pred_inside_span_batch =[], []
        for start, end in pred_spans:
            pred_inside_spans = list(itertools.combinations_with_replacement(torch.arange(start, end+1), 2))    # 重複組み合わせ
            pred_inside_spans.remove((start, end))
            pred_inside_span_batch.append(torch.tensor(pred_inside_spans).to(device))    # pred_inside_span_batch[batch][pred_inside_spans]
        
        # vec_combination の組に対する操作: predicate -> [vec_combi,2], predicate_inside -> [vec_combi, 1], not_predicate -> [vec_combi, 0]
        zeros = torch.zeros(len(lab2id)*len(input_ids)).reshape(len(input_ids), len(lab2id)).to(device)
        for c, (i, j) in enumerate(span_possible_tuples):   # span_possible_tuples[MAX_LENGTH][MAX_LENGTH]
            if span_available_indication_matrix[i,j] >= 1:   # 使用する範囲かどうか
                # predicate への操作
                vec_i, vec_j = vecs[i], vecs[j]     # vec_i[batch][768]

                pred_indications = []
                target_span = torch.tensor([i,j]).to(device)
                for pred_inside_span, p_span in zip(pred_inside_span_batch, pred_spans):
                    #if i,j == p_span[0].item(),p_span[1].item():
                    if all(target_span == p_span):
                        pred_indications.append(2)
                    elif pred_inside_span.nelement() == 0:
                        pred_indications.append(0)
                    elif any(torch.all(torch.eq(pred_inside_span, target_span), dim=1)):
                        pred_indications.append(1)
                    else:
                        pred_indications.append(0)
                pred_indications = torch.tensor(pred_indications).reshape(len(pred_inside_span_batch),1).to(device)
                span_vec_list.append( torch.cat([vec_i, vec_j, pred_indications], 1) )
            else:
                span_vec_list.append(0)
                continue
        #print(len(span_vec_list), len(span_vec_list[0]), len(span_vec_list[0][0]))
        # 全結合層で追加した層用に次元を変換. span_vecを通す．
        outs = [self.linear(vec) if span_available_indication_matrix[i,j] >= 1 else 0
                for vec,(i,j) in zip(span_vec_list, span_possible_tuples) ]   # vec はそれぞれのspanに対応したembedding(バッチのサイズあることに注意)
                
        outs = [self.relu(out) if span_available_indication_matrix[i,j] >= 1 else 0
                for out,(i,j) in zip(outs, span_possible_tuples) ]

        outs = [self.linear2(out) if span_available_indication_matrix[i,j] >= 1 else zeros
                for out,(i,j) in zip(outs, span_possible_tuples) ]
        
        ##outs=[MAX][batch][label] -> outs=[batch][max][label]
        outs = torch.concat([out.unsqueeze(1) for out in outs],dim=1)   # out.unsqueeze(1)=[batch][1][label]
        #print(outs.shape)
        results = F.log_softmax(outs, dim=1)  # outs[b_idx, max, label]
        results =  results.permute(0,2,1)     # results = [batch][label][max]
        #print(results.sum(dim=2))             # sum をdim=2にそって行う → dim2以外は次元はsame.結果は全部1になるはず
        return results


classifier = BertClassifier()


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


# 損失関数の設定
loss_function = nn.NLLLoss()

# GPUの設定
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ネットワークをGPUへ送る
classifier.to(device)
prev_f1, patience_counter = 0, 0

def whether_to_stop(epoch):
    global prev_f1
    global patience_counter
    print('Validation Start\n')
    torch.save(classifier.state_dict(), f"../../models/srl_{MODEL_NAME}_eachEP.pth")
    try:
        proc = subprocess.run(['python', '../Decode/decode_span2_manu.py', f'{MAX_LENGTH}', f'{MAX_ARGUMENT_SEQUENCE_LENGTH}', f'{ENC_NUM}', f'{DATAPATH}','bert', 'valid', f'{MODEL_NAME}'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output = proc.stdout
    except subprocess.CalledProcessError as e:
        # e.stderrでなくe.stdoutなのに注意．なお，e.outputでも可．
        output = proc.stdout
        print(output)
        print('ERROR:', e.stdout) # ERROR: b'cat: undefined.py: No such file or directory\n'    report = pd.read_csv(f"../../report/Validation/results_{MODEL_NAME}.csv", index_col=0, encoding='utf-8_sig')
    #print(output)
    report = pd.read_csv(f"span2_label.csv", index_col=0, encoding='utf-8_sig')
    f1 = report.loc['f1','correct_num'] if not np.isnan(report.loc['f1','correct_num']) else 0
    print(report)
    sys.stdout.flush() # 明示的にflush

    if prev_f1 < f1:
        patience_counter = 0
        prev_f1 = f1
        print('Valid f1 = ',f1,'\n')
        torch.save(classifier.state_dict(), f"../../models/srl_{MODEL_NAME}_best.pth")
        return False

    elif (prev_f1 >= f1) and (patience_counter < 2):   # 10回連続でf1が下がらなければ終了
        print('No change in valid f1\n') 
        patience_counter += 1     
        return False
    else: 
        print('Stop. No change in valid f1\n') 
        return True

def preprocess(sentence_s, predicates_s, token_num, args_s, form):
    sentences = []
    pred_span = []
    for sent, pred in zip(sentence_s, predicates_s):
        if form == 'unidic':
            sentences.append(bert_tokenizer_unidic(sent, pred))
        if form == 'bert':
            sentences.append(bert_tokenizer_bert(sent, pred))
        pred_span.append((pred['word_start'],pred['word_end']))
    sentences = torch.tensor(sentences, dtype=torch.long)
    pred_span = torch.tensor(pred_span, dtype=torch.long)
    token_num = torch.tensor(list(token_num), dtype=torch.long)
    #print(args_s)
    args = [torch.tensor(args) for args in args_s.tolist()]
    return [sentences, pred_span, token_num, args]

def mk_dataset(df, BATCH_SIZE, form):
    df.sort_values(by='num_of_tokens',inplace = True, ascending=True)
    batch_set = [df.iloc[i*BATCH_SIZE : (i+1)*BATCH_SIZE] for i in range(int(len(df)/BATCH_SIZE))]
    batch_set = [preprocess(set['sentence'], set['predicate'], set['num_of_tokens'], set['args'], form) for set in batch_set]
    return batch_set


import datetime
from sklearn.metrics import accuracy_score

if __name__ == "__main__":
    #classifier.load_state_dict(torch.load('../../models/srl_ohuchi_incNull_f1_best.pth'))

    """
    Train
    """
    # dataset, dateloader
    train_dataset = mk_dataset(train_df, 16, FORM)
    test_dataset = mk_dataset(test_df, 16, FORM)
    valid_dataset = mk_dataset(valid_df, 16, FORM)
    #whether_to_stop(0, valid_dataset)

    time_start = datetime.datetime.now()
    # エポック数は5で
    for epoch in range(25):
        all_loss = 0
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
            my_dict_gpu = {key: value.to('cuda') for key, value in enumerate(batch_args)}
            #print(target_ids.shape, pred_span.shape)
            
            outs = classifier(input_ids, my_dict_gpu, pred_span, token_num, span_available_indication)
            #print(pred_span)
            #print(target_ids)
            #print(outs)
            
            batch_loss = loss_function(outs.reshape(-1, MAX_LENGTH*MAX_LENGTH), target_ids.reshape(-1)) # outs:(batch, label, max) -> (batch*label,max)  target:(16, 31) -> (16*31)  one-hot: (16*31, max)
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
    #try:
    #    proc = subprocess.run(['python', '../Decode/decode_span2_manu.py', f'{MAX_LENGTH}', f'{MAX_ARGUMENT_SEQUENCE_LENGTH}', f'{ENC_NUM}', f'{DATAPATH}','bert', 'test', f'{MODEL_NAME}'], stdout=subprocess.PIPE)
    #except subprocess.CalledProcessError as e:
    #    # e.stderrでなくe.stdoutなのに注意．なお，e.outputでも可．
    #    print('ERROR:', e.stdout) # ERROR: b'cat: undefined.py: No such file or directory\n'
    #report = pd.read_csv(f"../../report/Test/results_{MODEL_NAME}.csv", index_col=0, encoding='utf-8_sig')
    #f1 = report.loc['f1','correct_num'] if not np.isnan(report.loc['f1','correct_num']) else 0
    #print(report)