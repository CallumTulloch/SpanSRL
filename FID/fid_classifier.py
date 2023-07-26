import numpy as np
import pandas as pd
import sys
pd.set_option('display.max_rows', 990)
pd.set_option('display.max_columns', 990)
np.set_printoptions(threshold=np.inf)
sys.path.append('../../')

import itertools
import subprocess
from tqdm import tqdm
from mk_train_test_for_fid import get_train_test

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#from pcgrad import PCGrad

#DATAPATH = r'C:\Users\callu\Desktop\Univ\SRL_v2\Data\data_v5_under53_fid.json'
DATAPATH = '/home/callum/Desktop/SRL_ALL/Data/data_v2_myCandidate.json'
FORM = 'bert'
# read data
#with open(DATAPATH, 'r', encoding="utf-8_sig") as json_file:
#    data = pd.read_json(json_file)
json_file = open(f'{DATAPATH}', 'r',encoding='utf-8')
data = pd.read_json(json_file, orient='records', lines=True)

# 正解ラベル（カテゴリー）をデータセットから取得
labels = []
for args in data['args']:
    labels += [ arg['argrole'] for arg in args]
labels = set(labels)
labels = sorted(list(labels)) + ['F-A', 'F-P', 'V', 'O', 'N']

# frameID の数は決まっている．v5_fid(0~2029)  v2_fid(1~1097)
FID_LAYER = max(data['predicate'].map(lambda x:x['frameID']))
#FID_LAYER = 1097
fid2id=dict(zip(np.arange(1,FID_LAYER+1), np.arange(FID_LAYER)))
fid2id[-1] = -1

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
print(f'MAX_TOKEN = {MAX_TOKEN}, MAX_LENGTH = {MAX_LENGTH}, MAX_ARGUMENT_SEQUENCE_LENGTH = {MAX_ARGUMENT_SEQUENCE_LENGTH}\n\n')

from torch.utils.data import Dataset
from transformers import BertModel
from transformers import AutoTokenizer
import datetime
from sklearn.metrics import accuracy_score

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
        self.linear_fid = nn.Linear(768*2+FID_LAYER, 768+FID_LAYER)    # input, output
        self.linear_fid2 = nn.Linear(768+FID_LAYER, FID_LAYER)    # input, output
        self.relu = nn.ReLU()

        # 重み初期化処理
        nn.init.normal_(self.linear_fid.weight, std=0.02)   # std 正規分布の標準偏差
        nn.init.normal_(self.linear_fid.bias, 0)

    # ベクトルを取得する用の関数
    def _get_cls_vec(self, vec):
        vecs = vec[:,0,:].view(-1, 768)     # cls ベクトル
        return vecs # vecs[batch][768]
    
    # ベクトルを取得する用の関数
    def _get_last_vecs(self, vec, pred_spans):
        vecs=[]
        for i, span in enumerate(pred_spans):
            span_len = span[1]-span[0]+1
            pred_start = MAX_LENGTH+2
            pred_end = pred_start+span_len
            vecs.append(vec[i, pred_start:pred_end, :].mean(0).reshape(1,768) )   # ex: vec[token_num][768]  mean-> vec[1][768] (0) は列に対して
        vecs = torch.cat([vec for vec in vecs], axis=0)        
        return vecs # vecs[batch][768]
    
    # ベクトルを取得する用の関数
    def _get_pred_vec(self, vec, pred_spans):
        vecs=[]
        for i, span in enumerate(pred_spans):
            vecs.append(vec[i, span[0]+1:span[1]+2, :].mean(0).reshape(1,768) )   # ex: vec[token_num][768]  mean-> vec[1][768] (0) は列に対して
        vecs = torch.cat([vec for vec in vecs], axis=0)
        return vecs # vecs[batch][768]

    #@profile
    def forward(self, input_ids, pred_spans, fid_vecs):
        # 順伝播の出力結果は辞書形式なので、必要な値のkeyを指定して取得する
        output = self.bert(input_ids)
        hidden_states = output['hidden_states']

        # 隠れ層からそれぞれ トークンのベクトルを取得する
        cls_vecs = self._get_cls_vec(hidden_states[-1])  # vecs[batch][768]
        #pred_vec = self._get_last_vecs(hidden_states[-1], pred_spans)  # vecs[batch][768]
        pred_vec = self._get_pred_vec(hidden_states[-1], pred_spans)  # vecs[batch][768]

        # fid
        #input_vecs = torch.cat([cls_vecs, pred_vec], axis= 1)
        input_vecs = torch.cat([cls_vecs, pred_vec, fid_vecs], axis= 1)
        outs_fid = self.linear_fid(input_vecs) #[batch][cls+last+fid]
        outs_fid = self.relu(outs_fid) #[batch][cls+last+fid]
        outs_fid = self.linear_fid2(outs_fid) #[batch][cls+last+fid]
        results_fid = F.log_softmax(outs_fid, dim=1) #[batch][fid]
        
        return results_fid


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
for param in classifier.linear_fid.parameters():
    param.requires_grad = True

# 事前学習済の箇所は学習率小さめ、最後の全結合層は大きめにする。
ENC_NUM = 1
optimizer = optim.Adam([
    {'params': classifier.bert.encoder.layer[-1].parameters(), 'lr': 5e-5},
    {'params': classifier.bert.encoder.layer[-2].parameters(), 'lr': 5e-5},
    {'params': classifier.bert.encoder.layer[-3].parameters(), 'lr': 5e-5},
    {'params': classifier.bert.encoder.layer[-4].parameters(), 'lr': 5e-5},
    {'params': classifier.linear_fid.parameters(), 'lr': 1e-4}
])

# 損失関数の設定
loss_function = nn.NLLLoss()

# GPUの設定
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ネットワークをGPUへ送る
classifier.to(device)
prev_acc = -1
patience_counter = 0

def whether_to_stop(epoch, valid_dataset):
    print('Validation Start\n')
    torch.save(classifier.state_dict(), f"models/fid_{MAX_LENGTH}_enc{ENC_NUM}_eachEP.pth")
    fid_preds, fid_ans = [],[]
    all_loss = 0
    with torch.no_grad():
        for i, (batch_features, batch_preds, batch_token, batch_cand_vecs, batch_fids) in enumerate(valid_dataset):   # labelsはバッチ16こに対し128トークンのラベル
            # 各特徴，ラベルをデバイスへ送る
            input_ids = batch_features.to(device)         # input token ids
            cand_vecs = batch_cand_vecs.to(device)        # candidate fid vecs
            pred_span = batch_preds.to(device)            # predicate span
            fids = batch_fids.to(device)                   # correct fids
            
            outs = classifier(input_ids, pred_span, cand_vecs)
            # fid loss and span loss
            loss_fid = loss_function(outs, fids)
            all_loss += loss_fid.item()

            optimizer.step()
            classifier.zero_grad()  # 累積されるので，ここで初期化しなくてはならない．

            _, pred = torch.max(outs, 1)
            fid_preds += pred.detach().clone().cpu()
            fid_ans += fids.detach().clone().cpu()
        acc = accuracy_score(fid_ans, fid_preds)
        print(acc)

    global prev_acc
    global patience_counter

    if prev_acc < acc:
        patience_counter = 0
        prev_acc = acc
        print('Valid acc = ',acc,'\n')
        torch.save(classifier.state_dict(), f"models/fid_{MAX_LENGTH}_enc{ENC_NUM}_best.pth")
        return False

    elif (prev_acc >= acc) and (patience_counter < 3):   # 10回連続でaccが下がらなければ終了
        print('No change in valid acc\n') 
        patience_counter += 1     
        return False
    else: 
        print('Stop. No change in valid acc\n') 
        return True

def mk_fid_vecs(fid_candidates_lists):
    fid_vecs=[]
    for fid_candidates in fid_candidates_lists:
        fid_vec = np.zeros(FID_LAYER)
        for fid in fid_candidates:
            if fid == -1:
                break
            else:
                fid_vec[fid] = 1
        fid_vecs.append(fid_vec)
    return fid_vecs


def preprocess(sentence_s, predicates_s, token_num, fid_candidates_lists, frameID, form):
    sentences = []
    pred_span = []
    for sent, pred in zip(sentence_s, predicates_s):
        if form == 'unidic':
            sentences.append(bert_tokenizer_unidic(sent, pred))
        if form == 'bert':
            sentences.append(bert_tokenizer_bert(sent, pred))
        pred_span.append((pred['word_start'],pred['word_end']))
    # 各要素の
    sentences = torch.tensor(sentences, dtype=torch.long)
    pred_span = torch.tensor(pred_span, dtype=torch.long)
    token_num = torch.tensor(list(token_num), dtype=torch.long)
    fid_candidate_vecs = torch.tensor(mk_fid_vecs(fid_candidates_lists), dtype=torch.long)
    frameID = torch.tensor(list(frameID), dtype=torch.long)
    return [sentences, pred_span, token_num, fid_candidate_vecs, frameID]

def mk_dataset(df, BATCH_SIZE, form):
    df.sort_values(by='num_of_tokens',inplace = True, ascending=True)
    batch_set = [df.iloc[i*BATCH_SIZE : (i+1)*BATCH_SIZE] for i in range(int(len(df)/BATCH_SIZE))]
    batch_set = [preprocess(set['sentence'], set['predicate'], set['num_of_tokens'], set['fid_candidates'], set['frameID'], form) for set in batch_set]
    return batch_set



# 各種データ作成（学習，テスト，検証）
data = data.sample(frac=1, random_state=0).reset_index(drop=True)
train_df, test_df, valid_df = get_train_test(MAX_LENGTH, MAX_ARGUMENT_SEQUENCE_LENGTH, data, lab2id, fid2id)
test_dataset = mk_dataset(test_df, 16, FORM)
valid_dataset = mk_dataset(valid_df, 16, FORM)

classifier = BertClassifier()
classifier.to(device)
classifier.load_state_dict(torch.load(f'../../models/fid_{MAX_LENGTH}_enc{ENC_NUM}_best.pth'))
fid_preds, fid_ans = [],[]
all_loss = 0
with torch.no_grad():
    for i, (batch_features, batch_labels, batch_preds, batch_token, batch_cand_vecs, batch_fids) in enumerate(test_dataset):   # labelsはバッチ16こに対し128トークンのラベル
        # 使用するスパンの範囲を教える変数（使用可：１， 使用不可：－１）
        print(f'Progress {i*BATCH_SIZE} / {len(valid_dataset)*BATCH_SIZE}')
        # 各特徴，ラベルをデバイスへ送る
        input_ids = batch_features.to(device)         # input token ids
        pred_span = batch_preds.to(device)            # predicate span
        cand_vecs = batch_cand_vecs.to(device)        # candidate fid vecs
        fids = batch_fids.to(device)                   # correct fids
        
        outs = classifier(input_ids, pred_span, cand_vecs)
        # fid loss and span loss 
        loss_fid = loss_function(outs, fids)
        all_loss += loss_fid.item()

        optimizer.step()
        classifier.zero_grad()  # 累積されるので，ここで初期化しなくてはならない．

        _, pred = torch.max(outs, 1)
        fid_preds += pred.detach().clone().cpu()
        fid_ans += fids.detach().clone().cpu()
    acc = accuracy_score(fid_ans, fid_preds)
    print(acc)