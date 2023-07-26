import numpy as np
import pandas as pd
import sys
pd.set_option('display.max_rows', 990)
pd.set_option('display.max_columns', 990)
np.set_printoptions(threshold=np.inf)
sys.path.append('../../')

import itertools
import subprocess
from preprocess.multi.mk_train_test_for_multi import get_train_test

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#from pcgrad import PCGrad

DATAPATH = '../../Data/data_v2_under53.json'
FORM = 'bert'
# read data
with open(DATAPATH, 'r', encoding="utf-8_sig") as json_file:
    data = pd.read_json(json_file)

# 正解ラベル（カテゴリー）をデータセットから取得
labels = []
for args in data['args']:
    labels += [ arg['argrole'] for arg in args]
labels = set(labels)
labels = sorted(list(labels)) + ['F-A', 'F-P', 'V', 'O', 'N']

# frameID の数は決まっている．v5_fid(0~2029)  v2_fid(1~1097)
FID_LAYER = max(data['predicate'].map(lambda x:x['frameID']))
fid2id=dict(zip(np.arange(1,FID_LAYER+1), np.arange(FID_LAYER)))
fid2id[-1] = -1


# カテゴリーのID辞書を作成，出力層の数定義
id2lab = dict(zip(list(range(len(labels))), labels))
lab2id = dict(zip(labels, list(range(len(labels)))))
print(lab2id, '\n')

# 各種定義
OUTPUT_LAYER = len(labels)                              # 全ラベルの数
PRED_SEP_CRITERION = 10 + 2                             # 述語情報のためのトークン数．sep:2, pred:8(最長)
MAX_LENGTH = 253
MAX_ARGUMENT_SEQUENCE_LENGTH = 30                       # 項の最高トークン数．（これより大きいものは予測不可能）
MAX_TOKEN = MAX_LENGTH + PRED_SEP_CRITERION + 1         # BERT に入力するトークン数．+1 は cls 分のトークン． 
print(f'MAX_TOKEN = {MAX_TOKEN}, MAX_LENGTH = {MAX_LENGTH}, MAX_ARGUMENT_SEQUENCE_LENGTH = {MAX_ARGUMENT_SEQUENCE_LENGTH}\n\n')

# 各種データ作成（学習，テスト，検証）
data = data.sample(frac=1, random_state=0).reset_index(drop=True)
train_df, test_df, valid_df = get_train_test(MAX_LENGTH, MAX_ARGUMENT_SEQUENCE_LENGTH, data, lab2id, fid2id)


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
        self.bert = BertModel.from_pretrained("cl-tohoku/bert-base-japanese-v2", output_attentions=True, output_hidden_states=True) # attention 受け取り，隠れ層取得
        # Final Layers
        self.linear = nn.Linear(768*2+1, 770)    # input, output
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(770, OUTPUT_LAYER)    # input, output
        self.linear_fid = nn.Linear(768*2+FID_LAYER, FID_LAYER)    # input, output

        # 重み初期化処理
        nn.init.normal_(self.linear.weight, std=0.02)   # std 正規分布の標準偏差
        nn.init.normal_(self.linear2.weight, std=0.02)
        nn.init.normal_(self.linear_fid.weight, std=0.02)
        nn.init.normal_(self.linear.bias, 0)
        nn.init.normal_(self.linear2.bias, 0)
        nn.init.normal_(self.linear_fid.bias, 0)

    # ベクトルを取得する用の関数
    def _get_cls_vec(self, vec):
        vecs = vec[:,0,:].view(-1, 768)     # cls ベクトル
        return vecs # vecs[batch][768]
    
    # ベクトルを取得する用の関数
    def _get_pred_vec(self, vec, pred_spans):
        vecs=[]
        for i, span in enumerate(pred_spans):
            vecs.append(vec[i, span[0]+1:span[1]+2, :].mean(0).reshape(1,768) )   # ex: vec[token_num][768]  mean-> vec[1][768] (0) は列に対して
        vecs = torch.cat([vec for vec in vecs], axis=0)
        return vecs # vecs[batch][768]
    # ベクトルを取得する用の関数
    
    def _get_pred_vec2(self, vec, pred_spans, token_nums):
        vecs=[]
        for i, pred, num in enumerate(zip(pred_spans, token_nums)):
            dis = pred[1]-pred[0]
            vecs.append(vec[i, num+2:num+3+dis, :].mean(0).reshape(1,768) ) 
        vecs = torch.cat([vec for vec in vecs], axis=0)
        return vecs # vecs[batch][768]
    
    # ベクトルを取得する用の関数
    def _get_token_vecs(self, vec, token_num):
        vecs = [vec[:,i+1,:].view(-1, 768) for i in range(token_num)]    # cls ベクトルはいらないので+1 range (0~48) -> (1~49)
        return vecs # vecs[MAX_LENGTH][batch][768]

    #@profile
    def forward(self, input_ids, pred_spans, token_nums, span_available_indication_matrix, fid_vecs):
        # 順伝播の出力結果は辞書形式なので、必要な値のkeyを指定して取得する
        output = self.bert(input_ids)
        hidden_states = output['hidden_states']

        # 隠れ層からそれぞれ トークンのベクトルを取得する
        vecs = self._get_token_vecs(hidden_states[-1], token_nums[-1])  # vecs[MAX_LENGTH][batch][768]
        cls_vecs = self._get_cls_vec(hidden_states[-1])  # vecs[batch][768]
        #pred_vecs = self._get_pred_vec(hidden_states[-1], pred_spans)  # vecs[batch][768]
        pred_vecs = self._get_pred_vec2(hidden_states[-1], pred_spans, token_nums)  # vecs[batch][768]

        # make span vec
        span_possible_tuples = [(i, j) for i in range(MAX_LENGTH) for j in range(MAX_LENGTH)]
        span_vec_list, pred_inside_span_batch =[], []
        for start, end in pred_spans:
            pred_inside_spans = list(itertools.combinations_with_replacement(torch.arange(start, end+1), 2))    # 重複組み合わせ
            pred_inside_spans.remove((start, end))
            pred_inside_span_batch.append(pred_inside_spans)    # pred_inside_span_batch[batch][pred_inside_spans]
        
        # vec に対する操作                : predicate -> [vec, 1]     , predicate_inside -> [vec, 1]      , not_predicate -> [vec, 0]
        # vec_combination の組に対する操作: predicate -> [vec_combi,2], predicate_inside -> [vec_combi, 1], not_predicate -> [vec_combi, 0]
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

        # 全結合層で追加した層用に次元を変換. span_vecを通す．
        outs = [self.linear(vec) if span_available_indication_matrix[i,j] == 1 else -1
                for vec,(i,j) in zip(span_vec_list, span_possible_tuples) ]   # vec はそれぞれのspanに対応したembedding(バッチのサイズあることに注意)
                
        outs = [self.relu(out) if span_available_indication_matrix[i,j] == 1 else -1
                for out,(i,j) in zip(outs, span_possible_tuples) ]

        outs = [self.linear2(out) if span_available_indication_matrix[i,j] == 1 else -1
                for out,(i,j) in zip(outs, span_possible_tuples) ]

        results = [F.log_softmax(out, dim=1) if span_available_indication_matrix[i,j] == 1 else torch.ones(OUTPUT_LAYER)*-1
                   for out,(i,j) in zip(outs, span_possible_tuples) ]
        
        # fid
        input_vecs = torch.cat([cls_vecs, pred_vecs, fid_vecs], 1)
        outs_fid = self.linear_fid(input_vecs) #[batch][cls+last+fid]
        results_fid = F.log_softmax(outs_fid, dim=1) #[batch][fid]
        
        return results, results_fid


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
for param in classifier.linear_fid.parameters():
    param.requires_grad = True

# 事前学習済の箇所は学習率小さめ、最後の全結合層は大きめにする。
ENC_NUM = 1
optimizer = optim.Adam([
    {'params': classifier.bert.encoder.layer[-1].parameters(), 'lr': 5e-5},
    {'params': classifier.bert.encoder.layer[-2].parameters(), 'lr': 5e-5},
    {'params': classifier.bert.encoder.layer[-3].parameters(), 'lr': 5e-5},
    {'params': classifier.bert.encoder.layer[-4].parameters(), 'lr': 5e-5},
    {'params': classifier.linear.parameters(), 'lr': 1e-4},
    {'params': classifier.linear2.parameters(), 'lr': 1e-4},
    {'params': classifier.linear_fid.parameters(), 'lr': 1e-4}
])
#optimizer = PCGrad(optim.Adam([
#    {'params': classifier.bert.encoder.layer[-1].parameters(), 'lr': 5e-5},
#    {'params': classifier.linear.parameters(), 'lr': 1e-4},
#    {'params': classifier.linear2.parameters(), 'lr': 1e-4},
#    {'params': classifier.linear_fid.parameters(), 'lr': 1e-4}
#]))


# 損失関数の設定
loss_function = nn.NLLLoss()

# GPUの設定
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ネットワークをGPUへ送る
classifier.to(device)
prev_f1 = -1
patience_counter = 0

def whether_to_stop(epoch):
    print('Validation Start\n')

    torch.save(classifier.state_dict(), f"../../models/multi_{MAX_LENGTH}_enc{ENC_NUM}_1v5_eachEP.pth")     # loss v srl
    # 標準エラーを標準出力に統合して取得
    try:
        proc = subprocess.run(['python3', '../Decode/decoder_for_multi.py', f'{MAX_LENGTH}', f'{MAX_ARGUMENT_SEQUENCE_LENGTH}', f'{ENC_NUM}', f'{DATAPATH}', f'{FORM}', 'valid'], stdout=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        # e.stderrでなくe.stdoutなのに注意．なお，e.outputでも可．
        print('ERROR:', e.stdout) # ERROR: b'cat: undefined.py: No such file or directory\n'
    report = pd.read_csv(f"../../report/Validation/results_{MAX_LENGTH}_enc{ENC_NUM}_multi.csv", index_col=0, encoding='utf-8_sig')
    f1 = report.loc['f1','correct_num'] if not np.isnan(report.loc['f1','correct_num']) else 0
    print(report)
    sys.stdout.flush() # 明示的にflush

    global prev_f1
    global patience_counter

    if prev_f1 < f1:
        patience_counter = 0
        prev_f1 = f1
        print('Valid f1 = ',f1,'\n')
        torch.save(classifier.state_dict(), f"../../models/multi_{MAX_LENGTH}_enc{ENC_NUM}_1v5_best.pth")
        return False

    elif (prev_f1 >= f1) and (patience_counter < 3):   # 10回連続でf1が下がらなければ終了
        print('No change in valid f1\n') 
        patience_counter += 1     
        return False
    else: 
        print('Stop. No change in valid f1\n') 
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

def preprocess(sentence_s, predicates_s, label_id, token_num, fid_candidates_lists, frameID, form):
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
    label_id = torch.tensor(list(label_id), dtype=torch.long)
    token_num = torch.tensor(list(token_num), dtype=torch.long)
    fid_candidate_vecs = torch.tensor(mk_fid_vecs(fid_candidates_lists), dtype=torch.long)
    frameID = torch.tensor(list(frameID), dtype=torch.long)
    return [sentences, label_id, pred_span, token_num, fid_candidate_vecs, frameID]

def mk_dataset(df, BATCH_SIZE, form):
    df.sort_values(by='num_of_tokens',inplace = True, ascending=True)
    batch_set = [df.iloc[i*BATCH_SIZE : (i+1)*BATCH_SIZE] for i in range(int(len(df)/BATCH_SIZE))]
    batch_set = [preprocess(set['sentence'], set['predicate'], set['label_id'], set['num_of_tokens'], set['fid_candidates'], set['frameID'], form) for set in batch_set]
    return batch_set


import datetime
from utils.evaluate import cal_accuracy
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

if __name__ == "__main__":
    #classifier.load_state_dict(torch.load(f'../../models/multi_{MAX_LENGTH}_enc{ENC_NUM}_best.pth'))
    #try:
    #    proc = subprocess.run(['python3', '../Decode/decoder_for_multi.py', f'{MAX_LENGTH}', f'{MAX_ARGUMENT_SEQUENCE_LENGTH}', f'{ENC_NUM}', f'{DATAPATH}', f'{FORM}', 'valid'], stdout=subprocess.PIPE)
    #except subprocess.CalledProcessError as e:
    #    # e.stderrでなくe.stdoutなのに注意．なお，e.outputでも可．
    #    print('ERROR:', e.stdout) # ERROR: b'cat: undefined.py: No such file or directory\n'
    #report = pd.read_csv(f"../../report/Validation/results_{MAX_LENGTH}_enc{ENC_NUM}_multi.csv", index_col=0, encoding='utf-8_sig')
    #f1 = report.loc['f1','correct_num'] if not np.isnan(report.loc['f1','correct_num']) else 0
    #print(report)
    #sys.stdout.flush() # 明示的にflush
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
    fid_preds, fid_ans = [],[]
    srl_preds, srl_ans = [],[]
    for epoch in range(50):
        all_loss = 0
        for i, (batch_features, batch_labels, batch_preds, batch_token, batch_cand_vecs, batch_fids) in enumerate(train_dataset):   # labelsはバッチ16こに対し128トークンのラベル
            # 使用するスパンの範囲を教える変数（使用可：１， 使用不可：－１）
            if i%100 == 0:
                print(f'Progress {i*BATCH_SIZE} / {len(train_dataset)*BATCH_SIZE}')
                sys.stdout.flush() # 明示的にflush
            token_num = int(batch_token[-1]) if int(batch_token[-1]) < MAX_LENGTH else MAX_LENGTH
            span_available_indication = np.zeros([MAX_LENGTH, MAX_LENGTH])
            span_available_indication[:,:] = -1
            num_of_span_vecs = torch.tensor(0)
            for span in list(itertools.combinations_with_replacement(np.arange(token_num), 2)):
                if (span[1] - span[0]) < MAX_ARGUMENT_SEQUENCE_LENGTH:
                    span_available_indication[span[0], span[1]] = 1
                    num_of_span_vecs += 1
            # 各特徴，ラベルをデバイスへ送る
            input_ids = batch_features.to(device)         # input token ids
            pred_span = batch_preds.to(device)            # predicate span
            temp = batch_labels.permute(1,2,0)            # batch_labels[batch][MAX_LENGTH][MAX_LENGTH] -> temp[MAX_LENGTH][MAX_LENGTH][batch]
            batch_labels_per_span = temp.view(MAX_LENGTH*MAX_LENGTH, len(batch_labels))
            label_ids = batch_labels_per_span.to(device)  # label_ids[MAX_LENGTH*MAX_LENGTH][batch] .各スパンごとの正解ラベル
            token_nums = torch.where(batch_token>=MAX_LENGTH, torch.tensor(MAX_LENGTH), batch_token).to(device)
            cand_vecs = batch_cand_vecs.to(device)        # candidate fid vecs
            fids = batch_fids.to(device)                   # correct fids
            
            outs = classifier(input_ids, pred_span, token_nums, span_available_indication, cand_vecs)
            # fid loss and span loss
            loss_fid = loss_function(outs[1], fids)
            _, pred = torch.max(outs[1], 1)
            fid_preds += pred.detach().clone().cpu()
            fid_ans += fids.detach().clone().cpu()
            for j,(input, target, indication) in enumerate(zip(outs[0], label_ids, span_available_indication.reshape(span_available_indication.size))):   # トークンの表現毎に学習を行う(inputはある表現に対する16（バッチ）のlogit)
                if indication == -1:
                    continue
                batch_loss = loss_function(input, target)   # 各データの正解ラベルを渡せば勝手にone-hotになる. input は softmax からの値． target は正解ラベル．
                loss = (loss_fid + batch_loss)/2.0
                loss.backward(retain_graph=True) # calculate the gradient can apply gradient modification                
                all_loss += loss.item()
                _, pred = torch.max(input, 1)
                srl_preds += pred.detach().clone().cpu()
                srl_ans += target.detach().clone().cpu()
            #for j,(input, target, indication) in enumerate(zip(outs[0], label_ids, span_available_indication.reshape(span_available_indication.size))):   # トークンの表現毎に学習を行う(inputはある表現に対する16（バッチ）のlogit)
            #    if indication == -1:
            #        continue
            #    batch_loss = loss_function(input[0], target)   # 各データの正解ラベルを渡せば勝手にone-hotになる. input は softmax からの値． target は正解ラベル．                
            #    losses = [loss_fid, batch_loss] # a list of per-task losses
            #    optimizer.pc_backward(losses, retain_graph=True) # calculate the gradient can apply gradient modification                
            #    all_loss += losses[0].item() + losses[1].item()
            optimizer.step()
            classifier.zero_grad()  # 累積されるので，ここで初期化しなくてはならない．
        print("epoch", epoch, "\t" , "loss", all_loss)
        print(f'FrameID : acc = {accuracy_score(fid_ans, fid_preds)}')
        print(f'SRL     : acc = {accuracy_score(srl_ans, srl_preds)}\n')
        sys.stdout.flush() # 明示的にflush
        #print(classification_report(fid_ans, fid_preds, labels=list(fid2id.kyes())) )
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
        proc = subprocess.run(['python3', '../Decode/decoder_for_multi.py', f'{MAX_LENGTH}', f'{MAX_ARGUMENT_SEQUENCE_LENGTH}', f'{ENC_NUM}', f'{DATAPATH}', f'{FORM}', 'test'], stdout=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        # e.stderrでなくe.stdoutなのに注意．なお，e.outputでも可．
        print('ERROR:', e.stdout) # ERROR: b'cat: undefined.py: No such file or directory\n'
    report = pd.read_csv(f"../../report/Test/results_{MAX_LENGTH}_enc{ENC_NUM}_multi.csv", index_col=0, encoding='utf-8_sig')
    print(report)