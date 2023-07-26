import numpy as np
import pandas as pd
import sys
import os
pd.set_option('display.max_rows', 990)
pd.set_option('display.max_columns', 990)
np.set_printoptions(threshold=np.inf)
sys.path.append('../')
sys.path.append('../../')
sys.path.append('../../../')
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import time
import itertools
from tqdm import tqdm
from preprocess.mk_train_test_span2 import get_train_test_decode

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gc

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
print(len(lab2id))

                        
# 各種定義
OUTPUT_LAYER = len(labels)                              # 全ラベルの数
PRED_SEP_CRITERION = 10 + 2                             # 述語情報のためのトークン数．sep:2, pred:8(最長)
MAX_LENGTH = 252                                        # ラベル割り当てするトークン数. 文章の最長は 243 token
MAX_ARGUMENT_SEQUENCE_LENGTH = 30                       # 項の最高トークン数．（これより大きいものは予測不可能）
MAX_TOKEN = MAX_LENGTH + PRED_SEP_CRITERION + 1         # BERT に入力するトークン数．+1 は cls 分のトークン． 
SPAN_AVAILABLE_INDICATION = np.zeros([MAX_LENGTH, MAX_LENGTH])
for span in list(itertools.combinations_with_replacement(np.arange(MAX_LENGTH), 2)):
        if (span[1] - span[0]) <= MAX_ARGUMENT_SEQUENCE_LENGTH:
            SPAN_AVAILABLE_INDICATION[span[0], span[1]] = 1
print(f'MAX_TOKEN = {MAX_TOKEN}, MAX_LENGTH = {MAX_LENGTH}, MAX_ARGUMENT_SEQUENCE_LENGTH = {MAX_ARGUMENT_SEQUENCE_LENGTH}\n\n')

# 各種データ作成（学習，テスト，検証）
DATA = DATA.sample(frac=1, random_state=0).reset_index(drop=True)
test_df = get_train_test_decode(MAX_LENGTH, DATA, lab2id)

# EMNLP
EXDATA = EXDATA.sample(frac=1, random_state=0).reset_index(drop=True)
extest_df = get_train_test_decode(MAX_LENGTH, EXDATA, lab2id)
test_df=pd.concat([test_df,extest_df],axis=0)
print(len(test_df))
    
from torch.utils.data import Dataset
from transformers import BertModel
from transformers import AutoTokenizer


BATCH_SIZE = 1
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

def free_gpu_cache():                        
    gc.collect()
    torch.cuda.empty_cache()

    
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
    def _get_final_vecs(self, vec, token_num):
        vecs = [vec[:,i+1,:].view(-1, 768) for i in range(MAX_LENGTH)]    # cls ベクトルはいらないので+1 range (0~48) -> (1~49)
        return vecs # vecs[token_num][batch][768]

    def forward(self, input_ids, pred_spans, token_num, span_available_indication_matrix):
        # 順伝播の出力結果は辞書形式なので、必要な値のkeyを指定して取得する
        output = self.bert(input_ids)
        hidden_states = output['hidden_states']

        # 隠れ層からそれぞれ トークンのベクトルを取得する
        vecs = self._get_final_vecs(hidden_states[-1], token_num)  # vecs[MAX_LENGTH][batch][768]

        # make span vec
        span_possible_tuples = [(i, j) for i in range(token_num) for j in range(token_num)]
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
        
        #print(len(outs),len(outs[0]),len(outs[0][0]))
        #for out in outs:
        #    print(out.shape)
        ##outs=[MAX*MAX][batch][label] -> outs=[batch][MAX*MAX][label]
        outs = torch.concat([out.unsqueeze(1) for out in outs],dim=1)   # out.unsqueeze(1)=[batch][1][label]
        #free_gpu_cache()
        results=F.log_softmax(outs, dim=1)
        #results=F.softmax(outs, dim=1)
        
        return results


classifier = BertClassifier()


# 損失関数の設定
loss_function = nn.NLLLoss()

# GPUの設定
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ネットワークをGPUへ送る
classifier.to(device)


from Evaluation.evaluate import cal_label_f1
from Evaluation.evaluate import cal_span_f1
from Evaluation.evaluate import seq_fusion_matrix
from Evaluation.evaluate import span2seq
from Evaluation.evaluate import get_pred_dic_lists
from utils.tools import filter_span_score

if __name__ == "__main__":
    # dataset, dateloader
    dataset = mk_dataset(test_df, 1, 'bert')
    # モデルをロード
    #classifier.load_state_dict(torch.load(f'/media/takeuchi/HDPH-UT/callum/Ohuchi_old/models/srl_ohuchi_incNull_f1_252_best.pth'))
    #classifier.load_state_dict(torch.load(f'/media/takeuchi/HDPH-UT/callum/Ohuchi_old/models/srl_ohuchi_incNull_f1_252_v2_best.pth'))
    classifier.load_state_dict(torch.load(f'/media/takeuchi/HDPH-UT/callum/SpanSRL/Span2/models/srl_span2_emnlp_eachEP.pth'))

    """
    Decode Area
    """
    index_for_correspondence = 0
    predictions = []
    core_labels = ['Arg','Arg0','Arg1', 'Arg2', 'Arg3', 'Arg4', 'Arg5']
    start = time.time()

    with torch.no_grad():
        for c, (batch_features, batch_preds, batch_token_num, batch_args) in enumerate(dataset):   # labelsはバッチ16こに対し128トークンのラベル
            # BATCH サイズは１
            token_num = batch_token_num[0].item()
            preds_span = [batch_preds[0][0].item(), batch_preds[0][1].item()]

            # 使用するスパンの範囲を教える変数（使用可：１， 使用不可：－１）
            # span_available_indication = np.zeros([MAX_LENGTH, MAX_LENGTH])
            # span_available_indication[:,:] = -1
            # token_num = token_num if MAX_LENGTH>=token_num else MAX_LENGTH
            # for span in list(itertools.combinations_with_replacement(np.arange(token_num), 2)):
            #     if (span[1] - span[0]) <= MAX_ARGUMENT_SEQUENCE_LENGTH:
            #         span_available_indication[span[0]][span[1]] = 1
            #print(np.count_nonzero((span_available_indication >= 1)))

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
            #out = out.to('cpu')
            #out = [tmp.to('cpu').detach().numpy().copy() for tmp in out] # cpu に移し，numpyに変換するためにdetach. 
            #out = np.array(out).transpose(1, 0, 2) # out[batch][MAX*MAX][labels]

            # prepare input for decode
            for scores, pred_span, token_num in zip(out, batch_preds, batch_token_num):   # batchから取り出し， 各文章毎に行う．
                span_score_lists, used_cores, confirmed_spans = [], [], []
                # 各スパンの各ラベルごとのscoreを入れていく
                token_num = MAX_LENGTH if token_num > MAX_LENGTH else token_num
                span_possible_tuples = [(i, j) for i in range(token_num) for j in range(token_num)]
                reshaped_scores = scores.reshape(token_num, token_num, -1)
                pred_span_scores = reshaped_scores[pred_span[0], pred_span[1], :]
                for i,j in span_possible_tuples:
                    if span_available_indication[i][j] != -1:
                        for label, score in zip( list(id2lab.values()), reshaped_scores[i][j] ):
                            span_score_lists.append([i, j, label, score])

                # filterにかけた後，貪欲法で値の大きいものから取っていく
                prev_num = len(span_score_lists)
                #span_score_lists = filter_span_score_for_batch_1(span_score_lists, pred_span, token_num)
                span_score_lists = filter_span_score(span_score_lists, pred_span, token_num, pred_span_scores, lab2id)
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
                free_gpu_cache()
    print('----------------------------------------')
    span_answers = test_df['span_answers'].to_list()
    lf1 = cal_label_f1(predictions, span_answers, lab2id)
    print(lf1)
    lf1.to_csv('span2_label.csv')
    sf1 = cal_span_f1(predictions, span_answers, lab2id, MAX_LENGTH)
    print(sf1)
    sf1.to_csv('span2_span.csv')
    #fm = seq_fusion_matrix(predictions, span_answers, lab2id)
    #print(fm)
    #fm.to_csv('span2_FusionMatrix.csv')
    
    # 解析用データ作成
    pred_seq = [span2seq(p, int(num_of_tokens)) for p, num_of_tokens in zip(predictions, test_df['num_of_tokens'].to_list())]
    test_df['BIOtag'] = pred_seq
    
    pred_dic_lists, match_count_list, args_count_list = get_pred_dic_lists(predictions, span_answers, lab2id)
    test_df['pred_arg'] = pred_dic_lists
    test_df['match_count'] = match_count_list
    test_df['args_num'] = args_count_list
    test_df['predict_num'] = [len(dic) for dic in pred_dic_lists]
    test_df['args'] = test_df['args2']
    test_df = test_df[['sentence', 'sentenceID', 'predicate','args', 'BIOtag', 'pred_arg', 'match_count', 'args_num', 'predict_num']]
    test_df.to_json('data_for_analy.json',orient='records', lines=True,force_ascii=False)
    
    print(f"Time : {time.time() - start}")
    #report.to_csv("../../report/enc4_incO_soft_soft_243.csv", encoding='utf-8_sig')
