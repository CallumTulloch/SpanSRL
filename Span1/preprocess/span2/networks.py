import time
import itertools
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gc
from transformers import BertModel
from transformers import AutoTokenizer

class BertClassifierDecodeOhuchi(nn.Module):
    def __init__(self, output_layer_dim, max_length, device):
        super(BertClassifierDecodeOhuchi, self).__init__()
        # 各種変数定義
        self.output_layer_dim = output_layer_dim
        self.max_length = max_length
        self.device = device
        
        # BERT
        self.bert = BertModel.from_pretrained('cl-tohoku/bert-base-japanese-v2', output_attentions=True, output_hidden_states=True) # attention 受け取り，隠れ層取得
        # Final Layers
        self.linear = nn.Linear(768*2+1, 770)    # input, output
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(770, self.output_layer_dim)    # input, output

        # 重み初期化処理
        nn.init.normal_(self.linear.weight, std=0.02)   # std 正規分布の標準偏差
        nn.init.normal_(self.linear2.weight, std=0.02)
        nn.init.normal_(self.linear.bias, 0)
        nn.init.normal_(self.linear2.bias, 0)

    # ベクトルを取得する用の関数
    def _get_final_vecs(self, vec):
        vecs = [vec[:,i+1,:].view(-1, 768) for i in range(self.max_length)]    # cls ベクトルはいらないので+1 range (0~48) -> (1~49)
        return vecs # vecs[token_num][batch][768]

    def forward(self, input_ids, pred_spans, token_num, span_available_indication_matrix):
        # 順伝播の出力結果は辞書形式なので、必要な値のkeyを指定して取得する
        output = self.bert(input_ids)
        hidden_states = output['hidden_states']

        # 隠れ層からそれぞれ トークンのベクトルを取得する
        vecs = self._get_final_vecs(hidden_states[-1])  # vecs[self.max_length][batch][768]

        # make span vec
        span_possible_tuples = [(i, j) for i in range(self.max_length) for j in range(self.max_length)]
        span_vec_list, pred_inside_span_batch =[], []
        for start, end in pred_spans:
            pred_inside_spans = list(itertools.combinations_with_replacement(torch.arange(start, end+1), 2))    # 重複組み合わせ
            pred_inside_spans.remove((start, end))
            pred_inside_span_batch.append(torch.tensor(pred_inside_spans).to(self.device))    # pred_inside_span_batch[batch][pred_inside_spans]
        
        # vec_combination の組に対する操作: predicate -> [vec_combi,2], predicate_inside -> [vec_combi, 1], not_predicate -> [vec_combi, 0]
        zeros = torch.zeros(self.output_layer_dim*len(input_ids)).reshape(len(input_ids), self.output_layer_dim).to(self.device)
        for c, (i, j) in enumerate(span_possible_tuples):   # span_possible_tuples[MAX_LENGTH][MAX_LENGTH]
            if span_available_indication_matrix[i,j] >= 1:   # 使用する範囲かどうか
                # predicate への操作
                vec_i, vec_j = vecs[i], vecs[j]     # vec_i[batch][768]

                pred_indications = []
                target_span = torch.tensor([i,j]).to(self.device)
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
                pred_indications = torch.tensor(pred_indications).reshape(len(pred_inside_span_batch),1).to(self.device)
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