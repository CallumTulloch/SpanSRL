import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, GPTNeoXPreTrainedModel, GPTNeoXModel
#from transformers import GPTNeoXJapaneseConfig, GPTNeoXJapaneseModel, GPTNeoXJapaneseForCausalLM

#RinnaClassifierDecodeIntegrated.from_pretrained()
#class RinnaClassifierDecodeIntegrated(GPTNeoXJapaneseModel):# ここはpretraind_model的なのを継承sルウ．
#    def __init__(self, config):
#        super().__init__(config)
#        self.config = config
        
    
class RinnaClassifierDecodeIntegrated(nn.Module):# ここはpretraind_model的なのを継承sルウ．
    def __init__(self, output_layer_dim, max_length, device):
        super(RinnaClassifierDecodeIntegrated, self).__init__()
        self.config = AutoConfig.from_pretrained("rinna/japanese-gpt-neox-3.6b")
        
        # 各種変数定義
        self.output_layer_dim = output_layer_dim
        self.max_length = max_length
        self.device = device
        
        # Rinna
        self.rinna = AutoModel.from_pretrained("rinna/japanese-gpt-neox-3.6b") # attention 受け取り，隠れ層取
        # Final Layers
        self.my_linear = nn.Linear(2816*2+1, 2816)    # input, output
        self.relu = nn.ReLU()
        self.my_linear2 = nn.Linear(2816, output_layer_dim)    # input, output

        # 重み初期化処理
        nn.init.normal_(self.my_linear.weight, std=0.02)   # std 正規分布の標準偏差
        nn.init.normal_(self.my_linear2.weight, std=0.02)
        nn.init.normal_(self.my_linear.bias, 0)
        nn.init.normal_(self.my_linear2.bias, 0)

    # ベクトルを取得する用の関数
    def _get_sent_token_vecs(self, vec, token_num):
        vecs = [vec[:,i+1,:].view(-1, 2816) for i in range(token_num)]  # BOSは必要ないから +1. range(token_num) は 0~
        return vecs # vecs[MAX_LENGTH][batch][2816]
    
    #@profile
    def forward(self, input_ids, pred_spans, token_num, span_available_indication_matrix, usage):
        if usage == 'decode':
                token_num = self.max_length if token_num > self.max_length else token_num

        # 順伝播の出力結果は辞書形式なので、必要な値のkeyを指定して取得する
        output = self.rinna(input_ids)
        hidden_states = output[0]   # これは既に最終層
        #print(hidden_states.shape)

        # 隠れ層からそれぞれ トークンのベクトルを取得する
        vecs = self._get_sent_token_vecs(hidden_states, token_num)  # vecs[MAX_LENGTH][batch][2816]
        #print(len(vecs))
        #print(vecs[0].shape)
        #print(input_ids[0], token_num, vecs[0].shape, len(vecs))

        # make span vec
        if usage == 'decode':
            span_possible_tuples = [(i, j) for i in range(token_num) for j in range(token_num)]
        elif usage == 'train':
            span_possible_tuples = [(i, j) for i in range(self.max_length) for j in range(self.max_length)]

            
        span_vec_list, pred_inside_span_batch =[], []
        for start, end in pred_spans:
            pred_inside_spans = list(itertools.combinations_with_replacement(torch.arange(start, end+1), 2))    # 重複組み合わせ
            pred_inside_spans.remove((start, end))
            pred_inside_span_batch.append(torch.tensor(pred_inside_spans).to(self.device))    # pred_inside_span_batch[batch][pred_inside_spans]
        
        # vec_combination の組に対する操作: predicate -> [vec_combi,2], predicate_inside -> [vec_combi, 1], not_predicate -> [vec_combi, 0]
        zeros = torch.zeros((2816*2+1)*len(input_ids)).reshape(len(input_ids), (2816*2+1)).to(self.device)
        #zeros2 = torch.zeros(self.output_layer_dim*len(input_ids)).reshape(len(input_ids), self.output_layer_dim).to(self.device)
        for c, (i, j) in enumerate(span_possible_tuples):   # span_possible_tuples[MAX_LENGTH][MAX_LENGTH]
            if span_available_indication_matrix[i,j] >= 1:   # 使用する範囲かどうか
                # predicate への操作
                vec_i, vec_j = vecs[i], vecs[j]     # vec_i[batch][2816]

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
                span_vec_list.append(zeros)
                continue

        # 全結合層で追加した層用に次元を変換. span_vecを通す．
        outs = [self.my_linear(vec) if span_available_indication_matrix[i,j] == 1 else -1
                for vec,(i,j) in zip(span_vec_list, span_possible_tuples) ]   # vec はそれぞれのspanに対応したembedding(バッチのサイズあることに注意)
                
        outs = [self.relu(out) if span_available_indication_matrix[i,j] == 1 else -1
                for out,(i,j) in zip(outs, span_possible_tuples) ]
        
        outs = [self.my_linear2(out) if span_available_indication_matrix[i,j] == 1 else -1
                for out,(i,j) in zip(outs, span_possible_tuples) ]
        
        # 各層でマスクを適用
        #print('here')
        #span_vec_list = torch.cat([vec.unsqueeze(0) for vec in span_vec_list], dim=0)   # vec.unsqueeze(0) -> (1, batch, 2816)
        #mask = span_available_indication_matrix.view(-1, 1)  # または span_available_indication_matrix.reshape(num_of_span, 1)
        #mask = mask.unsqueeze(-1)  # 最後の次元を追加

        # 各層でマスクを適用
        #outs = self.my_linear(span_vec_list)
        #outs = outs * mask  # maskの適用
        #outs = self.relu(outs)
        ##outs = outs * mask  # maskの適用
        #outs = self.my_linear2(outs)
        #outs = outs * mask  # maskの適用

        
        if usage == 'train':
            results = [F.log_softmax(out, dim=1) for out,(i,j) in zip(outs, span_possible_tuples) if span_available_indication_matrix[i,j] == 1]
            results = torch.cat([torch.unsqueeze(out,dim=0) for out in results], dim=0).permute(1,0,2) # [batch][max][label]
        elif usage == 'decode':
            results = [F.log_softmax(out, dim=1) if span_available_indication_matrix[i,j] == 1 else torch.ones(1, self.output_layer_dim)*-1  # 後でpermute使うから同じ次元のものにしておく
                      for out,(i,j) in zip(outs, span_possible_tuples) ]
        return results


# 定義の仕方を少し変える
class RinnaClassifierDecodeIntegrated2(GPTNeoXPreTrainedModel):# ここはpretraind_model的なのを継承sルウ．
    def __init__(self, config, output_layer_dim, max_length):
        super().__init__(config)
        
        # 各種変数定義
        self.output_layer_dim = output_layer_dim
        self.max_length = max_length
        
        # Rinna
        self.rinna = GPTNeoXModel(config)
        # Final Layers
        self.my_linear = nn.Linear(2816*2+1, 2816)    # input, output
        self.relu = nn.ReLU()
        self.my_linear2 = nn.Linear(2816, output_layer_dim)    # input, output

        # 重み初期化処理
        nn.init.normal_(self.my_linear.weight, std=0.02)   # std 正規分布の標準偏差
        nn.init.normal_(self.my_linear2.weight, std=0.02)
        nn.init.normal_(self.my_linear.bias, 0)
        nn.init.normal_(self.my_linear2.bias, 0)

    # ベクトルを取得する用の関数
    def _get_sent_token_vecs(self, vec, token_num):
        vecs = [vec[:,i+1,:].view(-1, 2816) for i in range(token_num)]  # BOSは必要ないから +1. range(token_num) は 0~
        return vecs # vecs[MAX_LENGTH][batch][2816]
    
    #@profile
    def forward(self, input_ids, pred_spans, token_num, span_available_indication_matrix, usage, device):
        if usage == 'decode':
                token_num = self.max_length if token_num > self.max_length else token_num

        # 順伝播の出力結果は辞書形式なので、必要な値のkeyを指定して取得する
        output = self.rinna(input_ids)
        hidden_states = output[0]   # これは既に最終層
        #print(hidden_states.shape)

        # 隠れ層からそれぞれ トークンのベクトルを取得する
        vecs = self._get_sent_token_vecs(hidden_states, token_num)  # vecs[MAX_LENGTH][batch][2816]
        #print(len(vecs))
        #print(vecs[0].shape)
        #print(input_ids[0], token_num, vecs[0].shape, len(vecs))

        # make span vec
        if usage == 'decode':
            span_possible_tuples = [(i, j) for i in range(token_num) for j in range(token_num)]
        elif usage == 'train':
            span_possible_tuples = [(i, j) for i in range(self.max_length) for j in range(self.max_length)]

            
        span_vec_list, pred_inside_span_batch =[], []
        for start, end in pred_spans:
            pred_inside_spans = list(itertools.combinations_with_replacement(torch.arange(start, end+1), 2))    # 重複組み合わせ
            pred_inside_spans.remove((start, end))
            pred_inside_span_batch.append(torch.tensor(pred_inside_spans).to(device))    # pred_inside_span_batch[batch][pred_inside_spans]
        
        # vec_combination の組に対する操作: predicate -> [vec_combi,2], predicate_inside -> [vec_combi, 1], not_predicate -> [vec_combi, 0]
        zeros = torch.zeros((2816*2+1)*len(input_ids)).reshape(len(input_ids), (2816*2+1)).to(device)
        #zeros2 = torch.zeros(self.output_layer_dim*len(input_ids)).reshape(len(input_ids), self.output_layer_dim).to(self.device)
        for c, (i, j) in enumerate(span_possible_tuples):   # span_possible_tuples[MAX_LENGTH][MAX_LENGTH]
            if span_available_indication_matrix[i,j] >= 1:   # 使用する範囲かどうか
                # predicate への操作
                vec_i, vec_j = vecs[i], vecs[j]     # vec_i[batch][2816]

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
                pred_indications = torch.tensor(pred_indications).reshape(len(pred_inside_span_batch),1).to(self.device)
                span_vec_list.append( torch.cat([vec_i, vec_j, pred_indications], 1) )
            else:
                span_vec_list.append(zeros)
                continue

        # 全結合層で追加した層用に次元を変換. span_vecを通す．
        outs = [self.my_linear(vec) if span_available_indication_matrix[i,j] == 1 else -1
                for vec,(i,j) in zip(span_vec_list, span_possible_tuples) ]   # vec はそれぞれのspanに対応したembedding(バッチのサイズあることに注意)
                
        outs = [self.relu(out) if span_available_indication_matrix[i,j] == 1 else -1
                for out,(i,j) in zip(outs, span_possible_tuples) ]
        
        outs = [self.my_linear2(out) if span_available_indication_matrix[i,j] == 1 else -1
                for out,(i,j) in zip(outs, span_possible_tuples) ]
        
        # 各層でマスクを適用
        #print('here')
        #span_vec_list = torch.cat([vec.unsqueeze(0) for vec in span_vec_list], dim=0)   # vec.unsqueeze(0) -> (1, batch, 2816)
        #mask = span_available_indication_matrix.view(-1, 1)  # または span_available_indication_matrix.reshape(num_of_span, 1)
        #mask = mask.unsqueeze(-1)  # 最後の次元を追加

        # 各層でマスクを適用
        #outs = self.my_linear(span_vec_list)
        #outs = outs * mask  # maskの適用
        #outs = self.relu(outs)
        ##outs = outs * mask  # maskの適用
        #outs = self.my_linear2(outs)
        #outs = outs * mask  # maskの適用

        
        if usage == 'train':
            results = [F.log_softmax(out, dim=1) for out,(i,j) in zip(outs, span_possible_tuples) if span_available_indication_matrix[i,j] == 1]
            results = torch.cat([torch.unsqueeze(out,dim=0) for out in results], dim=0).permute(1,0,2) # [batch][max][label]
        elif usage == 'decode':
            results = [F.log_softmax(out, dim=1) if span_available_indication_matrix[i,j] == 1 else torch.ones(1, self.output_layer_dim)*-1  # 後でpermute使うから同じ次元のものにしておく
                      for out,(i,j) in zip(outs, span_possible_tuples) ]
        return results