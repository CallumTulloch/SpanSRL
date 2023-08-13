import torch
import numpy as np

"""
SRL
"""
def tokenize(wakati, pred, max_length, max_token, pred_sep_num, tokenizer): 
    #print(wakati)
    token_num = len(wakati.split(' '))
    pred_token_num = len(pred['surface'].split(' '))
    pred_sep_num =  pred_token_num + 2 if pred_token_num <= pred_sep_num-2 else pred_sep_num

    if token_num <= max_length:
        front_padding_num = max_length - token_num
        back_padding_num = max_token - max_length - pred_sep_num - 2 # -1 は cls
        tokens = ['<s>'] + wakati.split(' ') + ['</s>'] + ['[PAD]']*front_padding_num + ['[SEP]'] + pred['surface'].split(' ')[:pred_token_num] + ['</s>'] + ['[PAD]']*back_padding_num
    else:   # TODO:ここ間違えてるぽい. 現在はelseに行かないから問題はない
        padding_num = max_token - max_length - pred_sep_num - 1 # padding >= 0 は保証
        tokens = ['<s>'] + wakati.split(' ')[:max_length] + ['</s>'] + ['[SEP]'] + pred['surface'].split(' ')[:pred_token_num] + + ['</s>'] + ['[PAD]']*padding_num
    assert len(tokens) == max_token
    ids = np.array(tokenizer.convert_tokens_to_ids(tokens))
    #print(tokenizer.convert_ids_to_tokens(ids))
    #print(len(ids))
    return ids


#def preprocess(sentence_s, predicates_s, label_id, token_num, max_length, max_token, pred_sep_num, tokenizer):
#    sentences = []
#    pred_span = []
#    for sent, pred in zip(sentence_s, predicates_s):
#        sentences.append(tokenize(sent, pred, max_length, max_token, pred_sep_num, tokenizer))
#        pred_span.append((pred['word_start'],pred['word_end']))
#    sentences = torch.tensor(sentences, dtype=torch.long)
#    pred_span = torch.tensor(pred_span, dtype=torch.long)
#    label_id = torch.tensor(list(label_id), dtype=torch.long)
#    token_num = torch.tensor(list(token_num), dtype=torch.long)
#    return [sentences, label_id, pred_span, token_num]
#
#
#def mk_dataset(df, batch_size, max_length, max_token, pred_sep_num, tokenizer, sort=True):
#    if sort:
#        df.sort_values(by='num_of_tokens',inplace = True, ascending=True)
#    batch_set = [df.iloc[i*batch_size : (i+1)*batch_size] for i in range(int(len(df)/batch_size))]
#    batch_set = [preprocess(set['sentence'], set['predicate'], set['label_id'], set['num_of_tokens'], max_length, max_token, pred_sep_num, tokenizer) for set in batch_set]
#    return batch_set


def preprocess(sentence_s, predicates_s, token_num, args_s, max_length, max_token, pred_sep_num, tokenizer):
    sentences = []
    pred_span = []
    for sent, pred in zip(sentence_s, predicates_s):
        sentences.append(tokenize(sent, pred, max_length, max_token, pred_sep_num, tokenizer))
        pred_span.append((pred['word_start'],pred['word_end']))
    sentences = torch.tensor(sentences, dtype=torch.long)
    pred_span = torch.tensor(pred_span, dtype=torch.long)
    token_num = torch.tensor(list(token_num), dtype=torch.long)
    #print(args_s)
    args = [torch.tensor(args) for args in args_s.tolist()]
    return [sentences, pred_span, token_num, args]

def mk_dataset(df, batch_size, max_length, max_token, pred_sep_num, tokenizer, sort=True):
    if sort:
        df.sort_values(by='num_of_tokens',inplace = True, ascending=True)
    batch_set = [df.iloc[i*batch_size : (i+1)*batch_size] for i in range(int(len(df)/batch_size))]
    batch_set = [preprocess(set['sentence'], set['predicate'], set['num_of_tokens'], set['args'], 
                            max_length, max_token, pred_sep_num, tokenizer) for set in batch_set]
    return batch_set