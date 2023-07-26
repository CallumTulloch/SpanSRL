import torch
import numpy as np

"""
SRL
"""
def bert_tokenizer_bert(wakati, pred, max_length, max_token, pred_sep_num, tokenizer): 
    #wakati = normalize(wakati)
    token_num = len(wakati.split(' '))
    pred_token_num = len(pred['surface'].split(' '))
    pred_sep_num =  pred_token_num + 2 if pred_token_num <= pred_sep_num-2 else pred_sep_num

    if token_num <= max_length:
        front_padding_num = max_length - token_num
        back_padding_num = max_token - max_length - pred_sep_num - 1 # -1 は cls
        tokens = ['[CLS]'] + wakati.split(' ') + ['[PAD]']*front_padding_num + ['[SEP]'] + pred['surface'].split(' ')[:pred_token_num] + ['[SEP]'] + ['[PAD]']*back_padding_num
    else:
        padding_num = max_token - max_length - pred_sep_num - 1 # padding >= 0 は保証
        tokens = ['[CLS]'] + wakati.split(' ')[:max_length] + ['[SEP]'] + pred['surface'].split(' ')[:pred_token_num] + ['[SEP]'] + ['[PAD]']*padding_num
    ids = np.array(tokenizer.convert_tokens_to_ids(tokens))
    #print(tokenizer.convert_ids_to_tokens(ids))
    #print(len(ids))
    return ids


def preprocess(sentence_s, args_s, predicates_s, label_id, token_num, max_length, max_token, pred_sep_num, tokenizer):
    sentences = []
    pred_span = []
    for sent, pred in zip(sentence_s, predicates_s):
        sentences.append(bert_tokenizer_bert(sent, pred, max_length, max_token, pred_sep_num, tokenizer))
        pred_span.append((pred['word_start'],pred['word_end']))
    sentences = torch.tensor(sentences, dtype=torch.long)
    args = [torch.tensor(args) for args in args_s.tolist()]
    pred_span = torch.tensor(pred_span, dtype=torch.long)
    label_id = torch.tensor(list(label_id), dtype=torch.long)
    token_num = torch.tensor(list(token_num), dtype=torch.long)
    return [sentences, args, label_id, pred_span, token_num]


def mk_dataset(df, batch_size, max_length, max_token, pred_sep_num, tokenizer, sort=True):
    if sort:
        df.sort_values(by='num_of_tokens',inplace = True, ascending=True)
    batch_set = [df.iloc[i*batch_size : (i+1)*batch_size] for i in range(int(len(df)/batch_size))]
    batch_set = [preprocess(set['sentence'], set['args'], set['predicate'], set['label_id'], set['num_of_tokens'], max_length, max_token, pred_sep_num, tokenizer) for set in batch_set]
    return batch_set

"""
Decode
"""
def preprocess_decode(sentence_s, predicates_s, token_num, max_length, max_token, pred_sep_num, tokenizer):
    sentences = []
    pred_span = []
    for sent, pred in zip(sentence_s, predicates_s):
        sentences.append(bert_tokenizer_bert(sent, pred, max_length, max_token, pred_sep_num, tokenizer))
        pred_span.append((pred['word_start'],pred['word_end']))
    sentences = torch.tensor(sentences, dtype=torch.long)
    pred_span = torch.tensor(pred_span, dtype=torch.long)
    token_num = torch.tensor(list(token_num), dtype=torch.long)
    return [sentences, pred_span, token_num]


def mk_dataset_decode(df, batch_size, max_length, max_token, pred_sep_num, tokenizer, sort=True):
    if sort:
        df.sort_values(by='num_of_tokens',inplace = True, ascending=True)
    batch_set = [df.iloc[i*batch_size : (i+1)*batch_size] for i in range(int(len(df)/batch_size))]
    batch_set = [preprocess_decode(set['sentence'], set['predicate'], set['num_of_tokens'], max_length, max_token, pred_sep_num, tokenizer) for set in batch_set]
    return batch_set