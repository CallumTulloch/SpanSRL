import sys
sys.path.append('../')
sys.path.append('../../')
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("rinna/japanese-gpt-neox-3.6b", use_fast=False)
#tokenizer = AutoTokenizer.from_pretrained("cl-tohoku/bert-base-japanese-v2", use_fast=False)
# print(tokenizer.special_tokens_map)
# print("bos_token :", tokenizer.bos_token, ",", tokenizer.bos_token_id)
# print("eos_token :", tokenizer.eos_token, ",", tokenizer.eos_token_id)
# print("unk_token :", tokenizer.unk_token, ",", tokenizer.unk_token_id)
# print("pad_token :", tokenizer.pad_token, ",", tokenizer.pad_token_id)
sent = '腹 が た ってきた 。'
sent = ''.join(sent.split())

token_ids_list = tokenizer(sent)['input_ids']
print(token_ids_list)

token_list = tokenizer.tokenize(sent)
print(token_list)

token_ids_list2 = [tokenizer(token)['input_ids'][:-1] for token in token_list]
print(token_ids_list2)

token_ids_list3 = [tokenizer.convert_tokens_to_ids(token) for token in token_list]
print(token_ids_list3)

print(tokenizer('「輩」という漢字')['input_ids'])


print(tokenizer('ってきた'))
print(tokenizer('って')['input_ids'][:-1], tokenizer('きた')['input_ids'][:-1])

print( tokenizer.convert_tokens_to_ids(['<0xE5>', '<0x97>', '<0x84>']))
print( tokenizer.convert_ids_to_tokens([236, 158, 139]))
print( tokenizer.tokenize('切磋琢磨する'))