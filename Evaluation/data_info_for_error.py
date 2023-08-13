import pandas as pd


DATAPATH = '/media/takeuchi/HDPH-UT/callum/SpanSRL/Data/common_data_v2_bert.json'
with open(DATAPATH, 'r', encoding="utf-8_sig") as json_file:
    data = pd.read_json(json_file)

# 上位10種類のArgを見つける
dic = {}
for index, row in data.iterrows():
    for arg in row['args']:
        try:
            dic[arg['argrole']] += 1
        except:
            dic[arg['argrole']] = 1
top10_args = sorted(dic.items(), key=lambda x:x[1], reverse=True)[:10]
under10_args = sorted(dic.items(), key=lambda x:x[1], reverse=False)[:11]
print(top10_args, '\n')
print([arg[0] for arg in under10_args])

# 混合行列の成型
df = pd.read_csv("span2_FusionMatrix.csv", index_col=0)
#args_col = sorted([arg[0] for arg in top10_args])
args = [arg[0] for arg in under10_args]
args.remove('Arg')
args_col = sorted(args)
df = df.loc[args_col, args_col]#
df['sum'] = df.apply(lambda x:sum(x), axis=1)
df.loc['sum',:] = df.apply(lambda x:sum(x), axis=0)
df.to_csv('new_FusionMatrix.csv')
print(df)#

