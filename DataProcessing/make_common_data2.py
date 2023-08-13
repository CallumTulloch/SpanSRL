import pandas as pd
import json


with open('../Data/common_data_v2_rinna.json', 'r', encoding="utf-8_sig") as json_file:
    df1 = pd.read_json(json_file)
    print(len(df1))
df2 = pd.read_json('../Data/temp.json')
# 'abs_id' をインデックスに設定
df1.set_index('abs_id', inplace=True)
df2.set_index('abs_id', inplace=True)

# df2 で df1 を更新
df1.update(df2)

# 必要に応じてインデックスをリセット
df1.reset_index(inplace=True)
    

# 変換
#df1_sorted.to_json(f"../Data/common_data_v2_bert.json",orient='records',force_ascii=False)
df1.to_json(f"../Data/common_data_v2_rinna3.json",orient='records',force_ascii=False)
print(len(df1))
    
with open(f"../Data/common_data_v2_rinna3.json") as f:
    data = json.load(f)
with open(f"../Data/common_data_v2_rinna3.json", mode="w") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)