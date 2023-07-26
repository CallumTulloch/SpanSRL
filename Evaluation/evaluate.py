import numpy as np
import pandas as pd

def cal_accuracy(ys, xs, label_names):  # 正解ラベル，予想ラベル，ラベル名
    array = np.zeros(3*len(label_names)).reshape(len(label_names), 3)
    #print(len(xs), len(ys))

    for x, y in zip(xs, ys):
        #print(type(y), y)
        array[y][2] += 1     # support: +1
        if x == y:
            array[x][0] += 1 # correct: +1
        else:
            array[x][1] += 1 # wrong: +1
    
    df = pd.DataFrame(array, columns=['correct_num', 'wrong_num', 'support'], index=label_names)
    df['recall'] = df['correct_num'] / df['support']
    df['precision'] = df['correct_num'] / (df['correct_num'] + df['wrong_num'])
    df['f1_score'] = (2 * df['recall'] * df['precision']) / (df['recall'] + df['precision'])

    acc = df['correct_num'].sum() / df['support'].sum()
    if 'B-V' in label_names:
        acc_ex_o = (df['correct_num'].sum() - df.loc['O', 'correct_num']) / (df['support'].sum() - df.loc['O', 'support'])
        acc_ex_all = (df['correct_num'].sum() - df.loc['O', 'correct_num'] - df.loc['B-V', 'correct_num'] - df.loc['I-V', 'correct_num']) \
                  / (df['support'].sum() - df.loc['O', 'support'] - df.loc['B-V', 'support'] - df.loc['I-V', 'support'])

    elif 'F-P' in label_names:
        acc_ex_o = (df['correct_num'].sum() - df.loc['O', 'correct_num']) \
                 / (df['support'].sum() - df.loc['O', 'correct_num'])
        acc_ex_all = (df['correct_num'].sum() - df.loc['O', 'correct_num'] - df.loc['F-P', 'correct_num'] - df.loc['F-A', 'correct_num'] - df.loc['V', 'correct_num'] - df.loc['N', 'correct_num']) \
                  / (df['support'].sum() - df.loc['O', 'support'] - df.loc['F-P', 'support'] - df.loc['F-A', 'support'] - df.loc['V', 'support'] - df.loc['N', 'support'])

    df = df.round(3)
    df.loc['acc'] = acc
    df.loc['acc_ex_o'] = acc_ex_o
    df.loc['acc_ex_all'] = acc_ex_all
    df.loc['macro_ave'] = df['f1_score'].sum() / len(label_names)
    return df


def cal_label_f1(xs, ys, lab2id):
    for lab in ['F-A', 'F-P', 'V', 'O', 'N']:
        try:
            del lab2id[lab]
        except:
            continue
    print(lab2id)
    array = np.zeros(len(lab2id)*5).reshape(len(lab2id), 5)
    for span_pre_sent, span_ans_sent in zip(xs, ys):
        # データ成型, 予想spanのラベルにフラグメントと述語は含まれない．
        span_label_pre_lists = [tuple(span[:3]) for span in span_pre_sent if (span[2] != 'O') and (span[2] != 'N')]   # pre にはOが含まれる. correct_spanの時にはNも含まれる
        span_label_ans_lists = [(span[1], span[2], span[0]) for span in span_ans_sent if (span[0] != 'V') and (span[0] != 'O')] # ans にはVが含まれる
        for y_start, y_end, y_lab in span_label_ans_lists:
            array[lab2id[y_lab]][4] += 1   # support + 1

        for pre in span_label_pre_lists:
            if pre in span_label_ans_lists:
                array[lab2id[pre[2]]][0] += 1 # correct + 1
                span_label_ans_lists.remove(pre)        
            else:
                array[lab2id[pre[2]]][1] += 1 # wrong_pred + 1
        
        for ans in span_label_ans_lists:
            array[lab2id[ans[2]]][3] += 1 # wrong_missed + 1

    array = array.T
    array[2] = array[0] + array[1]
    array = array.T
    # 各ラベルの評価
    df = pd.DataFrame(array, columns=['correct_num', 'wrong_num', 'predict_num', 'missed_num', 'support'], index=list(lab2id.keys()))
    df['precision'] = df['correct_num'] / df['predict_num']
    #df['precision_v2'] = df['correct_num'] / (df['predict_num'] + df['missed_num'])
    df['recall'] = df['correct_num'] / df['support']
    df['f1'] = 2*df['precision']*df['recall'] / (df['precision'] + df['recall'])
    #df['f1_v2'] = 2*df['precision_v2']*df['recall'] / (df['precision_v2'] + df['recall'])
    
    # 以下全体
    df.loc['sum'] = df.sum()
    if df.loc['sum', 'predict_num'] == 0:
        df.loc['sum', 'predict_num'] = 1
    df.loc['precision','correct_num'] = df.loc['sum', 'correct_num'] / df.loc['sum', 'predict_num']
    #df.loc['precision_v2','correct_num'] = df.loc['sum', 'correct_num'] / (df.loc['sum', 'predict_num'] + df.loc['sum', 'missed_num'])
    df.loc['recall','correct_num'] = df.loc['sum', 'correct_num'] / df.loc['sum', 'support']
    df.loc['f1','correct_num'] = 2*df.loc['precision','correct_num']*df.loc['recall','correct_num'] / (df.loc['precision','correct_num'] + df.loc['recall','correct_num'])
    df.loc['f1_macro','correct_num'] = df['f1'].sum() / len(lab2id)
    #df.loc['f1_v2','correct_num'] = 2*df.loc['precision_v2','correct_num']*df.loc['recall','correct_num'] / (df.loc['precision_v2','correct_num'] + df.loc['recall','correct_num'])
    df = df.round(4)
    return df

def cal_span_f1(xs, ys, lab2id, max_length):
    for lab in ['F-A', 'F-P', 'V', 'O', 'N']:
        try:
            del lab2id[lab]
        except:
            continue
    print(lab2id)
    array = np.zeros(max_length*5).reshape(max_length, 5)
    for span_pre_sent, span_ans_sent in zip(xs, ys):
        # データ成型, 予想spanのラベルにフラグメントと述語は含まれない．
        span_label_pre_lists = [tuple(span[:3]) for span in span_pre_sent if (span[2] != 'O') and (span[2] != 'N')]   # pre にはOが含まれる. correct_spanの時にはNも含まれる
        span_label_ans_lists = [(span[1], span[2], span[0]) for span in span_ans_sent if (span[0] != 'V') and (span[0] != 'O')] # ans にはVが含まれる
        for y_start, y_end, y_lab in span_label_ans_lists:
            array[y_end-y_start][4] += 1   # support + 1

        for pre in span_label_pre_lists:
            if pre in span_label_ans_lists:
                array[pre[1]-pre[0]][0] += 1 # correct + 1
                span_label_ans_lists.remove(pre)        
            else:
                array[pre[1]-pre[0]][1] += 1 # wrong_pred + 1
        
        for ans in span_label_ans_lists:
            array[ans[1]-ans[0]][3] += 1 # wrong_missed + 1

    array = array.T
    array[2] = array[0] + array[1]
    array = array.T
    # 各ラベルの評価
    df = pd.DataFrame(array, columns=['correct_num', 'wrong_num', 'predict_num', 'missed_num', 'support'], index=np.arange(1,max_length+1))
    df['precision'] = df['correct_num'] / df['predict_num']
    #df['precision_v2'] = df['correct_num'] / (df['predict_num'] + df['missed_num'])
    df['recall'] = df['correct_num'] / df['support']
    df['f1'] = 2*df['precision']*df['recall'] / (df['precision'] + df['recall'])
    #df['f1_v2'] = 2*df['precision_v2']*df['recall'] / (df['precision_v2'] + df['recall'])
    
    # 以下全体
    df.loc['sum'] = df.sum()
    if df.loc['sum', 'predict_num'] == 0:
        df.loc['sum', 'predict_num'] = 1
    df.loc['precision','correct_num'] = df.loc['sum', 'correct_num'] / df.loc['sum', 'predict_num']
    #df.loc['precision_v2','correct_num'] = df.loc['sum', 'correct_num'] / (df.loc['sum', 'predict_num'] + df.loc['sum', 'missed_num'])
    df.loc['recall','correct_num'] = df.loc['sum', 'correct_num'] / df.loc['sum', 'support']
    df.loc['f1','correct_num'] = 2*df.loc['precision','correct_num']*df.loc['recall','correct_num'] / (df.loc['precision','correct_num'] + df.loc['recall','correct_num'])
    #df.loc['f1_macro','correct_num'] = df['f1'].sum() / len(lab2id)
    #df.loc['f1_v2','correct_num'] = 2*df.loc['precision_v2','correct_num']*df.loc['recall','correct_num'] / (df.loc['precision_v2','correct_num'] + df.loc['recall','correct_num'])
    df = df.round(4)
    return df


def cal_span_only_f1(xs, ys, max_length):
    array = np.zeros(max_length*5).reshape(max_length, 5)
    for span_pre_sent, span_ans_sent in zip(xs, ys):
        # データ成型, 予想spanのラベルにフラグメントと述語は含まれない．
        span_label_pre_lists = [tuple(span[:2]) for span in span_pre_sent if (span[2] != 'O') and (span[2] != 'N')]   # pre にはOが含まれる. correct_spanの時にはNも含まれる
        span_label_ans_lists = [(span[1], span[2]) for span in span_ans_sent if (span[0] != 'V') and (span[0] != 'O')] # ans にはVが含まれる
        for y_start, y_end in span_label_ans_lists:
            array[y_end-y_start][4] += 1   # support + 1

        for pre in span_label_pre_lists:
            if pre in span_label_ans_lists:
                array[pre[1]-pre[0]][0] += 1 # correct + 1
                span_label_ans_lists.remove(pre)        
            else:
                array[pre[1]-pre[0]][1] += 1 # wrong_pred + 1
        
        for ans in span_label_ans_lists:
            array[ans[1]-ans[0]][3] += 1 # wrong_missed + 1

    array = array.T
    array[2] = array[0] + array[1]
    array = array.T

    # 各ラベルの評価
    df = pd.DataFrame(array, columns=['correct_num', 'wrong_num', 'predict_num', 'missed_num', 'support'], index=np.arange(1,max_length+1))
    df['precision'] = df['correct_num'] / df['predict_num']
    #df['precision_v2'] = df['correct_num'] / (df['predict_num'] + df['missed_num'])
    df['recall'] = df['correct_num'] / df['support']
    df['f1'] = 2*df['precision']*df['recall'] / (df['precision'] + df['recall'])
    #df['f1_v2'] = 2*df['precision_v2']*df['recall'] / (df['precision_v2'] + df['recall'])

    # 以下全体
    df.loc['sum'] = df.sum()
    if df.loc['sum', 'predict_num'] == 0:
        df.loc['sum', 'predict_num'] = 1
    df.loc['precision','correct_num'] = df.loc['sum', 'correct_num'] / df.loc['sum', 'predict_num']
    #df.loc['precision_v2','correct_num'] = df.loc['sum', 'correct_num'] / (df.loc['sum', 'predict_num'] + df.loc['sum', 'missed_num'])
    df.loc['recall','correct_num'] = df.loc['sum', 'correct_num'] / df.loc['sum', 'support']
    df.loc['f1','correct_num'] = 2*df.loc['precision','correct_num']*df.loc['recall','correct_num'] / (df.loc['precision','correct_num'] + df.loc['recall','correct_num'])
    #df.loc['f1_v2','correct_num'] = 2*df.loc['precision_v2','correct_num']*df.loc['recall','correct_num'] / (df.loc['precision_v2','correct_num'] + df.loc['recall','correct_num'])
    df = df.round(4)
    return df


def seq_fusion_matrix(xs, ys, lab2id):  
    """混合行列の計算
    spanが正解していてラベルを間違えたものの混合行列を計算する．
    
    Args:
        xs(各文章毎のラベル結果（start, end, label）のリスト) : 予測．不必要なラベルも含まれる
        ys(各文章毎のラベル結果（start, end, label）のリスト) : 正解．
    
    Returns:
        DataFrame : 混合行列
    """
    for lab in ['F-A', 'F-P', 'V', 'O', 'N']:
        try:
            del lab2id[lab]
        except:
            continue
    print(lab2id)
    array = np.zeros(len(lab2id)*(len(lab2id)).reshape(len(lab2id), (len(lab2id))))
    for span_pre_sent, span_ans_sent in zip(xs, ys):
        # データ成型, 予想spanのラベルにフラグメントと述語は含まれない．
        span_label_pre_lists = [tuple(span[:3]) for span in span_pre_sent if (span[2] != 'O') and (span[2] != 'N')]   # pre にはOが含まれる. correct_spanの時にはNも含まれる
        span_label_ans_lists = [(span[1], span[2], span[0]) for span in span_ans_sent if (span[0] != 'V') and (span[0] != 'O')] # ans にはVが含まれる
        for pre in span_label_pre_lists:
            for ans in span_label_ans_lists:
                if pre[0] == ans[0] and pre[1] == ans[1]:   # spanが正解していてラベルを間違えたものの混合行列
                    if  pre[2] != ans[2]:
                        array[lab2id[ans[2]]][lab2id[pre[2]]] += 1

    array = array.T
    array[2] = array[0] + array[1]
    array = array.T
    # 各ラベルの評価
    df = pd.DataFrame(array, columns= list(lab2id.keys()))
    df['precision'] = df['correct_num'] / df['predict_num']
    #df['precision_v2'] = df['correct_num'] / (df['predict_num'] + df['missed_num'])
    df['recall'] = df['correct_num'] / df['support']
    df['f1'] = 2*df['precision']*df['recall'] / (df['precision'] + df['recall'])
    #df['f1_v2'] = 2*df['precision_v2']*df['recall'] / (df['precision_v2'] + df['recall'])
    
    # 以下全体
    df.loc['sum'] = df.sum()
    if df.loc['sum', 'predict_num'] == 0:
        df.loc['sum', 'predict_num'] = 1
    df.loc['precision','correct_num'] = df.loc['sum', 'correct_num'] / df.loc['sum', 'predict_num']
    #df.loc['precision_v2','correct_num'] = df.loc['sum', 'correct_num'] / (df.loc['sum', 'predict_num'] + df.loc['sum', 'missed_num'])
    df.loc['recall','correct_num'] = df.loc['sum', 'correct_num'] / df.loc['sum', 'support']
    df.loc['f1','correct_num'] = 2*df.loc['precision','correct_num']*df.loc['recall','correct_num'] / (df.loc['precision','correct_num'] + df.loc['recall','correct_num'])
    df.loc['f1_macro','correct_num'] = df['f1'].sum() / len(lab2id)
    #df.loc['f1_v2','correct_num'] = 2*df.loc['precision_v2','correct_num']*df.loc['recall','correct_num'] / (df.loc['precision_v2','correct_num'] + df.loc['recall','correct_num'])
    df = df.round(4)
    return df


def get_pred_dic_lists(xs, ys, lab2id):
    pred_dic_lists = []
    match_count_list = []
    args_count_list = []
    for lab in ['F-A', 'F-P', 'V', 'O', 'N']:
        try:
            del lab2id[lab]
        except:
            continue
    array = np.zeros(len(lab2id)*5).reshape(len(lab2id), 5)
    for span_pre_sent, span_ans_sent in zip(xs, ys):
        pred_dic = []
        args_num = 0
        match_count = 0
        # データ成型, 予想spanのラベルにフラグメントと述語は含まれない．
        span_label_pre_lists = [tuple(span[:3]) for span in span_pre_sent if (span[2] != 'O') and (span[2] != 'N')]   # pre にはOが含まれる. correct_spanの時にはNも含まれる
        span_label_ans_lists = [(span[1], span[2], span[0]) for span in span_ans_sent if (span[0] != 'V') and (span[0] != 'O')] # ans にはVが含まれる
        args_count_list.append(len(span_label_ans_lists))
        for y_start, y_end, y_lab in span_label_ans_lists:
            array[lab2id[y_lab]][4] += 1   # support + 1

        for pre in span_label_pre_lists:
            if pre in span_label_ans_lists:
                array[lab2id[pre[2]]][0] += 1 # correct + 1
                span_label_ans_lists.remove(pre)        
                pred_dic.append({"start":pre[0], "end":pre[1], "role":pre[2], "true_false":True})
                match_count += 1
            else:
                array[lab2id[pre[2]]][1] += 1 # wrong_pred + 1
                pred_dic.append({"start":pre[0], "end":pre[1], "role":pre[2], "true_false":False})

        for ans in span_label_ans_lists:
            array[lab2id[ans[2]]][3] += 1 # wrong_missed + 1
        pred_dic_lists.append(pred_dic)
        match_count_list.append(match_count)

    return pred_dic_lists, match_count_list, args_count_list


# [span_idx[0], span_idx[1], label, score]
def span2seq(spans, token_len):
    seq = ['O'] * token_len
    for i, j, arg, score in spans:
        if arg != 'O':
            seq[i] = f'B-{arg}'
            for dis in range(j-i):
                seq[i+dis+1] = f'I-{arg}'
    return seq

#span2seq(10, [(0,2,'arg0'),(5,5,'arg1')])
        