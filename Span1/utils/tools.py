import torch
import numpy as np

def filter_span_score_lists(span_score_lists, pred_span, input_token_num, MAX_LENGTH):   # span_score_listsはマトリックスを崩した一つひとつの要素
    filterd_span_score_lists=[]
    null_span_score_array = np.zeros(MAX_LENGTH*MAX_LENGTH).reshape(MAX_LENGTH,MAX_LENGTH) # 行列で使うのはスパンに対応した部分のみ
    null_span_score = [span_score for span_score in span_score_lists if span_score[2] == 'N']
    for start, end, label, score in null_span_score:
        null_span_score_array[start][end] = score
    #print(null_span_score_array)
    
    pred_span_element = set( range(pred_span[0], pred_span[1]+1) )
    for span_score in span_score_lists:
        target_span_element = set( range(span_score[0], span_score[1]+1) )

        if ( (span_score[0] >= input_token_num) or (span_score[1] >= input_token_num) ):   # 入力した文章の長さを考慮したfilter. spanは0から開始している
            continue
        elif (span_score[2] in ['F-A','F-P','V','N']):   # 必要ないラベルのscoreを削除. マトリックスの横を全部消すイメージ
            continue
        elif span_score[3] <= null_span_score_array[span_score[0]][span_score[1]]:   # Null ラベル はこのスパンはありえないという意味のラベル．そのため，各スパンのNull ラベルより小さい スパンのscoreは削除する．
            continue
        elif target_span_element.isdisjoint(pred_span_element) == False:    # isdisjointは互いに独立であるかというメソッド. pred_span と span_score が被るかどうか．マトリックスの縦を消すイメージ
            continue
        else:
            #print(f'( ({span_score[0]} >= {input_token_num}) or ({span_score[1]} >= {input_token_num}) )')
            if span_score[2] == 'O':
                span_score[3] -= 0.9
            filterd_span_score_lists.append(span_score)
    return filterd_span_score_lists

# O を含む
def filter_span_score_for_batch_1(span_score_lists, pred_span, input_token_num):   # span_score_listsはマトリックスを崩した一つひとつの要素
    filterd_span_score_lists=[]
    null_span_score_array = np.zeros(input_token_num*input_token_num).reshape(input_token_num,input_token_num) # 行列で使うのはスパンに対応した部分のみ
    null_span_score = [span_score for span_score in span_score_lists if span_score[2] == 'N']
    for start, end, label, score in null_span_score:
        null_span_score_array[start][end] = score
    #print(null_span_score_array)
    
    pred_span_element = set( range(pred_span[0], pred_span[1]+1) )
    for span_score in span_score_lists:
        target_span_element = set( range(span_score[0], span_score[1]+1) )

        if ( (span_score[0] >= input_token_num) or (span_score[1] >= input_token_num) ):   # 入力した文章の長さを考慮したfilter. spanは0から開始している
            continue
        elif (span_score[2] in ['F-A','F-P','V','N']):   # 必要ないラベルのscoreを削除. マトリックスの横を全部消すイメージ
            continue
        elif span_score[3] <= null_span_score_array[span_score[0]][span_score[1]]:   # Null ラベル はこのスパンはありえないという意味のラベル．そのため，各スパンのNull ラベルより小さい スパンのscoreは削除する．
            continue
        elif target_span_element.isdisjoint(pred_span_element) == False:    # isdisjointは互いに独立であるかというメソッド. pred_span と span_score が被るかどうか．マトリックスの縦を消すイメージ
            continue
        else:
            filterd_span_score_lists.append(span_score)
    return filterd_span_score_lists

import itertools
# O を含む
def filter_span(matrix, pred_span, input_token_num, id2lab, lab2id, span_available_indication):   # matrix = [token,token,label]
    null_span_score_array = matrix[:, :, lab2id['N']]    
    pred_span_element = set( range(pred_span[0], pred_span[1]+1) )
    for i, j in itertools.product(range(input_token_num), repeat=2):
        if span_available_indication[i,j] != -1:
            matrix[i,j,:]=np.nan
            continue
        for id in lab2id.values():
            target_span_element = set( range(i, j+1) )
            #print(i, j, id2lab[j], matrix[i,j,id], null_span_score_array[i][j])
            #exit()
            if (id2lab[id] in ['F-A','F-P','V','N']):   # 必要ないラベルのscoreを削除. マトリックスの横を全部消すイメージ
                matrix[i,j,id]=np.nan
                continue
            if matrix[i,j,id] <= null_span_score_array[i,j]:   # Null ラベル はこのスパンはありえないという意味のラベル．そのため，各スパンのNull ラベルより小さい スパンのscoreは削除する．
                matrix[i,j,id]=np.nan
                continue
            if target_span_element.isdisjoint(pred_span_element) == False:    # isdisjointは互いに独立であるかというメソッド. pred_span と span_score が被るかどうか．マトリックスの縦を消すイメージ
                matrix[i,j,id]=np.nan
                continue
    return matrix


# Oを除外
def filter_span_score_for_batch_1_v2(span_score_lists, pred_span, input_token_num):   # span_score_listsはマトリックスを崩した一つひとつの要素
    filterd_span_score_lists=[]
    null_span_score_array = np.zeros(input_token_num*input_token_num).reshape(input_token_num,input_token_num) # 行列で使うのはスパンに対応した部分のみ
    null_span_score = [span_score for span_score in span_score_lists if span_score[2] == 'N']
    for start, end, label, score in null_span_score:
        null_span_score_array[start][end] = score
    #print(null_span_score_array)
    
    pred_span_element = set( range(pred_span[0], pred_span[1]+1) )
    for span_score in span_score_lists:
        target_span_element = set( range(span_score[0], span_score[1]+1) )

        if ( (span_score[0] >= input_token_num) or (span_score[1] >= input_token_num) ):   # 入力した文章の長さを考慮したfilter. spanは0から開始している
            continue
        elif (span_score[2] in ['F-A','F-P','V','N','O']):   # 必要ないラベルのscoreを削除. マトリックスの横を全部消すイメージ
            continue
        elif span_score[3] <= null_span_score_array[span_score[0]][span_score[1]]:   # Null ラベル はこのスパンはありえないという意味のラベル．そのため，各スパンのNull ラベルより小さい スパンのscoreは削除する．
            continue
        elif target_span_element.isdisjoint(pred_span_element) == False:    # isdisjointは互いに独立であるかというメソッド. pred_span と span_score が被るかどうか．マトリックスの縦を消すイメージ
            continue
        else:
            filterd_span_score_lists.append(span_score)
    return filterd_span_score_lists

# filter 結果が0 の場合一番可能性のあるスパンを1つだけ選択．
def get_one_span(span_score_lists, pred_span, input_token_num):   # span_score_listsはマトリックスを崩した一つひとつの要素
    filterd_span_score_lists=[]    
    pred_span_element = set( range(pred_span[0], pred_span[1]+1) )
    for span_score in span_score_lists:
        target_span_element = set( range(span_score[0], span_score[1]+1) )

        if ( (span_score[0] >= input_token_num) or (span_score[1] >= input_token_num) ):   # 入力した文章の長さを考慮したfilter. spanは0から開始している
            continue
        elif (span_score[2] in ['F-A','F-P','V','N','O']):   # 必要ないラベルのscoreを削除. マトリックスの横を全部消すイメージ
            continue
        elif target_span_element.isdisjoint(pred_span_element) == False:    # isdisjointは互いに独立であるかというメソッド. pred_span と span_score が被るかどうか．マトリックスの縦を消すイメージ
            continue
        else:
            filterd_span_score_lists.append(span_score)
            break
    return filterd_span_score_lists
    
# ラベル増加
def filter_span_score_for_batch_1_addlabel(span_score_lists, pred_span, input_token_num, f_labels):   # span_score_listsはマトリックスを崩した一つひとつの要素
    filterd_span_score_lists=[]
    null_span_score_array = np.zeros(input_token_num*input_token_num).reshape(input_token_num,input_token_num) # 行列で使うのはスパンに対応した部分のみ
    null_span_score = [span_score for span_score in span_score_lists if span_score[2] == 'N']
    for start, end, label, score in null_span_score:
        null_span_score_array[start][end] = score
    #print(null_span_score_array)
    
    pred_span_element = set( range(pred_span[0], pred_span[1]+1) )
    for span_score in span_score_lists:
        target_span_element = set( range(span_score[0], span_score[1]+1) )

        if ( (span_score[0] >= input_token_num) or (span_score[1] >= input_token_num) ):   # 入力した文章の長さを考慮したfilter. spanは0から開始している
            continue
        elif (span_score[2] in ['F-P','V','N'] + f_labels):   # 必要ないラベルのscoreを削除. マトリックスの横を全部消すイメージ
            continue
        elif span_score[3] <= null_span_score_array[span_score[0]][span_score[1]]:   # Null ラベル はこのスパンはありえないという意味のラベル．そのため，各スパンのNull ラベルより小さい スパンのscoreは削除する．
            continue
        elif target_span_element.isdisjoint(pred_span_element) == False:    # isdisjointは互いに独立であるかというメソッド. pred_span と span_score が被るかどうか．マトリックスの縦を消すイメージ
            continue
        else:
            filterd_span_score_lists.append(span_score)
    return filterd_span_score_lists

# Oを除外，ラベル増加
def filter_span_score_for_batch_1_v2_addlabel(span_score_lists, pred_span, input_token_num, f_labels):   # span_score_listsはマトリックスを崩した一つひとつの要素
    filterd_span_score_lists=[]
    null_span_score_array = np.zeros(input_token_num*input_token_num).reshape(input_token_num,input_token_num) # 行列で使うのはスパンに対応した部分のみ
    null_span_score = [span_score for span_score in span_score_lists if span_score[2] == 'N']
    for start, end, label, score in null_span_score:
        null_span_score_array[start][end] = score
    #print(null_span_score_array)
    
    pred_span_element = set( range(pred_span[0], pred_span[1]+1) )
    for span_score in span_score_lists:
        target_span_element = set( range(span_score[0], span_score[1]+1) )

        if ( (span_score[0] >= input_token_num) or (span_score[1] >= input_token_num) ):   # 入力した文章の長さを考慮したfilter. spanは0から開始している
            continue
        elif (span_score[2] in ['F-P','V','N','O'] + f_labels):   # 必要ないラベルのscoreを削除. マトリックスの横を全部消すイメージ
            continue
        elif span_score[3] <= null_span_score_array[span_score[0]][span_score[1]]:   # Null ラベル はこのスパンはありえないという意味のラベル．そのため，各スパンのNull ラベルより小さい スパンのscoreは削除する．
            continue
        elif target_span_element.isdisjoint(pred_span_element) == False:    # isdisjointは互いに独立であるかというメソッド. pred_span と span_score が被るかどうか．マトリックスの縦を消すイメージ
            continue
        else:
            filterd_span_score_lists.append(span_score)
    return filterd_span_score_lists

# スパンのみよう
def filter_span_score_for_batch_1_onlyspan(span_score_lists, pred_span, input_token_num):   # span_score_listsはマトリックスを崩した一つひとつの要素
    filterd_span_score_lists=[]
    o_span_score_array = np.zeros(input_token_num*input_token_num).reshape(input_token_num,input_token_num) # 行列で使うのはスパンに対応した部分のみ
    null_span_score = [span_score for span_score in span_score_lists if span_score[2] == 'O']
    for start, end, label, score in null_span_score:
        o_span_score_array[start][end] = score
    #print(o_span_score_array)
    
    pred_span_element = set( range(pred_span[0], pred_span[1]+1) )
    for span_score in span_score_lists:
        target_span_element = set( range(span_score[0], span_score[1]+1) )

        if ( (span_score[0] >= input_token_num) or (span_score[1] >= input_token_num) ):   # 入力した文章の長さを考慮したfilter. spanは0から開始している
            continue
        elif (span_score[2] in ['F-A','F-P','V']):   # 必要ないラベルのscoreを削除. マトリックスの横を全部消すイメージ
            continue
        elif span_score[3] <= o_span_score_array[span_score[0]][span_score[1]]:   # O ラベル はこのスパンではないという意味のラベル．そのため，各スパンのO ラベルより小さい スパンのscoreは削除する．
            continue
        elif target_span_element.isdisjoint(pred_span_element) == False:    # isdisjointは互いに独立であるかというメソッド. pred_span と span_score が被るかどうか．マトリックスの縦を消すイメージ
            continue
        else:
            filterd_span_score_lists.append(span_score)
    return filterd_span_score_lists

def filter_correct_span(span_score_lists, answer_span_lists):   # span_score_listsはマトリックスを崩した一つひとつの要素
    filterd_span_score_lists=[]
    answer_spans = [[span[1],span[2]] for span in answer_span_lists if (span[0] != 'O') and (span[0] != 'V')]
    
    for span in span_score_lists:
        if [span[0], span[1]] not in answer_spans:
            continue
        elif (span[2] in ['F-A','F-P','V','O']):   # 必要ないラベルのscoreを削除. マトリックスの横を全部消すイメージ. Nを選べば，正しく選択できなかったことを意味する．
            continue
        else:
            print(span)
            filterd_span_score_lists.append(span)
    return filterd_span_score_lists

def is_overlap(i, j, spans):
    target_span_element = set( np.arange(i, j+1) )

    for span in spans:
        span_element = set( np.arange(span[0], span[1]+1) )
        if target_span_element.isdisjoint(span_element) == False:
            return True
    return False

#ohuchi-san
def filter_span_score(span_score_lists, pred_span, input_token_num, pred_span_scores, lab2id):   # span_score_listsはマトリックスを崩した一つひとつの要素
    filterd_span_score_lists=[]
    pred_span_element = set( range(pred_span[0], pred_span[1]+1) )
    for span_score in span_score_lists: # [start][end][label][score]
        target_span_element = set( range(span_score[0], span_score[1]+1) )

        #if ( (span_score[0] >= input_token_num) or (span_score[1] >= input_token_num) ):   # 入力した文章の長さを考慮したfilter. spanは0から開始している
        #    print('########### FILTER-MISS ############')
        #    continue
        if pred_span_scores[lab2id[span_score[2]]] > span_score[3]:   # nullスパン(述語). マトリックスの横を全部消すイメージ. 
            continue
        elif target_span_element.isdisjoint(pred_span_element) == False:    # isdisjointは互いに独立であるかというメソッド. pred_span と span_score が被るかどうか．マトリックスの縦を消すイメージ
            continue
        else:
            filterd_span_score_lists.append(span_score)
    return filterd_span_score_lists

def filter_span_score_both(span_score_lists, pred_span, input_token_num):   # span_score_listsはマトリックスを崩した一つひとつの要素
    filterd_span_score_lists=[]
    null_span_score_array = np.zeros(input_token_num*input_token_num).reshape(input_token_num,input_token_num) # 行列で使うのはスパンに対応した部分のみ
    null_span_score = [span_score for span_score in span_score_lists if span_score[2] == 'N']
    for start, end, label, score in null_span_score:
        null_span_score_array[start][end] = score
    #print(null_span_score_array)
    
    pred_span_element = set( range(pred_span[0], pred_span[1]+1) )
    for span_score in span_score_lists:
        target_span_element = set( range(span_score[0], span_score[1]+1) )

        if ( (span_score[0] >= input_token_num) or (span_score[1] >= input_token_num) ):   # 入力した文章の長さを考慮したfilter. spanは0から開始している
            continue
        elif (span_score[2] in ['F-A','F-P','V','N']):   # 必要ないラベルのscoreを削除. マトリックスの横を全部消すイメージ
            continue
        elif span_score[3] <= null_span_score_array[span_score[0]][span_score[1]]:   # Null ラベル はこのスパンはありえないという意味のラベル．そのため，各スパンのNull ラベルより小さい スパンのscoreは削除する．
            continue
        elif target_span_element.isdisjoint(pred_span_element) == False:    # isdisjointは互いに独立であるかというメソッド. pred_span と span_score が被るかどうか．マトリックスの縦を消すイメージ
            continue
        else:
            filterd_span_score_lists.append(span_score)
    return filterd_span_score_lists