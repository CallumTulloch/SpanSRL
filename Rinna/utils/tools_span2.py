import torch
import numpy as np

# 述語スパンをnullスパンとし，そのスパンのスコアより低い各ラベルのスパンは消す）
# span_score = [span_idx[0], span_idx[1], label, score]
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

def filter_span_score_delArg(span_score_lists, pred_span, input_token_num, pred_span_scores, lab2id):   # span_score_listsはマトリックスを崩した一つひとつの要素
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
        elif (span_score[2] in ['Arg']):   # 必要ないラベルのscoreを削除. マトリックスの横を全部消すイメージ. Nを選べば，正しく選択できなかったことを意味する．
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

