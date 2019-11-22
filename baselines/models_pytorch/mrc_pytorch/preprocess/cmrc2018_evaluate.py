# -*- coding: utf-8 -*-
'''
Evaluation script for CMRC 2018
version: v5 - special
Note:
v5 - special: Evaluate on SQuAD-style CMRC 2018 Datasets
v5: formatted output, add usage description
v4: fixed segmentation issues
'''
from __future__ import print_function

import json
import re
from collections import OrderedDict

import nltk


# split Chinese with English
def mixed_segmentation(in_str, rm_punc=False):
    in_str = str(in_str).lower().strip()
    segs_out = []
    temp_str = ""
    sp_char = ['-', ':', '_', '*', '^', '/', '\\', '~', '`', '+', '=',
               '，', '。', '：', '？', '！', '“', '”', '；', '’', '《', '》', '……', '·', '、',
               '「', '」', '（', '）', '－', '～', '『', '』']
    for char in in_str:
        if rm_punc and char in sp_char:
            continue
        if re.search(r'[\u4e00-\u9fa5]', char) or char in sp_char:
            if temp_str != "":
                ss = nltk.word_tokenize(temp_str)
                segs_out.extend(ss)
                temp_str = ""
            segs_out.append(char)
        else:
            temp_str += char

    # handling last part
    if temp_str != "":
        ss = nltk.word_tokenize(temp_str)
        segs_out.extend(ss)

    return segs_out


# remove punctuation
def remove_punctuation(in_str):
    in_str = str(in_str).lower().strip()
    sp_char = ['-', ':', '_', '*', '^', '/', '\\', '~', '`', '+', '=',
               '，', '。', '：', '？', '！', '“', '”', '；', '’', '《', '》', '……', '·', '、',
               '「', '」', '（', '）', '－', '～', '『', '』']
    out_segs = []
    for char in in_str:
        if char in sp_char:
            continue
        else:
            out_segs.append(char)
    return ''.join(out_segs)


# find longest common string
def find_lcs(s1, s2):
    m = [[0 for i in range(len(s2) + 1)] for j in range(len(s1) + 1)]
    mmax = 0
    p = 0
    for i in range(len(s1)):
        for j in range(len(s2)):
            if s1[i] == s2[j]:
                m[i + 1][j + 1] = m[i][j] + 1
                if m[i + 1][j + 1] > mmax:
                    mmax = m[i + 1][j + 1]
                    p = i + 1
    return s1[p - mmax:p], mmax


def evaluate(ground_truth_file, prediction_file):
    f1 = 0
    em = 0
    total_count = 0
    skip_count = 0
    for instance in ground_truth_file["data"]:
        # context_id   = instance['context_id'].strip()
        # context_text = instance['context_text'].strip()
        for para in instance["paragraphs"]:
            for qas in para['qas']:
                total_count += 1
                query_id = qas['id'].strip()
                query_text = qas['question'].strip()
                answers = [x["text"] for x in qas['answers']]

                if query_id not in prediction_file:
                    print('Unanswered question: {}\n'.format(query_id))
                    skip_count += 1
                    continue

                prediction = str(prediction_file[query_id])
                f1 += calc_f1_score(answers, prediction)
                em += calc_em_score(answers, prediction)

    f1_score = 100.0 * f1 / total_count
    em_score = 100.0 * em / total_count
    return f1_score, em_score, total_count, skip_count


def evaluate2(ground_truth_file, prediction_file):
    f1 = 0
    em = 0
    total_count = 0
    skip_count = 0
    yes_count = 0
    yes_correct = 0
    no_count = 0
    no_correct = 0
    unk_count = 0
    unk_correct = 0

    for instance in ground_truth_file["data"]:
        for para in instance["paragraphs"]:
            for qas in para['qas']:
                total_count += 1
                query_id = qas['id'].strip()
                if query_id not in prediction_file:
                    print('Unanswered question: {}\n'.format(query_id))
                    skip_count += 1
                    continue

                prediction = str(prediction_file[query_id])

                if len(qas['answers']) == 0:
                    unk_count += 1
                    answers = [""]
                    if prediction == "":
                        unk_correct += 1
                else:
                    answers = []
                    for x in qas['answers']:
                        answers.append(x['text'])
                        if x['text'] == 'YES':
                            if prediction == 'YES':
                                yes_correct += 1
                            yes_count += 1
                        if x['text'] == 'NO':
                            if prediction == 'NO':
                                no_correct += 1
                            no_count += 1

                f1 += calc_f1_score(answers, prediction)
                em += calc_em_score(answers, prediction)

    f1_score = 100.0 * f1 / total_count
    em_score = 100.0 * em / total_count
    yes_acc = 100.0 * yes_correct / yes_count
    no_acc = 100.0 * no_correct / no_count
    unk_acc = 100.0 * unk_correct / unk_count
    return f1_score, em_score, yes_acc, no_acc, unk_acc, total_count, skip_count


def calc_f1_score(answers, prediction):
    f1_scores = []
    for ans in answers:
        ans_segs = mixed_segmentation(ans, rm_punc=True)
        prediction_segs = mixed_segmentation(prediction, rm_punc=True)
        lcs, lcs_len = find_lcs(ans_segs, prediction_segs)
        if lcs_len == 0:
            f1_scores.append(0)
            continue
        precision = 1.0 * lcs_len / len(prediction_segs)
        recall = 1.0 * lcs_len / len(ans_segs)
        f1 = (2 * precision * recall) / (precision + recall)
        f1_scores.append(f1)
    return max(f1_scores)


def calc_em_score(answers, prediction):
    em = 0
    for ans in answers:
        ans_ = remove_punctuation(ans)
        prediction_ = remove_punctuation(prediction)
        if ans_ == prediction_:
            em = 1
            break
    return em


def get_eval(original_file, prediction_file):
    ground_truth_file = json.load(open(original_file, 'r'))
    prediction_file = json.load(open(prediction_file, 'r'))
    F1, EM, TOTAL, SKIP = evaluate(ground_truth_file, prediction_file)
    AVG = (EM + F1) * 0.5
    output_result = OrderedDict()
    output_result['AVERAGE'] = '%.3f' % AVG
    output_result['F1'] = '%.3f' % F1
    output_result['EM'] = '%.3f' % EM
    output_result['TOTAL'] = TOTAL
    output_result['SKIP'] = SKIP

    return output_result


def get_eval_with_neg(original_file, prediction_file):
    ground_truth_file = json.load(open(original_file, 'r'))
    prediction_file = json.load(open(prediction_file, 'r'))
    F1, EM, YES_ACC, NO_ACC, UNK_ACC, TOTAL, SKIP = evaluate2(ground_truth_file, prediction_file)
    AVG = (EM + F1) * 0.5
    output_result = OrderedDict()
    output_result['AVERAGE'] = '%.3f' % AVG
    output_result['F1'] = '%.3f' % F1
    output_result['EM'] = '%.3f' % EM
    output_result['YES'] = '%.3f' % YES_ACC
    output_result['NO'] = '%.3f' % NO_ACC
    output_result['UNK'] = '%.3f' % UNK_ACC
    output_result['TOTAL'] = TOTAL
    output_result['SKIP'] = SKIP

    return output_result
