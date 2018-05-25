from pathlib import Path
from collections import Counter
import numpy as np

rank_a_file = '/data/david/fai_attr/sources/sources/fashionAI_attributes_answer_a_20180428.csv'
rank_b_file = '/data/david/fai_attr/sources/sources/fashionAI_attributes_answer_b_20180428.csv'
train_2_file = '/data/david/fai_attr/raw_data/ROUND2/PURE_TRAIN_V1/Annotations/label_without_head.csv'
val_1_file = '/data/david/fai_attr/raw_data/ROUND1/val_v1/Annotations/val.csv'


def cal_missing_token(file_path):
    assert Path(file_path).exists(), "%s not exists" % file_path
    with Path(file_path).open('r') as f:
        lines = f.readlines()
    all_tokens = [l.rstrip().split(',') for l in lines]
    missing_tokens = [t for t in all_tokens if t[2][0] == 'y']
    print("%s, not_exist/all %.3f" % (file_path, len(missing_tokens)/len(all_tokens)))


cal_missing_token(val_1_file)
cal_missing_token(train_2_file)
