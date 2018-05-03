from pathlib import Path
from collections import Counter
import numpy as np

rank_a_file = '/data/david/fai_attr/sources/sources/fashionAI_attributes_answer_a_20180428.csv'
rank_b_file = '/data/david/fai_attr/sources/sources/fashionAI_attributes_answer_b_20180428.csv'
assert Path(rank_a_file).exists(), "%s not exists" % rank_a_file
assert Path(rank_b_file).exists(), "%s not exists" % rank_b_file

with Path(rank_a_file).open('r') as f:
    lines = f.readlines()
rank_a_tokens = [l.rstrip().split(',') for l in lines]
missing_a_tokens = [t for t in rank_a_tokens if t[2][0] == 'y']
len(missing_a_tokens)/len(rank_a_tokens)

with Path(rank_b_file).open('r') as f:
    lines = f.readlines()
rank_b_tokens = [l.rstrip().split(',') for l in lines]
missing_b_tokens = [t for t in rank_b_tokens if t[2][0] == 'y']
len(missing_b_tokens)/len(rank_b_tokens)

