# check results

from pathlib import Path
from collections import Counter
import numpy as np

base_submission_path = '/home/david/fashionAI/fashionAI-Attributes-Gluon-project/submit/submision_upload.csv'
dest_submission_path = '/home/david/fashionAI/fashionAI-Attributes-Gluon-project/submit/v1/submission.csv'
assert Path(base_submission_path).exists(), "%s not exists" % base_submission_path
assert Path(dest_submission_path).exists(), "%s not exists" % dest_submission_path

with Path(base_submission_path).open('r') as f:
    lines = f.readlines()
base_tokens = [l.rstrip().split(',') for l in lines]

with Path(dest_submission_path).open('r') as f:
    lines = f.readlines()
dest_tokens = [l.rstrip().split(',') for l in lines]

assert len(base_tokens) == len(dest_tokens), "counts don't match"

base_list = dict([(line[0], [float(num) for num in line[2].split(';')]) for line in base_tokens])
dest_list = dict([(line[0], [float(num) for num in line[2].split(';')]) for line in dest_tokens])

loss = 0.0

for img_id in base_list.keys():
    base_label = base_list[img_id]
    dest_label = dest_list[img_id]
    loss += np.mean(np.array(base_label) - np.array(dest_label))

print("total differences: %.4f" % loss)


