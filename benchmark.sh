export MXNET_CUDNN_AUTOTUNE_DEFAULT=0

python3 train.py collar_design_labels
python3 train.py skirt_length_labels
python3 train.py lapel_design_labels
python3 train.py neckline_design_labels
python3 train.py coat_length_labels
python3 train.py neck_design_labels
python3 train.py pant_length_labels
python3 train.py sleeve_length_labels
