export MXNET_CUDNN_AUTOTUNE_DEFAULT=0

num_gpus=4,5,6
num_workers=108
batch_size=24
default_epochs=40
# default_model=resnet152_v2
# default_model=resnet50_v2
# default_model=densenet121
default_model=densenet201
# default_model=inceptionv3
# collar_mode=resnet50_v2


# python3 prepare_data.py

python3 train_task.py --task collar_design_labels --model $default_model --num-gpus $num_gpus -j $num_workers -b $batch_size --epochs $default_epochs
python3 train_task.py --task neckline_design_labels --model $default_model --num-gpus $num_gpus -j $num_workers -b $batch_size --epochs $default_epochs
python3 train_task.py --task skirt_length_labels --model $default_model --num-gpus $num_gpus -j $num_workers -b $batch_size --epochs $default_epochs
python3 train_task.py --task sleeve_length_labels --model $default_model --num-gpus $num_gpus -j $num_workers -b $batch_size --epochs $default_epochs
python3 train_task.py --task neck_design_labels --model $default_model --num-gpus $num_gpus -j $num_workers -b $batch_size --epochs $default_epochs
python3 train_task.py --task coat_length_labels --model $default_model --num-gpus $num_gpus -j $num_workers -b $batch_size --epochs $default_epochs
python3 train_task.py --task lapel_design_labels --model $default_model --num-gpus $num_gpus -j $num_workers -b $batch_size --epochs $default_epochs
python3 train_task.py --task pant_length_labels --model $default_model --num-gpus $num_gpus -j $num_workers -b $batch_size --epochs $default_epochs

cd submission
cat collar_design_labels.csv neckline_design_labels.csv skirt_length_labels.csv sleeve_length_labels.csv neck_design_labels.csv coat_length_labels.csv lapel_design_labels.csv pant_length_labels.csv > submission.csv

