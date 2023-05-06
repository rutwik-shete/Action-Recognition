#!/bin/sh

model=resnet18WithAttention
block_size=3
epochs=0
train_batch=20
val_batch=20
learning_rate=0.0003

#Concatinated Name For the Folder where log will be saved
save_log_name=${model}_${block_size}_${epochs}_${train_batch}_${val_batch}_${learning_rate}

# Path To One Directory Before Project Directory "Action-Recognition" , this is where modified data will be created
home_path=//home/media/SubhaPHD/deeptanshu/ar/

# Python main file path
main_file_path=${home_path}/Action-Recognition/main.py

# Store Original Dataset in "home_path" 
dataset_path=${home_path}/HMDB_simp

# Path where the logs and checkpoints will be saved , Added to git_ignore so that logs stay in your local
ckp=${home_path}/Action-Recognition/Logs/$save_log_name

python3 $main_file_path \
--model $model \
--home_path $home_path \
--dataset_path $dataset_path \
--resume $ckp \
--save_dir $ckp \
--block_size $block_size \
--train_batch_size $train_batch \
--val_batch_size $val_batch \
--test_batch_size $val_batch \
--lr $learning_rate \
--epochs $epochs \
--eval_freq 1 \
--run_name `whoami`_${save_log_name} \
-t $model \
-s $model
