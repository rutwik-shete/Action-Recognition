#!/bin/sh

model=timesformer400
block_size=8
epochs=3
train_batch=20
val_batch=20
learning_rate=0.0003

#Concatinated Name For the Folder where log will be saved
save_log_name=${model}_${block_size}_${epochs}_${train_batch}_${val_batch}_${learning_rate}

# Path To One Directory Before Project Directory "Action-Recognition" , this is where modified data will be created
home_path=/Users/rutwikshete/Desktop/Codeing/Surrey/SurreyAssignment

# Python main file path
main_file_path=${home_path}/Action-Recognition/main.py

# Store Original Dataset in "home_path" 
dataset_path=${home_path}/HMDB_simp

# Path where the logs and checkpoints will be saved , Added to git_ignore so that logs stay in your local
ckp=${home_path}/Action-Recognition/Logs/$save_log_name

export LD_LIBRARY_PATH=/user/HS402/rs01960/libstdc:$LD_LIBRARY_PATH

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
--run_name RutwikRunTest \
-t $model \
-s $model
# Rutwik_${save_log_name}