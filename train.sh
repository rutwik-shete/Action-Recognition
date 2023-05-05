#!/bin/sh

TimeSformerType=600
block_size=16
epochs=10
train_batch=20
val_batch=20
save_log_name=TimeSFormer_${TimeSformerType}_${block_size}_${epochs}_${train_batch}_${val_batch}

main_file_path=/Users/rutwikshete/Desktop/Codeing/Surrey/SurreyAssignment/Action-Recognition/main.py
home_path=/Users/rutwikshete/Desktop/Codeing/Surrey/SurreyAssignment/Datasets/
dataset_path=/Users/rutwikshete/Desktop/Codeing/Surrey/SurreyAssignment/Datasets/HMDB_simp
ckp=/Users/rutwikshete/Desktop/Codeing/Surrey/SurreyAssignment/Action-Recognition/Logs/$save_log_name

python3 $main_file_path \
--home_path $home_path \
--dataset_path $dataset_path \
--resume $ckp \
--save_dir $ckp \
--block_size $block_size \
--train_batch_size $train_batch \
--val_batch_size $val_batch \
--test_batch_size $val_batch \
--epochs $epochs \
--eval_freq 1 \
-t TimeSFormer_$TimeSformerType \
-s TimeSFormer_$TimeSformerType

