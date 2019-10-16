CUDA_VISIBLE_DEVICES=1 python script/train_taobao.py -p train --random_seed 19 --model_type DNN --seq_minlen 0 --epoch 1 --start_test_iter=2500 --test_iter=100
