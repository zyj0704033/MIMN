CUDA_VISIBLE_DEVICES=0 python script/train_taobao.py -p train --random_seed 19 --model_type CDNN --seq_minlen 0 --epoch 1 --start_test_iter=200 --test_iter=300
