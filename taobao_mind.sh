CUDA_VISIBLE_DEVICES=0 python script/train_taobao.py -p train --random_seed 19 --model_type MIND --seq_minlen 0 --cluster_num=3 --epoch=1 --start_test_iter=2000
