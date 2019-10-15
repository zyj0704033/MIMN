#coding:utf-8
import numpy
from data_iterator_new import DataIterator
import tensorflow as tf
from model import *
import time
import random
import sys
from utils import *
import multiprocessing
import argparse
import cPickle as pkl

parser = argparse.ArgumentParser()
parser.add_argument('-p', type=str, default='train', help='train | test')
parser.add_argument('--random_seed', type=int, default=19)
parser.add_argument('--model_type', type=str, default='none', help='DIEN | MIMN | ..')
parser.add_argument('--memory_size', type=int, default=4)
parser.add_argument('--mem_induction', type=int, default=0, help='0:false|1:true')
parser.add_argument('--util_reg', type=int, default=0, help='0:false|1:true')
parser.add_argument('--seq_maxlen', type=int, default=200, help='max len of user behavior seq')
parser.add_argument('--seq_minlen', type=int, default=0, help='min len of user behavior seq for filler sample')
parser.add_argument('--cluster_num', type=int, default=3, help='cluster num for capsule')
parser.add_argument('--start_test_iter', type=int, default=1000, help='the iter start to test')
parser.add_argument('--test_iter', type=int, default=100, help='the iter start to test')
parser.add_argument('--epoch', type=int, default=1, help='epoch num')


def generator_queue(generator, max_q_size=20,
                    wait_time=0.1, nb_worker=1):

    generator_threads = []
    q = multiprocessing.Queue(maxsize=max_q_size)
    _stop = multiprocessing.Event()
    try:
        def data_generator_task():
            while not _stop.is_set():
                try:
                    if q.qsize() < max_q_size:
                        #start_time = time.time()
                        generator_output = next(generator)
                        #end_time = time.time()
                        #print end_time - start_time
                        q.put(generator_output)
                    else:
                        time.sleep(wait_time)
                except Exception:
                    _stop.set()
                    print("over1")
                    #raise

        for i in range(nb_worker):
            thread = multiprocessing.Process(target=data_generator_task)
            generator_threads.append(thread)
            thread.daemon = True
            thread.start()
    except Exception:
        _stop.set()
        for p in generator_threads:
            if p.is_alive():
                p.terminate()
        q.close()
        print("over")

    return q, _stop, generator_threads

EMBEDDING_DIM = 16
HIDDEN_SIZE = 16 * 2 
best_auc = 0.0

def prepare_data(src, target):
    nick_id, item_id, cate_id = src
    label, hist_item, hist_cate, neg_item, neg_cate, hist_mask = target
    return nick_id, item_id, cate_id, label, hist_item, hist_cate, neg_item, neg_cate, hist_mask
    
def eval(sess, test_data, model, model_path, batch_size):
    loss_sum = 0.
    accuracy_sum = 0.
    aux_loss_sum = 0.
    nums = 0
    stored_arr = []
    test_data_pool, _stop, _ = generator_queue(test_data)
    while True:
        if  _stop.is_set() and test_data_pool.empty():
            break
        if not test_data_pool.empty():
            src,tgt = test_data_pool.get()
        else:
            continue
        nick_id, item_id, cate_id, label, hist_item, hist_cate, neg_item, neg_cate, hist_mask = prepare_data(src, tgt) 
        if len(nick_id) < batch_size:
            continue
        nums += 1
        target = label
        prob, loss, acc, aux_loss = model.calculate(sess, [nick_id, item_id, cate_id, hist_item, hist_cate, neg_item, neg_cate, hist_mask, label])
        loss_sum += loss
        aux_loss_sum = aux_loss
        accuracy_sum += acc
        prob_1 = prob[:, 0].tolist()
        target_1 = target[:, 0].tolist()
        for p ,t in zip(prob_1, target_1):
            stored_arr.append([p, t])
    
    test_auc = calc_auc(stored_arr)
    accuracy_sum = accuracy_sum / nums
    loss_sum = loss_sum / nums
    aux_loss_sum / nums
    global best_auc
    if best_auc < test_auc:
        best_auc = test_auc
        model.save(sess, model_path)
    return test_auc, loss_sum, accuracy_sum, aux_loss_sum

def train(
        train_file = "../UIC_MIMN/data/taobao_data/taobao_train.txt",
        test_file = "../UIC_MIMN/data/taobao_data/taobao_test.txt",
        feature_file = "../UIC_MIMN/data/taobao_data/taobao_feature.pkl",
        batch_size = 256,
        maxlen = 200,
        minlen = 0,
        test_iter = 100,
        save_iter = 9999999,
        model_type = 'DNN',
        Memory_Size = 4,
        Mem_Induction = 0, 
        Util_Reg = 0,
        cluster_num = 3,
        start_test_iter = 1000,
        epoch = 1
):
    if model_type != "MIMN" or model_type != "MIMN_with_aux":
        model_path = "dnn_save_path/book_ckpt_noshuff" + model_type
        best_model_path = "dnn_best_model/book_ckpt_noshuff" + model_type
    else:
        model_path = "dnn_save_path/book_ckpt_noshuff" + model_type+str(Memory_Size)+str(Mem_Induction)
        best_model_path = "dnn_best_model/book_ckpt_noshuff" + model_type+str(Memory_Size)+str(Mem_Induction)
    gpu_options = tf.GPUOptions(allow_growth=True)

    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        
        #train_data = DataIterator(train_file, batch_size=batch_size, maxlen=maxlen, minlen=minlen)
        test_data = DataIterator(test_file, batch_size=batch_size, maxlen=maxlen, minlen=minlen)    
        train_data = DataIterator(train_file, batch_size=batch_size, maxlen=maxlen, minlen=minlen)
        feature_num = pkl.load(open(feature_file))
        n_uid, n_mid = feature_num, feature_num
        BATCH_SIZE = batch_size
        SEQ_LEN = maxlen

        if model_type == 'DNN': 
            model = Model_DNN(n_uid, n_mid, EMBEDDING_DIM, HIDDEN_SIZE, BATCH_SIZE, SEQ_LEN)
        elif model_type == 'CDNN': 
            model = Model_CDNN(n_uid, n_mid, EMBEDDING_DIM, HIDDEN_SIZE, BATCH_SIZE, SEQ_LEN)
        elif model_type == 'PNN': 
            model = Model_PNN(n_uid, n_mid, EMBEDDING_DIM, HIDDEN_SIZE, BATCH_SIZE, SEQ_LEN)
        elif model_type == 'MIND': 
            model = Model_MIND(n_uid, n_mid, EMBEDDING_DIM, HIDDEN_SIZE, BATCH_SIZE, SEQ_LEN, cluster_num = cluster_num, is_attention=True, use_mask=False)
        elif model_type == 'GRU4REC': 
            model = Model_GRU4REC(n_uid, n_mid, EMBEDDING_DIM, HIDDEN_SIZE, BATCH_SIZE, SEQ_LEN)
        elif model_type == 'DIN': 
            model = Model_DIN(n_uid, n_mid, EMBEDDING_DIM, HIDDEN_SIZE, BATCH_SIZE, SEQ_LEN)
        elif model_type == 'ARNN': 
            model = Model_ARNN(n_uid, n_mid, EMBEDDING_DIM, HIDDEN_SIZE, BATCH_SIZE, SEQ_LEN)
        elif model_type == 'RUM' :
            model = Model_RUM(n_uid, n_mid, EMBEDDING_DIM, HIDDEN_SIZE, BATCH_SIZE, Memory_Size, SEQ_LEN)
        elif model_type == 'DIEN': 
            model = Model_DIEN(n_uid, n_mid, EMBEDDING_DIM, HIDDEN_SIZE, BATCH_SIZE, SEQ_LEN)
        elif model_type == 'DIEN_with_aux': 
            model = Model_DIEN(n_uid, n_mid, EMBEDDING_DIM, HIDDEN_SIZE, BATCH_SIZE, SEQ_LEN, use_negsample=True)
        elif model_type == 'MIMN':
            model = Model_MIMN(n_uid, n_mid, EMBEDDING_DIM, HIDDEN_SIZE, BATCH_SIZE, Memory_Size, SEQ_LEN, Mem_Induction, Util_Reg, mask_flag=True) 
        elif model_type == 'MIMN_with_aux':
            model = Model_MIMN(n_uid, n_mid, EMBEDDING_DIM, HIDDEN_SIZE, BATCH_SIZE, Memory_Size, SEQ_LEN, Mem_Induction, Util_Reg, use_negsample=True, mask_flag=True) 
        else:
            print ("Invalid model_type : %s", model_type)
            return
        
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        sys.stdout.flush()
        print('training begin')
        sys.stdout.flush()

        start_time = time.time()
        iter = 0
        lr = 0.001
        for itr in range(epoch):
            print("epoch"+str(itr))
            loss_sum = 0.0
            accuracy_sum = 0.
            aux_loss_sum = 0.
            train_data_pool,_stop,_ = generator_queue(train_data)
            gamma = 0
            while True:
                if  _stop.is_set() and train_data_pool.empty():
                    break
                if not train_data_pool.empty():
                    src,tgt = train_data_pool.get()
                else:
                    continue
                nick_id, item_id, cate_id, label, hist_item, hist_cate, neg_item, neg_cate, hist_mask = prepare_data(src, tgt)
                loss, acc, aux_loss = model.train(sess, [nick_id, item_id, cate_id, hist_item, hist_cate, neg_item, neg_cate, hist_mask, label, lr, gamma])
                loss_sum += loss
                accuracy_sum += acc
                aux_loss_sum += aux_loss
                iter += 1
                sys.stdout.flush()
                if (iter % 200 ) == 0:
                    print("iter: %d" %iter)
                if iter > 2000:
                    gamma = 1
                if iter < start_test_iter:
                    continue
                if (iter % test_iter) == 0:
                    print('iter: %d ----> train_loss: %.4f ---- train_accuracy: %.4f ---- train_aux_loss: %.4f' % \
                                          (iter, float(loss_sum / test_iter), accuracy_sum / test_iter,  aux_loss_sum / test_iter))
                    print('test_auc: %.4f ----test_loss: %.4f ---- test_accuracy: %.4f ---- test_aux_loss: %.4f' % eval(sess, test_data, model, best_model_path, batch_size))
                    loss_sum = 0.0
                    accuracy_sum = 0.0
                    aux_loss_sum = 0.0
                    test_time = time.time()
                    print("test interval: "+str((test_time-start_time)/60.0)+" min")
                if (iter % save_iter) == 0:
                    print('save model iter: %d' %(iter))
                    model.save(sess, model_path+"--"+str(iter))

def test(
        train_file = "./data/taobao_data/taobao_train.txt",
        test_file = "./data/taobao_data/taobao_test.txt",
        feature_file = "./data/taobao_data/taobao_feature.pkl",
        batch_size = 256,
        maxlen = 200,
        test_iter = 100,
        save_iter = 100,
        model_type = 'DNN',
        Memory_Size = 4,
        Mem_Induction = 0, 
        Util_Reg = 0
):
    
    if model_type != "MIMN" or model_type != "MIMN_with_aux":
        model_path = "dnn_best_model/book_ckpt_noshuff" + model_type
    else:
        model_path = "dnn_best_model/book_ckpt_noshuff" + model_type+str(Memory_Size)+str(Mem_Induction)

    gpu_options = tf.GPUOptions(allow_growth=True)

    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        test_data = DataIterator(test_file, batch_size, maxlen)
        feature_num = pkl.load(open(feature_file))
        n_uid, n_mid = feature_num, feature_num
        BATCH_SIZE = batch_size
        SEQ_LEN = maxlen

        print("******start to test  " + model_type + "  **********")
        if model_type == 'DNN': 
            model = Model_DNN(n_uid, n_mid, EMBEDDING_DIM, HIDDEN_SIZE, BATCH_SIZE, SEQ_LEN)
        elif model_type == 'CDNN': 
            model = Model_CDNN(n_uid, n_mid, EMBEDDING_DIM, HIDDEN_SIZE, BATCH_SIZE, SEQ_LEN)
        elif model_type == 'PNN': 
            model = Model_PNN(n_uid, n_mid, EMBEDDING_DIM, HIDDEN_SIZE, BATCH_SIZE, SEQ_LEN)
        elif model_type == 'GRU4REC': 
            model = Model_GRU4REC(n_uid, n_mid, EMBEDDING_DIM, HIDDEN_SIZE, BATCH_SIZE, SEQ_LEN)
        elif model_type == 'MIND': 
            model = Model_MIND(n_uid, n_mid, EMBEDDING_DIM, HIDDEN_SIZE, BATCH_SIZE, SEQ_LEN)
        elif model_type == 'DIN': 
            model = Model_DIN(n_uid, n_mid, EMBEDDING_DIM, HIDDEN_SIZE, BATCH_SIZE, SEQ_LEN)
        elif model_type == 'ARNN': 
            model = Model_ARNN(n_uid, n_mid, EMBEDDING_DIM, HIDDEN_SIZE, BATCH_SIZE, SEQ_LEN)
        elif model_type == 'RUM' :
            model = Model_RUM(n_uid, n_mid, EMBEDDING_DIM, HIDDEN_SIZE, BATCH_SIZE,  Memory_Size, SEQ_LEN)
        elif model_type == 'DIEN': 
            model = Model_DIEN(n_uid, n_mid, EMBEDDING_DIM, HIDDEN_SIZE, BATCH_SIZE, SEQ_LEN)
        elif model_type == 'DIEN_with_aux': 
            model = Model_DIEN(n_uid, n_mid, EMBEDDING_DIM, HIDDEN_SIZE, BATCH_SIZE, SEQ_LEN, use_negsample=True)
        elif model_type == 'MIMN':
            model = Model_MIMN(n_uid, n_mid, EMBEDDING_DIM, HIDDEN_SIZE, BATCH_SIZE, Memory_Size, SEQ_LEN, Mem_Induction, Util_Reg, mask_flag=True) 
        elif model_type == 'MIMN_with_aux':
            model = Model_MIMN(n_uid, n_mid, EMBEDDING_DIM, HIDDEN_SIZE, BATCH_SIZE, Memory_Size, SEQ_LEN, Mem_Induction, Util_Reg, use_negsample=True, mask_flag=True) 
        else:
            print ("Invalid model_type : %s", model_type)
            return

        model.restore(sess, model_path)
        print('test_auc: %.4f ----test_loss: %.4f ---- test_accuracy: %.4f ---- test_aux_loss: %.4f' % eval(sess, test_data, model, model_path, batch_size))

if __name__ == '__main__':
    print sys.argv
    args = parser.parse_args()
    SEED = args.random_seed
    Model_Type = args.model_type
    Memory_Size = args.memory_size
    Mem_Induction = args.mem_induction
    Util_Reg = args.util_reg
    seq_maxlen = args.seq_maxlen
    seq_minlen = args.seq_minlen
    cluster_num =  args.cluster_num
    start_test_iter = args.start_test_iter
    test_iter = args.test_iter
    epoch = args.epoch

    tf.set_random_seed(SEED)
    numpy.random.seed(SEED)
    random.seed(SEED)
    print("\33[33m Model_type:%s   seq_maxlen:%d   seq_minlen:%d   cluster_num:%d \33[0m" %(Model_Type, seq_maxlen, seq_minlen, cluster_num))

    if args.p == 'train':
        train(model_type=Model_Type, Memory_Size=Memory_Size, Mem_Induction=Mem_Induction, Util_Reg=Util_Reg, maxlen=seq_maxlen, minlen=seq_minlen, cluster_num=cluster_num, start_test_iter=start_test_iter, test_iter=test_iter, epoch=epoch)
    elif args.p == 'test':
        test(model_type=Model_Type, Memory_Size=Memory_Size, Mem_Induction=Mem_Induction, Util_Reg=Util_Reg)
    else:
        print('do nothing...')
