# -*- coding: utf-8 -*-
"""
Created on Sat Jul 11 15:43:43 2020

@author: YYH19@yyh.hfut@gmail.com
"""

import os,pdb
import pandas as pd
from collections import defaultdict
import tensorflow as tf
import numpy as np
import random, math
from time import time
import shutil, utils
import multiprocessing as mp
from numpy.random import seed
seed(2020)
from tensorflow import set_random_seed
set_random_seed(2021)


test_flag = 1
runid = 0
version = 1
device_1 = 1
layer = 3
dimension = 32
learning_rate = 0.0005
epochs = 150
start_epoch = 80
batch_size = 1280 * 8
lamda = 0.0005
gama = 0.1
alpha = 0.1
user_count = 31027
item_count = 33899
pre_flag = 1
add_count = 1000

if test_flag == 0:
    topk_u, topk_v = 1, 1
else:
    topk_u, topk_v = 2, 2


cur_ckpt = '../../saved_models/amazon_model/pretrain_model/epoch119.ckpt'
best_model = '../../saved_model/amazon_model/best_model/epoch112.ckpt'


### record results ###
base_path = '../../saved_model/amazon_model/runid_'+str(runid)+'/'
model_save_path = base_path + 'models/'
if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)
if test_flag == 0:
    train_txt = open(base_path+'loss.txt','a')
    shutil.copyfile('egln.py', base_path+'egln.py')
evaluate_txt = open(base_path+'evaluate.txt', 'a')
  

    
### read data ###
t1 = time()
data_path = '../../datasets/amazon_data/'
user_items = np.load(data_path+'user_items.npy', allow_pickle=True).tolist()
traindata = np.load(data_path+'traindata.npy', allow_pickle=True).tolist()
valdata = np.load(data_path+'valdata.npy', allow_pickle=True).tolist()
testdata = np.load(data_path+'testdata.npy', allow_pickle=True).tolist()
userset = set(range(user_count))
itemset = set(range(item_count))
t2 = time()
print('load data cost time:',round(t2-t1, 4))
print('layer:',layer)



def get_train_adj_matrix(train_rating):
    '''
    get adjacent matrix of traindata#
    '''
    item_user_train = defaultdict(set)
    for key in train_rating.keys():
        for i in train_rating[key]:
            item_user_train[i].add(key)
    user_item_indexs, user_item_values = [], []
    item_user_indexs, item_user_values = [], []

    for x in train_rating.keys():
        for y in train_rating[x]:
            user_item_indexs.append([x, y])
            user_item_values.append(1.0)
    for x in item_user_train.keys():
        for y in item_user_train[x]:
            item_user_indexs.append([x,y])
            item_user_values.append(1.0)
    user_item_sparse_matrix = tf.SparseTensor(indices=user_item_indexs, values=user_item_values, dense_shape=[user_count, item_count])
    item_user_sparse_matrix = tf.SparseTensor(indices=item_user_indexs, values=item_user_values, dense_shape=[item_count, user_count])
    user_item_dense_matrix = tf.sparse_tensor_to_dense(user_item_sparse_matrix, default_value=0, validate_indices=False, name=None)
    user_item_indexs = np.reshape(np.array(user_item_indexs), [-1,2])
    all_user_list = np.reshape(user_item_indexs[:,0], [-1,1])
    all_item_list = np.reshape(user_item_indexs[:,1], [-1,1])
    return user_item_sparse_matrix, item_user_sparse_matrix, user_item_dense_matrix, all_user_list, all_item_list
user_item_adj_matrix, item_user_adj_matrix, adj_matrix_dense,  all_user_list, all_item_list = get_train_adj_matrix(traindata)


def get_simi_matrix_old(user_matrix, item_matrix, w1, w2, adj_matrix):

    user_rep = tf.matmul(user_matrix, w1)
    item_rep = tf.matmul(item_matrix, w2)
    user_emb1 = tf.nn.l2_normalize(user_rep, axis=1)
    item_emb1 = tf.nn.l2_normalize(item_rep, axis=1)
    sim_matrix = tf.nn.sigmoid(tf.matmul(user_emb1, tf.transpose(item_emb1))) #[m,n]  
    loss_simi_adj = gama*tf.reduce_mean(tf.square(sim_matrix-adj_matrix))      
    
    user_topk = tf.nn.top_k(sim_matrix, topk_u)
    user_topk_values = tf.reshape(user_topk.values,[-1])
    user_topk_columns = tf.cast(tf.reshape(user_topk.indices, [-1,1]), dtype=tf.int64)
    user_all_rows = np.reshape(np.arange(user_count), [-1,1])
    user_topk_rows = tf.reshape(tf.tile(user_all_rows,multiples= [1,topk_u]), [-1,1])
    user_topk_indexs = tf.concat([user_topk_rows, user_topk_columns], 1)
    user_item_sparse_simi = tf.SparseTensor(indices=user_topk_indexs, values=user_topk_values, dense_shape=[user_count, item_count])    

    item_topk = tf.nn.top_k(tf.transpose(sim_matrix), topk_v)
    item_topk_values = tf.reshape(item_topk.values, [-1])
    item_topk_columns = tf.cast(tf.reshape(item_topk.indices, [-1,1]), dtype=tf.int64)
    item_all_rows = np.reshape(np.arange(item_count), [-1,1])
    item_topk_rows = tf.reshape(tf.tile(item_all_rows,multiples= [1,topk_v]), [-1,1])
    item_topk_indexs = tf.concat([item_topk_rows, item_topk_columns], 1)
    item_user_sparse_simi = tf.SparseTensor(indices=item_topk_indexs, values=item_topk_values, dense_shape=[item_count, user_count])    
    return user_item_sparse_simi, item_user_sparse_simi, loss_simi_adj, user_topk_indexs


########################################### test part ############################################
def _init(_test_ratings, _all_ratings, _topk_list, _predictions):
    global test_ratings, all_ratings, topk_list, predictions
    test_ratings = _test_ratings
    all_ratings = _all_ratings
    topk_list = _topk_list
    predictions = _predictions


def get_one_performance(_uid):
    u = _uid
    metrics = {}
    pos_index = list(test_ratings[u])
    pos_length = len(test_ratings[u])
    neg_index = list(itemset-set(all_ratings[u]))
    pos_index.extend(neg_index)        
    pre_one = predictions[u][pos_index] 
    indices = utils.largest_indices(pre_one, topk_list[-1])
    indices=list(indices[0])
    for topk in topk_list:
        hit_value = 0
        dcg_value = 0  
        for idx in range(topk):
            ranking = indices[idx]
            if ranking < pos_length:
                hit_value += 1
                dcg_value += math.log(2) / math.log(idx+2) 
        target_length = min(topk,pos_length)
        hr_cur = hit_value/target_length
        ndcg_cur = dcg_value/utils.get_idcg(target_length)
        metrics[topk] = {'hr': hr_cur, 'ndcg':ndcg_cur}
    return metrics


def evaluate(_testdata, _user_items, _topk_list):
    hr_topk_list = defaultdict(list)
    ndcg_topk_list = defaultdict(list)
    hr_out, ndcg_out = {}, {}
    user_matrix, item_matrix = sess.run([final_user_emb, final_item_emb])
    _predictions = np.matmul(user_matrix, item_matrix.T)
    test_users = _testdata.keys()
    with mp.Pool(processes=10, initializer=_init, initargs=(_testdata, _user_items, _topk_list, _predictions)) as pool:
        all_metrics = pool.map(get_one_performance, test_users)
    for i, one_metrics in enumerate(all_metrics):
        for topk in _topk_list:
            hr_topk_list[topk].append(one_metrics[topk]['hr'])
            ndcg_topk_list[topk].append(one_metrics[topk]['ndcg'])
    for topk in _topk_list:
        hr_out[topk] = np.mean(hr_topk_list[topk])
        ndcg_out[topk] = np.mean(ndcg_topk_list[topk])
    return hr_out, ndcg_out


def user_group_test(test_ratings, all_ratings, topk=10):
    '''
#    用来比较不同spasity下user_group的测试结果
    '''
    user_matrix, item_matrix =  sess.run([final_user_emb, final_item_emb])
    all_hr_list = defaultdict(list)
    all_ndcg_list = defaultdict(list)
    hr_out = {}
    ndcg_out = {}
    user_group = np.load(data_path+'user_group.npy', allow_pickle=True).tolist()
    ratings = user_matrix.dot(item_matrix.T)
    for group_idx in range(len(user_group)):
        group_data = user_group[group_idx]
        for u in group_data:
            if u in test_ratings.keys():
                pos_index = list(test_ratings[u])
                pos_length = len(test_ratings[u])
                neg_index = list(itemset-set(all_ratings[u]))
                pos_index.extend(neg_index)        
                pre_one=ratings[u][pos_index] 
                indices=utils.largest_indices(pre_one, topk)
                indices=list(indices[0]) 
                hit_value = 0
                dcg_value = 0  
                for idx in range(topk):
                    ranking = indices[idx]
                    if ranking < pos_length:
                        hit_value += 1
                        dcg_value += math.log(2) / math.log(idx+2) 
                target_length = min(topk, pos_length) 
                all_hr_list[group_idx].append(hit_value/target_length)
                idcg_value = utils.utils.get_idcg(target_length)
                all_ndcg_list[group_idx].append(dcg_value/idcg_value)    
    for group_idx in range(len(user_group)):
        hr_out[group_idx] = round(sum(all_hr_list[group_idx])/len(all_hr_list[group_idx]), 5)
        ndcg_out[group_idx] = round(sum(all_ndcg_list[group_idx])/len(all_hr_list[group_idx]), 5)
        print('group_idx', group_idx, 'hr:', hr_out[group_idx], 'ndcg:',ndcg_out[group_idx])
    return hr_out, ndcg_out
   
    
########################################### construct model ###########################################
os.environ["CUDA_VISIBLE_DEVICES"] = str(device_1)
user_emb = tf.Variable(tf.random.normal([user_count, dimension], stddev=0.01, dtype=tf.float32), name='user_emb')
item_emb = tf.Variable(tf.random.normal([item_count, dimension], stddev=0.01, dtype=tf.float32), name='item_emb')
sim_w1 = tf.Variable(tf.random.normal([dimension, 16], dtype=tf.float32), name='sim_w1')
sim_w2 = tf.Variable(tf.random.normal([dimension, 16], dtype=tf.float32), name='sim_w2')
discri_w1 = tf.Variable(tf.random_normal([64, 32], stddev=0.01), name='discri_w1')
discri_b1 = tf.Variable(tf.zeros([32]), name='discri_b1') 
discri_w2 = tf.Variable(tf.random_normal([32, 1], stddev=0.01), name='discri_w2')
discri_b2 = tf.Variable(tf.zeros([1]), name='discri_b2') 
discri_w3 = tf.Variable(tf.random_normal([64, 32], stddev=0.01), name='discri_w3')
discri_b3 = tf.Variable(tf.zeros([32]), name='discri_b3') 
discri_w4 = tf.Variable(tf.random_normal([32, 1], stddev=0.01), name='discri_w4')
discri_b4 = tf.Variable(tf.zeros([1]), name='discri_b4') 
bilinear_w = tf.Variable(tf.random_normal([64, 64], stddev=0.01), name='bilinear_w')
bilinear_b = tf.Variable(tf.zeros([1]), name='bilinear_b') 
val_dict = {'user_emb':user_emb, 'item_emb':item_emb,
            'sim_w1':sim_w1, 'sim_w2':sim_w2,
            'discri_w1':discri_w1, 'discri_b1':discri_b1, 
            'discri_w2':discri_w2, 'discri_b2':discri_b2,
            'discri_w3':discri_w3, 'discri_b3':discri_b3,
            'discri_w4':discri_w4, 'discri_b4':discri_b4,
            'discri_w3':discri_w3, 'discri_b3':discri_b3,
            'bilinear_w':bilinear_w, 'bilinear_b':bilinear_b}
   
 
################################### get adaptive adjacent matrix #######################################
user_item_simi_matrix, item_user_simi_matrix, loss_s, add_edges = get_simi_matrix_old(user_emb, item_emb, sim_w1, sim_w2, adj_matrix_dense)
add_sparse_user_matrix = tf.sparse_add(user_item_adj_matrix, user_item_simi_matrix)
add_sparse_item_matrix = tf.sparse_add(item_user_adj_matrix, item_user_simi_matrix)
user_item_final_matrix = utils.nor_sparse_matrix(add_sparse_user_matrix)
item_user_final_matrix = utils.nor_sparse_matrix(add_sparse_item_matrix)


################################### gcn model input feature matrix ######################################
def model_gcn_with_feature(_user_emb, _item_emb, _layer):
    all_user_emb, all_item_emb = [_user_emb], [_item_emb]
    for _ in range(_layer):
        tmp_user_emb = tf.sparse_tensor_dense_matmul(user_item_final_matrix, all_item_emb[-1]) + all_user_emb[-1]
        tmp_item_emb = tf.sparse_tensor_dense_matmul(item_user_final_matrix, all_user_emb[-1]) + all_item_emb[-1]
        all_user_emb.append(tmp_user_emb)
        all_item_emb.append(tmp_item_emb)
    return all_user_emb[-1], all_item_emb[-1]
final_user_emb,final_item_emb = model_gcn_with_feature(user_emb, item_emb, layer)
_shuffle_user_emb, _shuffle_item_emb = utils.shuffle_embedding(user_emb, item_emb, dimension, 16)
shuffle_user_emb, shuffle_item_emb = model_gcn_with_feature(_shuffle_user_emb, _shuffle_item_emb, layer)


################################## gcn model input adjacent matrix ######################################
def model_gcn_with_structure(_user_item_matrix, _item_user_matrix, _layer):
    all_user_emb, all_item_emb = [user_emb], [item_emb]
    for _ in range(_layer):
        tmp_user_emb = tf.sparse_tensor_dense_matmul(_user_item_matrix, all_item_emb[-1]) + all_user_emb[-1]
        tmp_item_emb = tf.sparse_tensor_dense_matmul(_item_user_matrix, all_user_emb[-1]) + all_item_emb[-1]
        all_user_emb.append(tmp_user_emb)
        all_item_emb.append(tmp_item_emb)
    return all_user_emb[-1], all_item_emb[-1]
user_item_cor_matrix, item_user_cor_matrix = utils.graph_random(user_count, item_count, add_count)
add_uv_cor_matrix = tf.sparse_add(user_item_cor_matrix, add_sparse_user_matrix)
add_vu_cor_matrix = tf.sparse_add(item_user_cor_matrix, add_sparse_item_matrix)
nor_uv_cor_matrix = utils.nor_sparse_matrix(add_uv_cor_matrix)
nor_vu_cor_matrix = utils.nor_sparse_matrix(add_vu_cor_matrix)
cur_user_emb, cur_item_emb = model_gcn_with_structure(nor_uv_cor_matrix, nor_vu_cor_matrix, layer)


########################################### rating loss #################################################
u_input = tf.placeholder("int32", [None, 1])
i_input = tf.placeholder("int32", [None, 1])
j_input = tf.placeholder("int32", [None, 1])
ua = tf.gather_nd(final_user_emb, u_input)
vi = tf.gather_nd(final_item_emb, i_input)
vj = tf.gather_nd(final_item_emb, j_input)
Rai = tf.reduce_sum(tf.multiply(ua, vi), 1, keepdims=True)
Raj = tf.reduce_sum(tf.multiply(ua, vj), 1, keepdims=True)
auc = tf.reduce_mean(tf.to_float((Rai-Raj)>0))
bprloss = -tf.reduce_mean(tf.log(tf.clip_by_value(tf.nn.sigmoid(Rai-Raj),1e-9,1.0)))
regulation = lamda * tf.reduce_mean(tf.square(ua)+tf.square(vi)+tf.square(vj)) 
loss_r = bprloss + regulation
    

####################################### discriminator loss ##############################################
def make_discriminator_bilinear(_lo_emb, _gl_emb):
    '''
    input: _lo_emb[None,64], _gl_emb[None,64]
    output: label[None,1]
    '''
    emb_d1 = tf.matmul(_lo_emb, bilinear_w)
    emb_d2 = tf.multiply(emb_d1, _gl_emb)
    emb_d3 = tf.reduce_sum(emb_d2, 1, keepdims=True) + bilinear_b
    return emb_d3


def local_global_v1():
    '''
    pos:<u,i>, neg:<u,j>
    ''' 
    pos_local_emb = tf.concat([tf.sigmoid(ua), tf.sigmoid(vi)], 1)
    neg_local_emb = tf.concat([tf.sigmoid(ua), tf.sigmoid(vj)], 1)    
#    avg_global_emb = tf.reduce_mean(pos_local_emb, 0, keepdims=True)
    
    add_user_list, add_item_list = tf.split(add_edges, 2, axis=1)
    all_user_emb_1 = tf.gather_nd(final_user_emb, all_user_list) #[,32]
    all_item_emb_1 = tf.gather_nd(final_item_emb, all_item_list) #[,32]
    all_user_emb_2 = tf.gather_nd(final_user_emb, add_user_list)
    all_item_emb_2 = tf.gather_nd(final_item_emb, add_item_list)
    all_user_emb = tf.concat([all_user_emb_1, all_user_emb_2], 0)
    all_item_emb = tf.concat([all_item_emb_1, all_item_emb_2], 0)
    all_edge_emb = tf.concat([tf.sigmoid(all_user_emb), tf.sigmoid(all_item_emb)], 1) #[,64]
    avg_global_emb = tf.reduce_mean(all_edge_emb, 0, keepdims=True)
    
    get_shape = tf.reduce_sum(ua, 1, keepdims=True)
    global_emb = tf.tile(avg_global_emb, [batch_size, 1])
    one_label = tf.ones_like(get_shape, dtype=tf.float32)
    zero_label = tf.zeros_like(get_shape, dtype=tf.float32)
    real_predict = make_discriminator_bilinear(pos_local_emb, global_emb)
    fake_predict = make_discriminator_bilinear(neg_local_emb, global_emb)
    d_loss_all = tf.nn.sigmoid_cross_entropy_with_logits(logits=real_predict, labels=one_label) + \
                tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_predict, labels=zero_label)          
    loss_d = alpha * tf.reduce_mean(d_loss_all)
    return loss_d


def local_global_v2():
    '''
    G'=(A,F')
    '''
    pos_local_emb = tf.concat([tf.sigmoid(ua), tf.sigmoid(vi)], 1)
    _ua = tf.gather_nd(shuffle_user_emb, u_input)
    _vi = tf.gather_nd(shuffle_item_emb, i_input)
    neg_local_emb = tf.concat([tf.sigmoid(_ua), tf.sigmoid(_vi)], 1)
    avg_global_emb = tf.reduce_mean(pos_local_emb, 0, keepdims=True)  
    
    get_shape = tf.reduce_sum(ua, 1, keepdims=True)
    global_emb = tf.tile(avg_global_emb, [batch_size, 1])
    one_label = tf.ones_like(get_shape, dtype=tf.float32)
    zero_label = tf.zeros_like(get_shape, dtype=tf.float32)
    real_predict = make_discriminator_bilinear(pos_local_emb, global_emb)
    fake_predict = make_discriminator_bilinear(neg_local_emb, global_emb)
    d_loss_all = tf.nn.sigmoid_cross_entropy_with_logits(logits=real_predict, labels=one_label) + \
                tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_predict, labels=zero_label)          
    loss_d = alpha * tf.reduce_mean(d_loss_all)
    return loss_d


def local_global_v3():
    '''
    G'=(A',F)
    '''
    pos_local_emb = tf.concat([tf.sigmoid(ua), tf.sigmoid(vi)], 1)
    _ua = tf.gather_nd(cur_user_emb, u_input)
    _vi = tf.gather_nd(cur_item_emb, i_input)
    neg_local_emb = tf.concat([tf.sigmoid(_ua), tf.sigmoid(_vi)], 1)
    avg_global_emb = tf.reduce_mean(pos_local_emb, 0, keepdims=True)
    
    get_shape = tf.reduce_sum(ua, 1, keepdims=True)
    global_emb = tf.tile(avg_global_emb, [batch_size, 1])
    one_label = tf.ones_like(get_shape, dtype=tf.float32)
    zero_label = tf.zeros_like(get_shape, dtype=tf.float32)
    real_predict = make_discriminator_bilinear(pos_local_emb, global_emb)
    fake_predict = make_discriminator_bilinear(neg_local_emb, global_emb)
    d_loss_all = tf.nn.sigmoid_cross_entropy_with_logits(logits=real_predict, labels=one_label) + \
                tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_predict, labels=zero_label)          
    loss_d = alpha * tf.reduce_mean(d_loss_all)
    return loss_d

if version == 1:
    loss_d = local_global_v1()
if version == 2:
    loss_d = local_global_v2()
if version == 3:
    loss_d = local_global_v3()
loss = loss_r + loss_s +loss_d
opt = tf.compat.v1.train.AdamOptimizer(learning_rate).minimize(loss)

with tf.device('CPU'):
    opt = tf.compat.v1.train.AdamOptimizer(learning_rate).minimize(loss)

######################################## start tensorflow session ########################################
init = tf.global_variables_initializer()
saver = tf.train.Saver(val_dict, max_to_keep=0)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)
sess.run(init)
if pre_flag == 1:
    saver.restore(sess, cur_ckpt)


if test_flag == 0:
    ### start train ###
    print('*****start train*****')
    for epoch in range(epochs):
        # train part
        tt1 = time()
        t_train = utils.get_bpr_data(user_items, traindata, item_count, 1)
        indexs = np.arange(t_train.shape[0])
        np.random.shuffle(indexs)
        sum_auc,sum_loss1,sum_loss2,sum_loss3,sum_train = 0, 0, 0, 0, 0 
        for k in range(int(t_train.shape[0]/batch_size)+1):
            start_index = k*batch_size
            end_index = min(t_train.shape[0], (k+1)*batch_size)
            if end_index == t_train.shape[0]:
                start_index = end_index - batch_size
            triple_data = t_train[indexs[start_index:end_index]]
            u_list, i_list, j_list = triple_data[:,0], triple_data[:,1], triple_data[:,2]
            _auc,_loss1,_loss2,_loss3,_ = sess.run([auc,loss_s,loss_r,loss_d,opt], \
                                    feed_dict={u_input:np.reshape(u_list,[-1,1]),
                                               i_input:np.reshape(i_list,[-1,1]), 
                                               j_input:np.reshape(j_list,[-1,1])})
            sum_auc += _auc * len(u_list)
            sum_loss1 += _loss1 * len(u_list)
            sum_loss2 += _loss2 * len(u_list)
            sum_loss3 += _loss3 * len(u_list)
            sum_train += len(u_list)           
        mean_auc = sum_auc/sum_train
        mean_loss1 = sum_loss1/sum_train
        mean_loss2 = sum_loss2/sum_train
        mean_loss3 = sum_loss3/sum_train
        mean_loss = mean_loss1+mean_loss2+mean_loss3
        print('epoch:{:d}, trainauc:{:.4f}, s_loss:{:.4f}, r_loss:{:.4f}, d_loss:{:.4f}, trainloss:{:.4f}'
              .format(epoch, mean_auc, mean_loss1, mean_loss2, mean_loss3, mean_loss))
        train_txt.write('epoch:{:d}, trainauc:{:.4f}, s_loss:{:.4f}, r_loss:{:.4f}, d_loss:{:.4f}, trainloss:{:.4f}'
                        .format(epoch, mean_auc, mean_loss1, mean_loss2, mean_loss3, mean_loss)+ '\n')
        tt2 = time()
        
        #val part
        t_val = utils.get_bpr_data(user_items, valdata, item_count, 1)
        indexs = np.arange(t_val.shape[0])
        np.random.shuffle(indexs)
        sum_auc, sum_loss1, sum_loss2, sum_loss3, sum_train = 0, 0, 0, 0, 0 
        for k in range(int(t_val.shape[0]/batch_size)+1):
            start_index = k*batch_size
            end_index = min(t_val.shape[0], (k+1)*batch_size)
            if end_index == t_val.shape[0]:
                start_index = end_index - batch_size
            triple_data = t_val[indexs[start_index:end_index]]
            u_list, i_list, j_list = triple_data[:,0], triple_data[:,1], triple_data[:,2]
            _auc,_loss1,_loss2,_loss3 = sess.run([auc,loss_s,loss_r,loss_d], \
                                feed_dict={u_input:np.reshape(u_list,[-1,1]),
                                           i_input:np.reshape(i_list,[-1,1]), 
                                           j_input:np.reshape(j_list,[-1,1])})
            sum_auc += _auc * len(u_list)
            sum_loss1 += _loss1 * len(u_list)
            sum_loss2 += _loss2 * len(u_list)
            sum_loss3 += _loss3 * len(u_list)
            sum_train += len(u_list) 
        mean_auc = sum_auc/sum_train
        mean_loss1 = sum_loss1/sum_train
        mean_loss2 = sum_loss2/sum_train
        mean_loss3 = sum_loss3/sum_train
        mean_loss = mean_loss1+mean_loss2+mean_loss3
        print('epoch:{:d}, validauc:{:.4f}, s_loss:{:.4f}, r_loss:{:.4f}, d_loss:{:.4f}, validloss:{:.4f}'
              .format(epoch, mean_auc, mean_loss1, mean_loss2, mean_loss3, mean_loss))
        train_txt.write('epoch:{:d}, validauc:{:.4f}, s_loss:{:.4f}, r_loss:{:.4f}, d_loss:{:.4f}, validloss:{:.4f}'
                        .format(epoch, mean_auc, mean_loss1, mean_loss2, mean_loss3, mean_loss)+ '\n')
        tt3 = time()
        
        saver.save(sess, model_save_path+'epoch'+str(epoch)+'.ckpt')
        print('cost time:{:.4f}'.format(tt3-tt1),'\n')
        train_txt.write('cost time:{:.4f}'.format(tt3-tt1)+'\n\n')
    print('*****train over*****')
    train_txt.close()


######################################## start overall evaluate ##########################################   
if test_flag == 1:
    saver.restore(sess, best_model)
    _hr, _ndcg = evaluate(testdata, user_items, [5,10,15,20,25,30,35,40,45,50])
    for key in _hr.keys():
        print('topk:{:d}, hr{:.5f}, ndcg:{:.5f}'.format(key, _hr[key], _ndcg[key]))
        evaluate_txt.write('topk:{:d}, hr{:.5f}, ndcg:{:.5f}'.format(key, _hr[key], _ndcg[key]) + '\n')
    evaluate_txt.write('\n')    
    
else:
    maxndcg = 0
    for epoch in range(start_epoch, epochs):
        cur_ckpt = model_save_path+'epoch'+str(epoch)+'.ckpt'
        saver.restore(sess, cur_ckpt)  
        evaluate_txt.write('from the model' + cur_ckpt +'\n')
        print('from the model', cur_ckpt)
        _hr, _ndcg = evaluate(testdata, user_items, [5,10,15,20,25,30,35,40,45,50])
        maxndcg = max(_ndcg[10], maxndcg)
        if _ndcg[10] == maxndcg:
            best_model = cur_ckpt
        for key in _hr.keys():
            print('topk:{:d}, hr{:.5f}, ndcg:{:.5f}'.format(key, _hr[key], _ndcg[key]))
            evaluate_txt.write('topk:{:d}, hr{:.5f}, ndcg:{:.5f}'.format(key, _hr[key], _ndcg[key]) + '\n')
        evaluate_txt.write('\n')
        print('\n')
    evaluate_txt.write('\n')
    print('***evalate over***')
    print('the best epoch is', best_model)

