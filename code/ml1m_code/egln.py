# -*- coding: utf-8 -*-
"""
Created on Sat Jul 11 15:43:43 2020

@author: YYH19@yyh/hfut@gmail.com
"""

import os,pdb
import pandas as pd
from collections import defaultdict
import tensorflow as tf
import numpy as np
import random, math
from time import time
from shutil import copyfile
import shutil
import multiprocessing as mp
from numpy.random import seed
seed(2020)
from tensorflow import set_random_seed
set_random_seed(2021)


### parameters ###
test_flag = 0
version = 1
runid = 10
gcn_layer = 0  ###测试计算residual_graph的时候送入gcn哪一层的embedding效果最好
layer = 2
device_id = 1
dimension = 32
learning_rate = 0.001
epochs = 800
batch_size = 1024 * 20
lamda = 0.5 * 1e-3
gama = 0.1
alpha = 0.1
user_count = 6040
item_count = 3952
shuffle_rate = 0.9

if test_flag == 0:
    topk_u, topk_v = 4, 6
else:
    topk_u, topk_v = 4, 30



best_model = '../../saved_models/ml1m_model/epoch_791_ndcg_0.17202843767029785.ckpt'


### record results ###
base_path = '../../saved_models/ml1m_model/runid_'+str(runid)+'/'
if test_flag == 0:
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    train_txt = open(base_path+'loss.txt','a')
    model_save_path = base_path + 'models/'
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    copyfile('egln.py', base_path+'egln.py')
evaluate_txt = open(base_path+'evaluate.txt','a')



### read data ###
t1 = time()
data_path = '../../datasets/ml1m_data/'
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
    item_user_dense_matrix = tf.sparse_tensor_to_dense(item_user_sparse_matrix, default_value=0, validate_indices=False, name=None)
    user_item_indexs = np.reshape(np.array(user_item_indexs), [-1,2])
    all_user_list = np.reshape(user_item_indexs[:,0], [-1,1])
    all_item_list = np.reshape(user_item_indexs[:,1], [-1,1])
    return user_item_sparse_matrix, item_user_sparse_matrix, user_item_dense_matrix, item_user_dense_matrix, all_user_list, all_item_list
user_item_adj_matrix, item_user_adj_matrix, adj_matrix_dense, item_user_dense_matrix, all_user_list, all_item_list = get_train_adj_matrix(traindata)



def get_bpr_data(rating_all, rating_train, item_count, neg_sample):
    '''
    get triple data [u,i,j]
    '''
    t = []
    for u in rating_train.keys():
        for i in rating_train[u]:
            for _ in range(neg_sample):
                j = random.randint(0, item_count-1)
                while j in rating_all[u]:
                    j = random.randint(0, item_count-1)
                t.append([u,i,j])
    return np.reshape(np.asarray(t), [-1,3]) 

   
    
def get_simi_matrix_old(user_matrix, item_matrix, w1, w2, adj_matrix, topk1, topk2, gama):

    user_rep = tf.matmul(user_matrix, w1)
    item_rep = tf.matmul(item_matrix, w2)       
    user_emb1 = tf.nn.l2_normalize(user_rep, axis=1)
    item_emb1 = tf.nn.l2_normalize(item_rep, axis=1)
    sim_matrix = tf.nn.sigmoid(tf.matmul(user_emb1, tf.transpose(item_emb1))) #[m,n]    

    loss_simi_adj = gama*tf.reduce_mean(tf.square(sim_matrix-adj_matrix))
    user_topk = tf.nn.top_k(sim_matrix, topk1)
    user_topk_values = tf.reshape(user_topk.values,[-1])
    user_topk_columns = tf.cast(tf.reshape(user_topk.indices, [-1,1]), dtype=tf.int64)
    user_all_rows = np.reshape(np.arange(user_count), [-1,1])
    user_topk_rows = tf.reshape(tf.tile(user_all_rows,multiples= [1,topk1]), [-1,1])
    user_topk_indexs = tf.concat([user_topk_rows, user_topk_columns], 1)
    user_item_sparse_simi = tf.SparseTensor(indices=user_topk_indexs, values=user_topk_values, dense_shape=[user_count, item_count])    

    item_topk = tf.nn.top_k(tf.transpose(sim_matrix), topk2)
    item_topk_values = tf.reshape(item_topk.values, [-1])
    item_topk_columns = tf.cast(tf.reshape(item_topk.indices, [-1,1]), dtype=tf.int64)
    item_all_rows = np.reshape(np.arange(item_count), [-1,1])
    item_topk_rows = tf.reshape(tf.tile(item_all_rows,multiples= [1,topk2]), [-1,1])
    item_topk_indexs = tf.concat([item_topk_rows, item_topk_columns], 1)
    item_user_sparse_simi = tf.SparseTensor(indices=item_topk_indexs, values=item_topk_values, dense_shape=[item_count, user_count])    
    return user_item_sparse_simi, item_user_sparse_simi, loss_simi_adj, user_topk_indexs


def shuffle_embedding(input_emb_u,input_emb_v,rate):
    mid = int(dimension*rate)
    mid = 29
    fixed_emb_u, dynamic_emb_u = tf.split(input_emb_u, [mid, dimension-mid], 1) 
    dynamic_emb_u = tf.gather(tf.transpose(dynamic_emb_u), tf.random.shuffle(tf.range(dimension-mid)))
    out_emb_u = tf.concat([fixed_emb_u, tf.transpose(dynamic_emb_u)], 1)
    fixed_emb_v, dynamic_emb_v = tf.split(input_emb_v, [mid, dimension-mid], 1) 
    dynamic_emb_v = tf.gather(tf.transpose(dynamic_emb_v), tf.random.shuffle(tf.range(dimension-mid)))
    out_emb_v = tf.concat([fixed_emb_v, tf.transpose(dynamic_emb_v)], 1)
    return out_emb_u, out_emb_v


def graph_corruption(_rate):
    user_item_indexs, user_item_values = [], []
    item_user_indexs, item_user_values = [], []
    for u in traindata.keys():
        add_indexs = random.sample(itemset, int(_rate*len(itemset)))
        for v in add_indexs:
            user_item_indexs.append([u,v])
            item_user_indexs.append([v,u])
            user_item_values.append(1.0)
            item_user_values.append(1.0)
    user_item_sparse_matrix = tf.SparseTensor(indices=user_item_indexs, values=user_item_values, dense_shape=[user_count, item_count])
    item_user_sparse_matrix = tf.SparseTensor(indices=item_user_indexs, values=item_user_values, dense_shape=[item_count, user_count])
    return user_item_sparse_matrix, item_user_sparse_matrix  


################################################# test part #############################################################

        
def get_idcg(length):
    idcg = 0.0
    for i in range(length):
        idcg = idcg + math.log(2) / math.log(i + 2)
    return idcg


def largest_indices(ary, n):
    """Returns the n largest indices from a numpy array."""
    flat = ary.flatten()
    indices = np.argpartition(flat, -n)[-n:]
    indices = indices[np.argsort(-flat[indices])]
    return np.unravel_index(indices, ary.shape)


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
    indices = largest_indices(pre_one, topk_list[-1])
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
        ndcg_cur = dcg_value/get_idcg(target_length)
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
            pos_index = list(test_ratings[u])
            pos_length = len(test_ratings[u])
            neg_index = list(itemset-set(all_ratings[u]))
            pos_index.extend(neg_index)        
            pre_one=ratings[u][pos_index] 
            indices=largest_indices(pre_one, topk)
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
            idcg_value = get_idcg(target_length)
            all_ndcg_list[group_idx].append(dcg_value/idcg_value)    
    for group_idx in range(len(user_group)):
        hr_out[group_idx] = round(sum(all_hr_list[group_idx])/len(all_hr_list[group_idx]), 4)
        ndcg_out[group_idx] = round(sum(all_ndcg_list[group_idx])/len(all_hr_list[group_idx]), 4)
        print('group_idx', group_idx, 'hr:', hr_out[group_idx], 'ndcg:',ndcg_out[group_idx])
    return hr_out, ndcg_out


def nor_sparse_matrix(sparse_matrix):
    sum_matrix = tf.sparse.reduce_sum(sparse_matrix, 1, keepdims=True)
    nor_matrix = tf.divide(sparse_matrix, sum_matrix)
    return nor_matrix
    

# construct tensorflow graph
os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
user_emb = tf.Variable(tf.random.normal([user_count, dimension], stddev=0.01, dtype=tf.float32), name='user_emb')
item_emb = tf.Variable(tf.random.normal([item_count, dimension], stddev=0.01, dtype=tf.float32), name='item_emb')
sim_w1 = tf.Variable(tf.random.normal([dimension, 16], dtype=tf.float32), name='sim_w1')
sim_w2 = tf.Variable(tf.random.normal([dimension, 16], dtype=tf.float32), name='sim_w2')
bilinear_w1 = tf.Variable(tf.random_normal([64, 64], stddev=0.01), name='bilinear_w1')
bilinear_b1 = tf.Variable(tf.zeros([1]), name='bilinear_b1') 
bilinear_w2 = tf.Variable(tf.random_normal([32, 32], stddev=0.01), name='bilinear_w2')
bilinear_b2 = tf.Variable(tf.zeros([1]), name='bilinear_b2') 
val_dict = {'user_emb':user_emb, 'item_emb':item_emb, 'sim_w1':sim_w1, 'sim_w2':sim_w2,
            'bilinear_w1':bilinear_w1, 'bilinear_b1':bilinear_b1,
            'bilinear_w2':bilinear_w2, 'bilinear_b2':bilinear_b2}


def gcn_model(_uv_matrix, _vu_matrix, layer):
    all_user_emb, all_item_emb = [user_emb], [item_emb]
    for _ in range(layer):
        tmp_user_emb = tf.sparse.sparse_dense_matmul(_uv_matrix, all_item_emb[-1]) + all_user_emb[-1]
        tmp_item_emb = tf.sparse.sparse_dense_matmul(_vu_matrix, all_user_emb[-1]) + all_item_emb[-1]
        all_user_emb.append(tmp_user_emb)
        all_item_emb.append(tmp_item_emb)
    return all_user_emb[-1], all_item_emb[-1]

input_user_emb, input_item_emb = gcn_model(user_item_adj_matrix, item_user_adj_matrix, gcn_layer)
user_item_simi_matrix, item_user_simi_matrix, loss_s, add_edges = get_simi_matrix_old(input_user_emb, input_item_emb, sim_w1, sim_w2, adj_matrix_dense, topk_u, topk_v, gama)


# get adaptive adjacent matrix
#user_item_simi_matrix, item_user_simi_matrix, loss_s, add_edges = get_simi_matrix_old(user_emb, item_emb, sim_w1, sim_w2, adj_matrix_dense, topk_u, topk_v, gama)
add_sparse_user_matrix = tf.sparse_add(user_item_adj_matrix, user_item_simi_matrix)
add_sparse_item_matrix = tf.sparse_add(item_user_adj_matrix, item_user_simi_matrix)
user_item_final_matrix = nor_sparse_matrix(add_sparse_user_matrix)
item_user_final_matrix = nor_sparse_matrix(add_sparse_item_matrix)

# gcn model
def model_gcn(_user_emb, _item_emb, _layer):
    user_emb_layer1 = tf.sparse_tensor_dense_matmul(user_item_final_matrix, _item_emb) + _user_emb
    item_emb_layer1 = tf.sparse_tensor_dense_matmul(item_user_final_matrix, _user_emb) + _item_emb
    user_emb_layer2 = tf.sparse_tensor_dense_matmul(user_item_final_matrix, item_emb_layer1) + user_emb_layer1
    item_emb_layer2 = tf.sparse_tensor_dense_matmul(item_user_final_matrix, user_emb_layer1) + item_emb_layer1
    user_emb_layer3 = tf.sparse_tensor_dense_matmul(user_item_final_matrix, item_emb_layer2) + user_emb_layer2
    item_emb_layer3 = tf.sparse_tensor_dense_matmul(item_user_final_matrix, user_emb_layer2) + item_emb_layer2
    user_emb_layer4 = tf.sparse_tensor_dense_matmul(user_item_final_matrix, item_emb_layer3) + user_emb_layer3
    item_emb_layer4 = tf.sparse_tensor_dense_matmul(item_user_final_matrix, user_emb_layer3) + item_emb_layer3
    if _layer == 1:
        final_user_emb, final_item_emb = user_emb_layer1, item_emb_layer1
    if _layer == 2:
        final_user_emb, final_item_emb = user_emb_layer2, item_emb_layer2
    if _layer == 3:
        final_user_emb, final_item_emb = user_emb_layer3, item_emb_layer3
    if _layer == 4:
        final_user_emb, final_item_emb = user_emb_layer4, item_emb_layer4
    return final_user_emb, final_item_emb
final_user_emb,final_item_emb = model_gcn(user_emb, item_emb, layer)
_shuffle_user_emb, _shuffle_item_emb = shuffle_embedding(user_emb, item_emb, shuffle_rate)
shuffle_user_emb, shuffle_item_emb = model_gcn(_shuffle_user_emb, _shuffle_item_emb, layer)

 
# rating prediction part
user_input = tf.placeholder("int32", [None, 1])
item_input = tf.placeholder("int32", [None, 1])
latent_user = tf.gather_nd(final_user_emb, user_input)
latent_item = tf.gather_nd(final_item_emb, item_input)
latent_mul = tf.multiply(latent_user, latent_item)
predictions = tf.sigmoid(tf.reduce_sum(latent_mul, 1, keepdims=True))


# rating loss part
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


############### discriminator loss ###################
def make_discriminator_bilinear1(_lo_emb, _gl_emb):
    '''
    input: _lo_emb[None,64], _gl_emb[None,64]
    output: label[None,1]
    '''
    emb_d1 = tf.matmul(_lo_emb, bilinear_w1)
    emb_d2 = tf.multiply(emb_d1, _gl_emb)
    emb_d3 = tf.reduce_sum(emb_d2, 1, keepdims=True) + bilinear_b1
    return emb_d3


def make_discriminator_bilinear2(_lo_emb, _gl_emb):
    '''
    input: _lo_emb[None,32], _gl_emb[None,32]
    output: label[None,1]
    '''
    emb_d1 = tf.matmul(_lo_emb, bilinear_w2)
    emb_d2 = tf.multiply(emb_d1, _gl_emb)
    emb_d3 = tf.reduce_sum(emb_d2, 1, keepdims=True) + bilinear_b2
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
    real_predict = make_discriminator_bilinear1(pos_local_emb, global_emb)
    fake_predict = make_discriminator_bilinear1(neg_local_emb, global_emb)
    d_loss_all = tf.nn.sigmoid_cross_entropy_with_logits(logits=real_predict, labels=one_label) + \
                tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_predict, labels=zero_label)          
    loss_d = alpha * tf.reduce_mean(d_loss_all)
    return loss_d


def local_global_v2():
    pos_local_emb = tf.concat([tf.sigmoid(ua), tf.sigmoid(vi)], 1)
    _ua = tf.gather_nd(shuffle_user_emb, u_input)
    _vi = tf.gather_nd(shuffle_item_emb, i_input)
    neg_local_emb = tf.concat([tf.sigmoid(_ua), tf.sigmoid(_vi)], 1)
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
    real_predict = make_discriminator_bilinear1(pos_local_emb, global_emb)
    fake_predict = make_discriminator_bilinear1(neg_local_emb, global_emb)
    d_loss_all = tf.nn.sigmoid_cross_entropy_with_logits(logits=real_predict, labels=one_label) + \
                tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_predict, labels=zero_label)          
    loss_d = alpha * tf.reduce_mean(d_loss_all)
    return loss_d
    
    

if version == 1:
    loss_d = local_global_v1()
if version == 2:
    loss_d = local_global_v2()

loss = loss_r + loss_s +loss_d
opt = tf.compat.v1.train.AdamOptimizer(learning_rate).minimize(loss)


# start tensorflow session
init = tf.global_variables_initializer()
saver = tf.train.Saver(val_dict, max_to_keep=5)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)
sess.run(init)
#saver.restore(sess,best_model)



################# start train ###################
if test_flag == 0:
    print('start training...')
    trainauc,trainloss = [], []
    valauc,valloss = [], []
    valndcg = {}
    max_ndcg = 0
    for epoch in range(epochs):
        # train part
        t1 = time()
        t_train = get_bpr_data(user_items, traindata, item_count, 1)
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
        print('epoch:{:d}, trainauc:{:.4f}, loss_s:{:.4f}, loss_r:{:.4f}, loss_d:{:.4f}, trainloss:{:.4f}'
              .format(epoch, mean_auc, mean_loss1, mean_loss2, mean_loss3, mean_loss))
        train_txt.write('epoch:{:d}, trainauc:{:.4f}, loss_s:{:.4f}, loss_r:{:.4f}, loss_d:{:.4f}, trainloss:{:.4f}'
                        .format(epoch, mean_auc, mean_loss1, mean_loss2, mean_loss3, mean_loss)+ '\n')
        t2 = time()
        
        #val part
        t_val = get_bpr_data(user_items, valdata, item_count, 1)
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
        print('epoch:{:d}, validauc:{:.4f}, loss_s:{:.4f}, loss_r:{:.4f}, loss_d:{:.4f}, validloss:{:.4f}'
              .format(epoch, mean_auc, mean_loss1, mean_loss2, mean_loss3, mean_loss))
        train_txt.write('epoch:{:d}, validauc:{:.4f}, loss_s:{:.4f}, loss_r:{:.4f}, loss_d:{:.4f}, validloss:{:.4f}'
                        .format(epoch, mean_auc, mean_loss1, mean_loss2, mean_loss3, mean_loss)+ '\n')
        t3 = time()    
        
        _hr, _ndcg = evaluate(testdata, user_items, [5,10,15,20,25,30,35,40,45,50])
        valndcg[epoch] = _ndcg[10]
        max_ndcg = max(max_ndcg, _ndcg[10])
        if _ndcg[10] == max_ndcg:
            saver.save(sess, model_save_path+'epoch_'+str(epoch)+'_ndcg_'+str(_ndcg[10])+'.ckpt')
            best_ckpt = model_save_path+'epoch_'+str(epoch)+'_ndcg_'+str(_ndcg[10])+'.ckpt'
        print('hr@10:{:.5f}, ndcg@10:{:.5f}, train time:{:.4f}, test time:{:.4f}'.format(_hr[10], _ndcg[10], t2-t1, t3-t2), '\n')
        train_txt.write('hr@10:{:.5f}, ndcg@10:{:.5f}, train time:{:.4f}, test time:{:.4f}'.format(_hr[10], _ndcg[10], t2-t1, t3-t2)+'\n\n')
    print('*****train over*****')
    train_txt.close()

    print('best ckpt is:', best_ckpt)
    saver.restore(sess, best_ckpt)
    _hr, _ndcg = evaluate(testdata, user_items, [5,10,15,20,25,30,35,40,45,50])
    for key in _hr.keys():
        print('topk:{:d}, hr{:.5f}, ndcg:{:.5f}'.format(key, _hr[key], _ndcg[key]))
        evaluate_txt.write('topk:{:d}, hr{:.5f}, ndcg:{:.5f}'.format(key, _hr[key], _ndcg[key]) + '\n')
    evaluate_txt.close()

if test_flag == 1:
    print('best ckpt is:', best_ckpt)
    saver.restore(sess, best_ckpt)
    _hr, _ndcg = evaluate(testdata, user_items, [5,10,15,20,25,30,35,40,45,50])
    for key in _hr.keys():
        print('topk:{:d}, hr{:.5f}, ndcg:{:.5f}'.format(key, _hr[key], _ndcg[key]))
        evaluate_txt.write('topk:{:d}, hr{:.5f}, ndcg:{:.5f}'.format(key, _hr[key], _ndcg[key]) + '\n')
    evaluate_txt.close()
    evaluate_txt.close()
    
