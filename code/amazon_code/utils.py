# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 13:52:44 2021

@author: Administrator
"""

import numpy as np
import math, random
import tensorflow as tf
from collections import defaultdict



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


def get_adj_matrix_lightgcn(train_rating, user_count, item_count):
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
        len_u = len(train_rating[x])
        for y in train_rating[x]:
            user_item_indexs.append([x, y])
            len_v = len(item_user_train[y])
            d_uv = pow(len_u, 0.5) * pow(len_v, 0.5)
            user_item_values.append(1.0/d_uv)
            item_user_indexs.append([y,x])
            item_user_values.append(1.0/d_uv)
    user_item_sparse_matrix = tf.SparseTensor(indices=user_item_indexs, values=user_item_values, dense_shape=[user_count, item_count])
    item_user_sparse_matrix = tf.SparseTensor(indices=item_user_indexs, values=item_user_values, dense_shape=[item_count, user_count])
    return user_item_sparse_matrix, item_user_sparse_matrix


def get_adj_matrix_gcn(train_rating, user_count, item_count):
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
        len_u = len(train_rating[x])
        for y in train_rating[x]:
            user_item_indexs.append([x, y])
            user_item_values.append(1.0/len_u)
    for x in item_user_train.keys():
        len_v = len(item_user_train[x])
        for y in item_user_train[x]:
            item_user_indexs.append([x,y])
            item_user_values.append(1.0/len_v)
    user_item_sparse_matrix = tf.SparseTensor(indices=user_item_indexs, values=user_item_values, dense_shape=[user_count, item_count])
    item_user_sparse_matrix = tf.SparseTensor(indices=item_user_indexs, values=item_user_values, dense_shape=[item_count, user_count])
    return user_item_sparse_matrix, item_user_sparse_matrix



def get_adj_matrix_lrgccf(train_rating, user_count, item_count):
    '''
    get adjacent matrix of traindata#
    '''
    userset = set(range(user_count))
    itemset = set(range(item_count))
    item_user_train = defaultdict(set)
    for key in train_rating.keys():
        for i in train_rating[key]:
            item_user_train[i].add(key)
    user_item_indexs, user_item_values = [], []
    item_user_indexs, item_user_values = [], []
    d_user, d_item = [], [] 
    for x in train_rating.keys():
        len_u = len(train_rating[x])
        for y in train_rating[x]:
            len_v = len(item_user_train[y])
            d_uv = pow(len_u, 0.5) * pow(len_v, 0.5)
            user_item_indexs.append([x, y])
            user_item_values.append(1.0/d_uv)
            item_user_indexs.append([y,x])
            item_user_values.append(1.0/d_uv)
    for x in userset:
        if x in train_rating.keys():
            d_user.append(1.0/len(train_rating[x]))
        else:
            d_user.append(0)
    for y in itemset:
        if y in item_user_train.keys():
            len_v = len(item_user_train[y])
            d_item.append(1.0/len_v)
        else:
            d_item.append(0)
    d_user = np.reshape(np.array(d_user), [-1,1])
    d_item = np.reshape(np.array(d_item), [-1,1])
    user_item_sparse_matrix = tf.SparseTensor(indices=user_item_indexs, values=user_item_values, dense_shape=[user_count, item_count])
    item_user_sparse_matrix = tf.SparseTensor(indices=item_user_indexs, values=item_user_values, dense_shape=[item_count, user_count])
    return user_item_sparse_matrix, item_user_sparse_matrix, d_user, d_item


def nor_sparse_matrix(sparse_matrix):
    sum_matrix = tf.sparse.reduce_sum(sparse_matrix, 1, keepdims=True)
    nor_matrix = tf.divide(sparse_matrix, sum_matrix)
    return nor_matrix
    

def shuffle_embedding(input_emb_u,input_emb_v,dimension,mid=16):
    fixed_emb_u, dynamic_emb_u = tf.split(input_emb_u, [mid, dimension-mid], 1) 
    fixed_emb_v, dynamic_emb_v = tf.split(input_emb_v, [mid, dimension-mid], 1)
    dynamic_emb_u = tf.gather(tf.transpose(dynamic_emb_u), tf.random.shuffle(tf.range(dimension-mid)))
    out_emb_u = tf.concat([fixed_emb_u, tf.transpose(dynamic_emb_u)], 1)
    dynamic_emb_v = tf.gather(tf.transpose(dynamic_emb_v), tf.random.shuffle(tf.range(dimension-mid)))
    out_emb_v = tf.concat([fixed_emb_v, tf.transpose(dynamic_emb_v)], 1)
#    out_emb_u = tf.concat([fixed_emb_u, tf.zeros_like(dynamic_emb_u)], 1)
#    out_emb_v = tf.concat([fixed_emb_v, tf.zeros_like(dynamic_emb_v)], 1)
    return out_emb_u, out_emb_v

def graph_random(user_count, item_count, count):
    '''
    随机加上n条边
    '''
    user_item_indexs, item_user_indexs = [], []
    user_item_values, item_user_values = [], []
    for _ in range(count):
        u = random.randint(0, user_count-1)
        v = random.randint(0, item_count-1)
        user_item_indexs.append([u,v])
        item_user_indexs.append([v,u])
        user_item_values.append(1.0)
        item_user_values.append(1.0)
    user_item_sparse_matrix = tf.SparseTensor(indices=user_item_indexs, values=user_item_values, dense_shape=[user_count, item_count])
    item_user_sparse_matrix = tf.SparseTensor(indices=item_user_indexs, values=item_user_values, dense_shape=[item_count, user_count])
    return user_item_sparse_matrix, item_user_sparse_matrix


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