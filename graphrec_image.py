#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from util import get_data100k,read_process,get_edgelist,pa_index
import networkx as nx
from networkx.algorithms import bipartite

import numpy as np
import pandas as pd
import time
from collections import deque

import tensorflow as tf
from six import next
from sklearn import preprocessing
import sys
from scipy.sparse import lil_matrix
from scipy.sparse import coo_matrix

BATCH_SIZE = 1000
USER_NUM = 943
ITEM_NUM = 1682
df_train, df_test = get_data100k()
#df_train, df_test = get_data1m()
##preferential attachment    
paUsers,paItems=pa_index(df_train,df_test)
################laplacian
la_movie,la_user,norm_la_movie,norm_la_user=laplacian_graph(df_train,df_test)

img=pd.read_csv('./data/cnn_100k.csv')
del img['Unnamed: 0']
img['path']-=1
img=img.set_index(np.arange(len(img)))
img_ma = np.zeros((ITEM_NUM,ITEM_NUM), dtype=np.float32) #np.asarray([[0 for x in range(USER_NUM)] for y in range(ITEM_NUM)],dtype=np.float16)
for index, row in img.iterrows():
    itemid=int(row['path'])
    img_ma[itemid]=row[1:]

AdjacencyUsers = np.zeros((USER_NUM,ITEM_NUM), dtype=np.float32) #np.asarray([[0 for x in range(ITEM_NUM)] for y in range(USER_NUM)],dtype=np.float16)
DegreeUsers_norm = np.zeros((USER_NUM,USER_NUM), dtype=np.float32)# np.asarray([[0 for x in range(1)] for y in range(USER_NUM)],dtype=np.float16)
AdjacencyItems = np.zeros((ITEM_NUM,USER_NUM), dtype=np.float32) #np.asarray([[0 for x in range(USER_NUM)] for y in range(ITEM_NUM)],dtype=np.float16)
DegreeItems_norm =  np.zeros((ITEM_NUM,ITEM_NUM), dtype=np.float32) #np.asarray([[0 for x in range(1)] for y in range(ITEM_NUM)],dtype=np.float16)
for index, row in df_train.iterrows():
    userid=int(row['user'])
    itemid=int(row['item'])
    AdjacencyUsers[userid][itemid]=row['rate']/5
    AdjacencyItems[itemid][userid]=row['rate']/5
    DegreeUsers_norm[userid][userid]+=1
    DegreeItems_norm[itemid][itemid]+=1
    
for i in range(len(DegreeUsers_norm)):
    for j in range(len(DegreeUsers_norm[i])):
        if DegreeUsers_norm[i][j] !=0:
            DegreeUsers_norm[i][j]=DegreeUsers_norm[i][j]**-0.5
            break    
            
for i in range(len(DegreeItems_norm)):
    for j in range(len(DegreeItems_norm[i])):
        if DegreeItems_norm[i][j] !=0:
            DegreeItems_norm[i][j]=DegreeItems_norm[i][j]**-0.5
            break

class ShuffleIterator(object):

    def __init__(self, inputs, batch_size=10):
        self.inputs = inputs
        self.batch_size = batch_size
        self.num_cols = len(self.inputs)
        self.len = len(self.inputs[0])
        self.inputs = np.transpose(np.vstack([np.array(self.inputs[i]) for i in range(self.num_cols)]))

    def __len__(self):
        return self.len

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        ids = np.random.randint(0, self.len, (self.batch_size,))
        out = self.inputs[ids, :]
        return [out[:, i] for i in range(self.num_cols)]


class OneEpochIterator(ShuffleIterator):
    def __init__(self, inputs, batch_size=10):
        super(OneEpochIterator, self).__init__(inputs, batch_size=batch_size)
        if batch_size > 0:
            self.idx_group = np.array_split(np.arange(self.len), np.ceil(self.len / batch_size))
        else:
            self.idx_group = [np.arange(self.len)]
        self.group_id = 0

    def next(self):
        if self.group_id >= len(self.idx_group):
            self.group_id = 0
            raise StopIteration
        out = self.inputs[self.idx_group[self.group_id], :]
        self.group_id += 1
        return [out[:, i] for i in range(self.num_cols)]

def inferenceDense(phase,user_batch, item_batch,idx_user,idx_item, user_num, item_num,UReg=0.05,IReg=0.1):

    user_batch = tf.nn.embedding_lookup(idx_user, user_batch, name="embedding_user")
    item_batch = tf.nn.embedding_lookup(idx_item, item_batch, name="embedding_item")

    ul1mf=tf.layers.dense(inputs=user_batch, units=MFSIZE,activation=tf.nn.relu, kernel_initializer=tf.random_normal_initializer(stddev=0.01))
    il1mf=tf.layers.dense(inputs=item_batch, units=MFSIZE,activation=tf.nn.relu, kernel_initializer=tf.random_normal_initializer(stddev=0.01))
    InferInputMF=tf.multiply(ul1mf, il1mf)


    infer=tf.reduce_sum(InferInputMF, 1, name="inference")

    regularizer = tf.add(UW*tf.nn.l2_loss(ul1mf), IW*tf.nn.l2_loss(il1mf), name="regularizer")

    return infer, regularizer

def optimization(infer, regularizer, rate_batch, learning_rate=0.0005, reg=0.1):

    global_step = tf.train.get_global_step()
    assert global_step is not None
    cost_l2 = tf.nn.l2_loss(tf.subtract(infer, rate_batch))
    cost = tf.add(cost_l2, regularizer)
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(cost, global_step=global_step)
    return cost, train_op

def clip(x):
    return np.clip(x, 1.0, 5.0) 


def GraphRec_image(train, test,ver, Dataset='100k'):
    AdjacencyUsers = np.zeros((USER_NUM,ITEM_NUM), dtype=np.float32) #np.asarray([[0 for x in range(ITEM_NUM)] for y in range(USER_NUM)],dtype=np.float16)
    DegreeUsers = np.zeros((USER_NUM,1), dtype=np.float32)# np.asarray([[0 for x in range(1)] for y in range(USER_NUM)],dtype=np.float16)
    
    AdjacencyItems = np.zeros((ITEM_NUM,USER_NUM), dtype=np.float32) #np.asarray([[0 for x in range(USER_NUM)] for y in range(ITEM_NUM)],dtype=np.float16)
    DegreeItems =  np.zeros((ITEM_NUM,1), dtype=np.float32) #np.asarray([[0 for x in range(1)] for y in range(ITEM_NUM)],dtype=np.float16)
    for index, row in train.iterrows():
      userid=int(row['user'])
      itemid=int(row['item'])
      AdjacencyUsers[userid][itemid]=row['rate']/5.0
      AdjacencyItems[itemid][userid]=row['rate']/5.0
      DegreeUsers[userid][0]+=1
      DegreeItems[itemid][0]+=1
    
    DUserMax=np.amax(DegreeUsers) 
    DItemMax=np.amax(DegreeItems)
    DegreeUsers=np.true_divide(DegreeUsers, DUserMax)
    DegreeItems=np.true_divide(DegreeItems, DItemMax)
    
    AdjacencyUsers=np.asarray(AdjacencyUsers,dtype=np.float32)
    AdjacencyItems=np.asarray(AdjacencyItems,dtype=np.float32)

    if ver=='ver1':
        UserFeatures= np.concatenate((np.identity(USER_NUM,dtype=np.bool_), AdjacencyUsers), axis=1) 
        ItemFeatures= np.concatenate((np.identity(ITEM_NUM,dtype=np.bool_), AdjacencyItems,img_ma), axis=1) 
    if ver=='ver2':
        UserFeatures= np.concatenate((np.identity(USER_NUM,dtype=np.bool_), AdjacencyUsers,DegreeUsers), axis=1) 
        ItemFeatures= np.concatenate((np.identity(ITEM_NUM,dtype=np.bool_), AdjacencyItems,DegreeItems,img_ma), axis=1) 
    if ver=='ver3':
        UserFeatures= np.concatenate((np.identity(USER_NUM,dtype=np.bool_), AdjacencyUsers,DegreeUsers,paUsers), axis=1) 
        ItemFeatures= np.concatenate((np.identity(ITEM_NUM,dtype=np.bool_), AdjacencyItems,DegreeItems,paItems_img_ma), axis=1) 
    if ver=='ver4':
        UserFeatures= np.concatenate((np.identity(USER_NUM,dtype=np.bool_), AdjacencyUsers,DegreeUsers_norm,paUsers), axis=1) 
        ItemFeatures= np.concatenate((np.identity(ITEM_NUM,dtype=np.bool_), AdjacencyItems,DegreeItems_norm,paItems_img_ma), axis=1) 
    if ver=='ver5':
        UserFeatures= np.concatenate((np.identity(USER_NUM,dtype=np.bool_), AdjacencyUsers,norm_la_user,paUsers), axis=1) 
        ItemFeatures= np.concatenate((np.identity(ITEM_NUM,dtype=np.bool_), AdjacencyItems,norm_la_movie,paItems,img_ma), axis=1) 
    if ver=='ver6':
        UserFeatures= np.concatenate((np.identity(USER_NUM,dtype=np.bool_), AdjacencyUsers,la_user,paUsers), axis=1) 
        ItemFeatures= np.concatenate((np.identity(ITEM_NUM,dtype=np.bool_), AdjacencyItems,la_movie,paItems,img_ma), axis=1) 

    UserFeaturesLength=UserFeatures.shape[1]
    ItemFeaturesLength=ItemFeatures.shape[1]

    samples_per_batch = len(train) // BATCH_SIZE
    iter_train = ShuffleIterator([train["user"],train["item"],train["rate"]],batch_size=BATCH_SIZE)
    iter_test = OneEpochIterator([test["user"],test["item"],test["rate"]],batch_size=10000)

    user_batch = tf.placeholder(tf.int32, shape=[None], name="id_user")
    item_batch = tf.placeholder(tf.int32, shape=[None], name="id_item")
    rate_batch = tf.placeholder(tf.float64, shape=[None])
    phase = tf.placeholder(tf.bool, name='phase')

    w_user = tf.constant(UserFeatures,name="userids", shape=[USER_NUM, UserFeatures.shape[1]],dtype=tf.float64)
    w_item = tf.constant(ItemFeatures,name="itemids", shape=[ITEM_NUM, ItemFeatures.shape[1]],dtype=tf.float64)


    infer, regularizer = inferenceDense(phase,user_batch, item_batch,w_user,w_item, user_num=USER_NUM, item_num=ITEM_NUM)
    global_step = tf.contrib.framework.get_or_create_global_step()
    _, train_op = optimization(infer, regularizer, rate_batch, learning_rate=LR, reg=0.09)

    init_op = tf.global_variables_initializer()
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.5
    finalerror=-1
    with tf.Session(config=config) as sess:
        sess.run(init_op)
        print("{} {} {} {}".format("epoch", "train_error", "val_error", "elapsed_time"))
        errors = deque(maxlen=samples_per_batch)
        start = time.time()
        for i in range(EPOCH_MAX * samples_per_batch):
            #users, items, rates,y,m,d,dw,dy,w = next(iter_train)
            users, items, rates = next(iter_train)
            _, pred_batch = sess.run([train_op, infer], feed_dict={user_batch: users,
                                                                   item_batch: items,
                                                                   rate_batch: rates,
                                                                   phase:True})
            pred_batch = clip(pred_batch)
            errors.append(np.power(pred_batch - rates, 2))
            if i % samples_per_batch == 0:
                train_err = np.sqrt(np.mean(errors))
                test_err2 = np.array([])
                degreelist=list()
                predlist=list()
                for users, items, rates in iter_test:
                    pred_batch = sess.run(infer, feed_dict={user_batch: users,
                                                            item_batch: items,                                                                                             
                                                            phase:False})

                    pred_batch = clip(pred_batch)            
                    test_err2 = np.append(test_err2, np.power(pred_batch - rates, 2))
                end = time.time()
                test_err = np.sqrt(np.mean(test_err2))
                finalerror=test_err
                print("{:3d},{:f},{:f},{:f}(s)".format(i // samples_per_batch, train_err, test_err, end - start))
                start = end

MFSIZE=50
UW=0.05
IW=0.02
LR=0.00003
EPOCH_MAX = 300
tf.reset_default_graph()
GraphRec_image(df_train, df_test,ver='ver1',Dataset='100k')

