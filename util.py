#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from util import get_data100k,read_process,get_edgelist,pa_index
import networkx as nx
from networkx.algorithms import bipartite
from preprocess import encoder
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

def read_process(filname, sep="\t"):
    col_names = ["user", "item", "rate", "st"]
    df = pd.read_csv(filname, sep=sep, header=None, names=col_names, engine='python')
    df["user"] -= 1
    df["item"] -= 1
    for col in ("user", "item"):
        df[col] = df[col].astype(np.int32)
    df["rate"] = df["rate"].astype(np.float32)
    return df

def get_data100k():
    global PERC
    df = read_process("./data/ml-100k/u.data", sep="\t")
    rows = len(df)
    df_train, df_test = train_test_split(df, test_size=0.1, random_state=123)
    return df_train,df_test

def get_data1m():
    ratings = pd.read_csv("./data/rating_1m.csv")
    del ratings['Unnamed: 0']
    ratings.columns=["user", "item",'rate','timestamp']
    encoded_ratings = encoder(ratings, "user", "item",'rate','timestamp')
    USER_NUM = encoded_ratings.USER.nunique()
    ITEM_NUM = encoded_ratings.ITEM.nunique()
    ratings['rate'] = ratings['rate'].values.astype(np.float32)
    min_rating = min(ratings['rate'])
    max_rating = max(ratings['rate'])
    df_train, df_test = train_test_split(encoded_ratings, test_size=0.5, random_state=123)
    return df_train,df_test

def get_edgelist():
    edge_list = []    
    for i in range(len(df_train)):
        user, movie= df_train.user[i], df_train.item[i]
        edge_list.append((user, movie))
    return edge_list

def pa_index(df_train,df_test)
    df_train=df_train.set_index(np.arange(len(df_train)))
    df_test=df_test.set_index(np.arange(len(df_test)))

    df_train = df_train.astype({'item': 'int'})
    df_train = df_train.astype({'user': 'str'})
    a=list(set(df_train.user.tolist()))
    b=list(set(df_train.item.tolist()))

    B = nx.Graph()
    # Add nodes with the node attribute "bipartite"
    B.add_nodes_from(a, bipartite=1)
    B.add_nodes_from(b, bipartite=0)
    # Add edges only between nodes of opposite node sets
    B.add_edges_from(get_edgelist())

    bottom_nodes, top_nodes = bipartite.sets(B)

    G1 = bipartite.projected_graph(B, top_nodes)
    G2 = bipartite.projected_graph(B, bottom_nodes)

    pa1 = nx.preferential_attachment(G1)
    pa2 = nx.preferential_attachment(G2)

    U = []
    V = []
    C=[]
    for u,v,c in pa1:
        U.append(u)
        V.append(v)
        C.append(c)
    df = pd.DataFrame({'movie_id1':U, 'movie_id2':V,'pa1':C})

    U = []
    V = []
    C=[]
    for u,v,c in pa2:
        U.append(u)
        V.append(v)
        C.append(c)
    df1 = pd.DataFrame({'user_id1':U, 'user_id2':V,'pa2':C})

    paUsers = np.zeros((USER_NUM,USER_NUM), dtype=np.float32) #np.asarray([[0 for x in range(ITEM_NUM)] for y in range(USER_NUM)],dtype=np.float16)
    paItems = np.zeros((ITEM_NUM,ITEM_NUM), dtype=np.float32) #np.asarray([[0 for x in range(USER_NUM)] for y in range(ITEM_NUM)],dtype=np.float16)

    for index, row in df1.iterrows():
        userid1=int(row['user_id1'])
        userid2=int(row['user_id2'])
        paUsers[userid1][userid2]=row['pa2']

    for index, row in df.iterrows():
        itemid1=int(row['movie_id1'])
        itemid2=int(row['movie_id2'])
        paItems[itemid1][itemid2]=row['pa1']

    Max1=np.amax(paUsers) 
    Max2=np.amax(paItems)
    paUsers=np.true_divide(paUsers, Max1)
    paItems=np.true_divide(paItems, Max2)

    paUsers=np.asarray(paUsers,dtype=np.float32)
    paItems=np.asarray(paItems,dtype=np.float32)

    return paUsers,paItems


def laplacian_graph(df_train,df_test):
    test=df_test.copy()
    test['rate']=0
    la_train=pd.concat([df_train,test])
    la_train=la_train.set_index(np.arange(len(la_train)))

    la_train = la_train.astype({'item': 'int'})
    la_train = la_train.astype({'user': 'str'})

    a=list(set(la_train.user.tolist()))
    b=list(set(la_train.item.tolist()))

    B = nx.Graph()
    # Add nodes with the node attribute "bipartite"
    B.add_nodes_from(a, bipartite=1)
    B.add_nodes_from(b, bipartite=0)
    # Add edges only between nodes of opposite node sets

    B.add_weighted_edges_from(get_edgelist())

    bottom_nodes, top_nodes = bipartite.sets(B)

    G1 = bipartite.weighted_projected_graph(B, top_nodes, ratio=False)  #movie
    G2 = bipartite.weighted_projected_graph(B, bottom_nodes, ratio=False) #user

    la_movie=nx.normalized_laplacian_matrix(G1, nodelist=None, weight='weight')
    la_user=nx.normalized_laplacian_matrix(G2, nodelist=None, weight='weight')
    from scipy.sparse import csr_matrix
    la_movie=csr_matrix.todense(la_movie)
    la_user=csr_matrix.todense(la_user)
    m=nx.to_numpy_array(G1)
    u=nx.to_numpy_array(G2)

    Max1=np.amax(m) 
    Max2=np.amax(u)
    norm_la_movie=np.true_divide(m, Max1)
    norm_la_user=np.true_divide(u, Max2)
    return la_movie,la_user,norm_la_movie,norm_la_user

