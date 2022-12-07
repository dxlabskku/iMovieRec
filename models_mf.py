#!/usr/bin/env python
# coding: utf-8

# In[2]:


import sys
sys.path.append("../")
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import math
from IPython.display import SVG, display
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic(u'matplotlib inline')
from keras.models import Model,load_model, Sequential
from keras.layers import Input, Embedding, Flatten, Dot, Add, Lambda, Activation, Reshape,Conv2D,MaxPooling2D,Dropout,Concatenate,Dense,BatchNormalization
from keras.regularizers import l2
from keras.constraints import non_neg
from keras.utils import plot_model
from keras.utils.vis_utils import model_to_dot
import vis
import tensorflow as tf
import time
import datetime
from keras.optimizers import Adam

def single_layer_mf_image(n_users, n_items, n_factors):
    item_input = Input(shape=[1])
    item_embedding = Embedding(n_items, n_factors, 
                               embeddings_regularizer=l2(1e-6))(item_input)
    item_vec = Flatten()(item_embedding)
    
    image_input = Input(shape=(224,224,3))
    imgflow = tf.keras.layers.Conv2D(32,(3,3),padding='same',activation = 'relu')(image_input)
    imgflow = tf.keras.layers.Conv2D(32,(3,3),padding='same',activation = 'relu')(imgflow)
    imgflow = MaxPooling2D(pool_size=(4,4))(imgflow)
    imgflow = Dropout(0.25)(imgflow)

    imgflow = tf.keras.layers.Conv2D(32,(3,3),padding='same',activation = 'relu')(imgflow)
    imgflow = MaxPooling2D(pool_size=(4,4))(imgflow)
    imgflow = Dropout(0.25)(imgflow)

    imgflow = tf.keras.layers.Conv2D(32,(3,3),padding='same',activation = 'relu')(imgflow)
    imgflow = Dropout(0.25)(imgflow)

    imgflow = tf.keras.layers.Conv2D(32,(3,3),padding='same',activation = 'relu')(imgflow)
    imgflow = Dropout(0.25)(imgflow)
    imgflow = Flatten()(imgflow)
    imgflow = Dense(512,activation='relu')(imgflow)
    imgflow = BatchNormalization()(imgflow)
    imgflow = Dense(256,activation='relu')(imgflow)
    imgflow = BatchNormalization()(imgflow)
    imgflow = Dense(128,activation='relu')(imgflow)
    
    Concat = tf.keras.layers.concatenate(inputs=[item_vec,imgflow],axis=1)    
    Dense_1 = Dense(n_factors, kernel_initializer='glorot_normal')(Concat)
    user_input = Input(shape=[1])
    user_embedding = Embedding(n_users, n_factors, 
                               embeddings_regularizer=l2(1e-6))(user_input)
    user_vec = Flatten()(user_embedding)

    rating = Dot(axes=1)([Dense_1, user_vec])
    model = Model([user_input, item_input, image_input], rating)
    model.compile(loss='mean_squared_error', optimizer="adam")
    return model

def single_layer_mf_image_withbias(n_users, n_items, n_factors):
    item_input = Input(shape=[1])
    item_embedding = Embedding(n_items, n_factors, embeddings_regularizer=l2(1e-5))(item_input)
    item_vec = Flatten()(item_embedding)
    
    image_input = Input(shape=(224,224,3))
    imgflow = tf.keras.layers.Conv2D(32,(3,3),padding='same',activation = 'relu')(image_input)
    imgflow = tf.keras.layers.Conv2D(32,(3,3),padding='same',activation = 'relu')(imgflow)
    imgflow = MaxPooling2D(pool_size=(4,4))(imgflow)
    imgflow = Dropout(0.25)(imgflow)

    imgflow = tf.keras.layers.Conv2D(32,(3,3),padding='same',activation = 'relu')(imgflow)
    imgflow = MaxPooling2D(pool_size=(4,4))(imgflow)
    imgflow = Dropout(0.25)(imgflow)

    imgflow = tf.keras.layers.Conv2D(32,(3,3),padding='same',activation = 'relu')(imgflow)
    imgflow = Dropout(0.25)(imgflow)

    imgflow = tf.keras.layers.Conv2D(32,(3,3),padding='same',activation = 'relu')(imgflow)
    imgflow = Dropout(0.25)(imgflow)
    imgflow = Flatten()(imgflow)
    imgflow = Dense(512,activation='relu')(imgflow)
    imgflow = BatchNormalization()(imgflow)
    imgflow = Dense(256,activation='relu')(imgflow)
    imgflow = BatchNormalization()(imgflow)
    imgflow = Dense(128,activation='relu')(imgflow)
    
    Concat = tf.keras.layers.concatenate(inputs=[item_vec,imgflow],axis=1)
    Dense_1 = Dense(n_factors, kernel_initializer='glorot_normal')(Concat)
    
    item_bias = Embedding(n_items, 1, embeddings_regularizer=l2(1e-6))(item_input)
    item_bias_vec = Flatten()(item_bias)

    user_input = Input(shape=[1])
    user_embedding = Embedding(n_users, n_factors, embeddings_regularizer=l2(1e-6))(user_input)
    user_vec = Flatten()(user_embedding)
    user_bias = Embedding(n_users, 1, embeddings_regularizer=l2(1e-6))(user_input)
    user_bias_vec = Flatten()(user_bias)

    DotProduct = Dot(axes=1)([Dense_1, user_vec])
    AddBias = Add()([DotProduct, item_bias_vec, user_bias_vec])
    
    y = Activation('sigmoid')(AddBias)
    rating_output = Lambda(lambda x: x * (max_rating - min_rating) + min_rating)(y)

    model = Model([user_input, item_input,image_input], rating_output)
    
    model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.001))
    
    return model

def SVD_image_withbias(n_users, n_items, n_factors):
    item_input = Input(shape=[1])
    item_embedding = Embedding(n_items, n_factors, embeddings_regularizer=l2(1e-5))(item_input)
    item_vec = Flatten()(item_embedding)
    
    image_input = Input(shape=(224,224,3))
    imgflow = tf.keras.layers.Conv2D(32,(3,3),padding='same',activation = 'relu')(image_input)
    imgflow = tf.keras.layers.Conv2D(32,(3,3),padding='same',activation = 'relu')(imgflow)
    imgflow = MaxPooling2D(pool_size=(4,4))(imgflow)
    imgflow = Dropout(0.25)(imgflow)

    imgflow = tf.keras.layers.Conv2D(32,(3,3),padding='same',activation = 'relu')(imgflow)
    imgflow = MaxPooling2D(pool_size=(4,4))(imgflow)
    imgflow = Dropout(0.25)(imgflow)

    imgflow = tf.keras.layers.Conv2D(32,(3,3),padding='same',activation = 'relu')(imgflow)
    imgflow = Dropout(0.25)(imgflow)

    imgflow = tf.keras.layers.Conv2D(32,(3,3),padding='same',activation = 'relu')(imgflow)
    imgflow = Dropout(0.25)(imgflow)
    imgflow = Flatten()(imgflow)
    imgflow = Dense(512,activation='relu')(imgflow)
    imgflow = BatchNormalization()(imgflow)
    imgflow = Dense(256,activation='relu')(imgflow)
    imgflow = BatchNormalization()(imgflow)
    imgflow = Dense(128,activation='relu')(imgflow)

    Concat = tf.keras.layers.concatenate(inputs=[item_vec,imgflow],axis=1)
    Dense_1 = Dense(n_factors, kernel_initializer='glorot_normal')(Concat)

    item_bias = Embedding(n_items, 1, embeddings_regularizer=l2(1e-5))(item_input)
    item_bias_vec = Flatten()(item_bias)
    
    user_input = Input(shape=[1])
    user_embedding = Embedding(n_users, n_factors, embeddings_regularizer=l2(1e-5))(user_input)
    user_vec = Flatten()(user_embedding)

    user_bias = Embedding(n_users, 1, embeddings_regularizer=l2(1e-5))(user_input)
    user_bias_vec = Flatten()(user_bias)

    DotProduct = Dot(axes=1)([Dense_2, user_vec])
    AddBias = Add()([DotProduct, item_bias_vec, user_bias_vec])

    y = Activation('sigmoid')(AddBias)
    rating_output = Lambda(lambda x: x * (max_rating - min_rating) + min_rating)(y)
    
    model = Model([user_input, item_input,image_input], rating_output)
    
    model.compile(loss='mean_squared_error', optimizer="adam")
    
    return model

def two_layer_mf_image(n_users, n_items, n_factors):
    item_input = Input(shape=[1])
    item_embedding = Embedding(n_items, n_factors, embeddings_regularizer=l2(1e-6),
                               embeddings_initializer='glorot_normal')(item_input)
    item_vec = Flatten()(item_embedding)
    
    image_input = Input(shape=(224,224,3))
    imgflow = tf.keras.layers.Conv2D(32,(3,3),padding='same',activation = 'relu')(image_input)
    imgflow = tf.keras.layers.Conv2D(32,(3,3),padding='same',activation = 'relu')(imgflow)
    imgflow = MaxPooling2D(pool_size=(4,4))(imgflow)
    imgflow = Dropout(0.25)(imgflow)

    imgflow = tf.keras.layers.Conv2D(32,(3,3),padding='same',activation = 'relu')(imgflow)
    imgflow = MaxPooling2D(pool_size=(4,4))(imgflow)
    imgflow = Dropout(0.25)(imgflow)

    imgflow = tf.keras.layers.Conv2D(32,(3,3),padding='same',activation = 'relu')(imgflow)
    imgflow = Dropout(0.25)(imgflow)

    imgflow = tf.keras.layers.Conv2D(32,(3,3),padding='same',activation = 'relu')(imgflow)
    imgflow = Dropout(0.25)(imgflow)
    imgflow = Flatten()(imgflow)
    imgflow = Dense(512,activation='relu')(imgflow)
    imgflow = BatchNormalization()(imgflow)
    imgflow = Dense(256,activation='relu')(imgflow)
    imgflow = BatchNormalization()(imgflow)
    imgflow = Dense(128,activation='relu')(imgflow)
    
    Concat_1 = tf.keras.layers.concatenate(inputs=[item_vec,imgflow],axis=1)
    Concat_1 = Dense(n_factors, kernel_initializer='glorot_normal')(Concat_1)

    user_input = Input(shape=[1])
    user_embedding = Embedding(n_users, n_factors, embeddings_regularizer=l2(1e-6),
                               embeddings_initializer='glorot_normal')(user_input)
    user_vec = Flatten()(user_embedding)
    
    Concat = tf.keras.layers.concatenate([Concat_1, user_vec],axis=1)
    ConcatDrop = Dropout(0.5)(Concat)

    kernel_initializer='he_normal'
    Dense_1 = Dense(10, kernel_initializer='glorot_normal')(ConcatDrop)
    Dense_1_Drop = Dropout(0.5)(Dense_1)
    Dense_2 = Dense(1, kernel_initializer='glorot_normal')(Dense_1_Drop)

    y = Activation('sigmoid')(Dense_2)
    rating_output = Lambda(lambda x: x * (max_rating - min_rating) + min_rating)(y)
    
    model = Model([user_input, item_input,image_input], rating_output)
    
    model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.001))
    
    return model

def two_layer_mf_image_withbias(n_users, n_items, n_factors):
    item_input = Input(shape=[1])
    item_embedding = Embedding(n_items, n_factors, embeddings_regularizer=l2(1e-6),
                               embeddings_initializer='glorot_normal',
                               name='ItemEmbedding')(item_input)
    item_vec = Flatten(name='FlattenItemE')(item_embedding)
    
    
    image_input = Input(shape=(224,224,3),name='img')
    imgflow = tf.keras.layers.Conv2D(32,(3,3),padding='same',activation = 'relu')(image_input)
    imgflow = tf.keras.layers.Conv2D(32,(3,3),padding='same',activation = 'relu')(imgflow)
    imgflow = MaxPooling2D(pool_size=(4,4))(imgflow)
    imgflow = Dropout(0.25)(imgflow)

    imgflow = tf.keras.layers.Conv2D(32,(3,3),padding='same',activation = 'relu')(imgflow)
    imgflow = MaxPooling2D(pool_size=(4,4))(imgflow)
    imgflow = Dropout(0.25)(imgflow)

    imgflow = tf.keras.layers.Conv2D(32,(3,3),padding='same',activation = 'relu')(imgflow)
    imgflow = Dropout(0.25)(imgflow)

    imgflow = tf.keras.layers.Conv2D(32,(3,3),padding='same',activation = 'relu')(imgflow)
    imgflow = Dropout(0.25)(imgflow)
    imgflow = Flatten()(imgflow)
    imgflow = Dense(512,activation='relu')(imgflow)
    imgflow = BatchNormalization()(imgflow)
    imgflow = Dense(256,activation='relu')(imgflow)
    imgflow = BatchNormalization()(imgflow)
    imgflow = Dense(128,activation='relu')(imgflow)
    
    Concat_1 = tf.keras.layers.concatenate(inputs=[item_vec,imgflow],axis=1)

    Concat_1 = Dense(n_factors, kernel_initializer='glorot_normal')(Concat_1)
    
    item_bias = Embedding(n_items, 1, embeddings_regularizer=l2(1e-6), 
                          embeddings_initializer='glorot_normal')(item_input)
    item_bias_vec = Flatten()(item_bias)

    user_input = Input(shape=[1])
    user_embedding = Embedding(n_users, n_factors, embeddings_regularizer=l2(1e-6),
                               embeddings_initializer='glorot_normal')(user_input)
    user_vec = Flatten()(user_embedding)
    
    user_bias = Embedding(n_users, 1, embeddings_regularizer=l2(1e-6), 
                        embeddings_initializer='glorot_normal')(user_input)
    user_bias_vec = Flatten()(user_bias)

    Concat = tf.keras.layers.concatenate([Concat_1, user_vec],axis=1)
    ConcatDrop = Dropout(0.5)(Concat)

    kernel_initializer='he_normal'
    Dense_1 = Dense(10, kernel_initializer='glorot_normal')(ConcatDrop)
    Dense_1_Drop = Dropout(0.5)(Dense_1)
    Dense_2 = Dense(1, kernel_initializer='glorot_normal')(Dense_1_Drop)

    AddBias = Add()([Dense_2, item_bias_vec, user_bias_vec])
    
    y = Activation('sigmoid')(AddBias)
    rating_output = Lambda(lambda x: x * (max_rating - min_rating) + min_rating)(y)
    
    model = Model([user_input, item_input,image_input], rating_output)

    model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.001))
    
    return model

def NMF_image(n_users, n_items, n_factors):
    item_input = Input(shape=[1])
    item_embedding = Embedding(n_items, n_factors, embeddings_regularizer=l2(1e-5),
                               embeddings_constraint= non_neg())(item_input)
    item_vec = Flatten()(item_embedding)

    image_input = Input(shape=(224,224,3))
    imgflow = tf.keras.layers.Conv2D(32,(3,3),padding='same',activation = 'relu')(image_input)
    imgflow = tf.keras.layers.Conv2D(32,(3,3),padding='same',activation = 'relu')(imgflow)
    imgflow = MaxPooling2D(pool_size=(4,4))(imgflow)
    imgflow = Dropout(0.25)(imgflow)

    imgflow = tf.keras.layers.Conv2D(32,(3,3),padding='same',activation = 'relu')(imgflow)
    imgflow = MaxPooling2D(pool_size=(4,4))(imgflow)
    imgflow = Dropout(0.25)(imgflow)

    imgflow = tf.keras.layers.Conv2D(32,(3,3),padding='same',activation = 'relu')(imgflow)
    imgflow = Dropout(0.25)(imgflow)

    imgflow = tf.keras.layers.Conv2D(32,(3,3),padding='same',activation = 'relu')(imgflow)
    imgflow = Dropout(0.25)(imgflow)
    imgflow = Flatten()(imgflow)
    imgflow = Dense(512,activation='relu')(imgflow)
    imgflow = BatchNormalization()(imgflow)
    imgflow = Dense(256,activation='relu')(imgflow)
    imgflow = BatchNormalization()(imgflow)
    imgflow = Dense(128,activation='relu')(imgflow)
    
    Concat_1 = tf.keras.layers.concatenate(inputs=[item_vec,imgflow],axis=1)
    Concat_1 = Dense(n_factors, kernel_initializer='glorot_normal')(Concat_1)

    user_input = Input(shape=[1])
    user_embedding = Embedding(n_users, n_factors, embeddings_regularizer=l2(1e-5), 
                               embeddings_constraint= non_neg())(user_input)
    user_vec = Flatten()(user_embedding)

    DotProduct = Dot(axes=1)([Concat_1, user_vec])

    model = Model([user_input, item_input,image_input], DotProduct)

    model.compile(loss='mean_squared_error', optimizer="adam")
    
    return model

def NMF_image_withbias(n_users, n_items, n_factors):
    item_input = Input(shape=[1])
    item_embedding = Embedding(n_items, n_factors, embeddings_regularizer=l2(1e-5),
                               embeddings_constraint= non_neg())(item_input)
    item_vec = Flatten()(item_embedding)

    image_input = Input(shape=(224,224,3))
    imgflow = tf.keras.layers.Conv2D(32,(3,3),padding='same',activation = 'relu')(image_input)
    imgflow = tf.keras.layers.Conv2D(32,(3,3),padding='same',activation = 'relu')(imgflow)
    imgflow = MaxPooling2D(pool_size=(4,4))(imgflow)
    imgflow = Dropout(0.25)(imgflow)

    imgflow = tf.keras.layers.Conv2D(32,(3,3),padding='same',activation = 'relu')(imgflow)
    imgflow = MaxPooling2D(pool_size=(4,4))(imgflow)
    imgflow = Dropout(0.25)(imgflow)

    imgflow = tf.keras.layers.Conv2D(32,(3,3),padding='same',activation = 'relu')(imgflow)
    imgflow = Dropout(0.25)(imgflow)

    imgflow = tf.keras.layers.Conv2D(32,(3,3),padding='same',activation = 'relu')(imgflow)
    imgflow = Dropout(0.25)(imgflow)
    imgflow = Flatten()(imgflow)
    imgflow = Dense(512,activation='relu')(imgflow)
    imgflow = BatchNormalization()(imgflow)
    imgflow = Dense(256,activation='relu')(imgflow)
    imgflow = BatchNormalization()(imgflow)
    imgflow = Dense(128,activation='relu')(imgflow)
    
    Concat_1 = tf.keras.layers.concatenate(inputs=[item_vec,imgflow],axis=1)
    Concat_1 = Dense(n_factors, kernel_initializer='glorot_normal')(Concat_1)
    
    item_bias = Embedding(n_items, 1, embeddings_regularizer=l2(1e-5))(item_input)
    item_bias_vec = Flatten()(item_bias)

    user_input = Input(shape=[1])
    user_embedding = Embedding(n_users, n_factors, embeddings_regularizer=l2(1e-5), 
                               embeddings_constraint= non_neg())(user_input)
    user_vec = Flatten()(user_embedding)
    
    user_bias = Embedding(n_users, 1, embeddings_regularizer=l2(1e-5))(user_input)
    user_bias_vec = Flatten()(user_bias)

    DotProduct = Dot(axes=1)([Concat_1, user_vec])
    AddBias = Add()([DotProduct, item_bias_vec, user_bias_vec])
    
    y = Activation('sigmoid')(AddBias)
    rating_output = Lambda(lambda x: x * (max_rating - min_rating) + min_rating)(y)
    
    model = Model([user_input, item_input,image_input], rating_output)
    
    model.compile(loss='mean_squared_error', optimizer="adam")
    
    return model

