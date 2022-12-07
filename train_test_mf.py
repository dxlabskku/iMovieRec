#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys
sys.path.append("../")
import numpy as np
import pandas as pd
from IPython.display import SVG
import matplotlib.pyplot as plt
import seaborn as sns
from preprocess import encoding
from sklearn.model_selection import train_test_split
import math
from IPython.display import SVG, display
get_ipython().magic(u'matplotlib inline')
import keras
from keras.models import Model
from keras.layers import Input, Embedding, Flatten, Dot, Add, Lambda, Activation, Reshape
from keras.regularizers import l2
from keras.constraints import non_neg
from keras.utils import plot_model
from keras.utils.vis_utils import model_to_dot
import vis
from sklearn.metrics import classification_report,mean_squared_error
from keras.optimizers import Adam
import resource
from PIL import Image
from models_mf import single_layer_mf_image,single_layer_mf_image_withbias,SVD_image_withbias,two_layer_mf_image,two_layer_mf_image_withbias,NMF_image,NMF_image_withbias,

def read_100k():
    users = pd.read_csv("./data/users.csv")
    items = pd.read_csv("./data/items.csv")
    ratings = pd.read_csv("data/ratings.csv")
    encoded_ratings= encoding(ratings, "user_id", "movie_id",'rating','unix_timestamp')
    n_users = encoded_ratings.USER.nunique()
    n_items = encoded_ratings.ITEM.nunique()
    ratings['rating'] = ratings['rating'].values.astype(np.float32)
    min_rating = min(ratings['rating'])
    max_rating = max(ratings['rating'])
    train, test = train_test_split(encoded_ratings, test_size=0.1, random_state=123)
    return train, test

def read_1m():    
    ratings = pd.read_csv("./data/rating_1m.csv")
    ratings.columns=["user", "item",'rate','timestamp']
    encoded_ratings= encoding(ratings, "user_id", "movie_id",'rating','unix_timestamp')
    n_users = encoded_ratings.USER.nunique()
    n_items = encoded_ratings.ITEM.nunique()
    ratings['rating'] = ratings['rating'].values.astype(np.float32)
    min_rating = min(ratings['rating'])
    max_rating = max(ratings['rating'])
    train, test = train_test_split(encoded_ratings, test_size=0.1, random_state=123)
    return train, test

train,test=read_100k()

np_useridTr = train['USER'].values
np_itemidTr = train['ITEM'].values
np_rateidTr = train['RATING'].values

np_useridTe = test['USER'].values
np_itemidTe = test['ITEM'].values
np_rateidTe = test['RATING'].values      


#image  
img_dir='/data/test_img/'
moviedirs = os.listdir(img_dir)
moviedirs2moviename = {}
for dirs,name in zip(moviedirs,moviedirs):
    if dirs != '.ipynb_checkpo':
        moviedirs2moviename[dirs[:]] = name[:]
moviedirs2moviename
encoded_ratings=encoded_ratings.astype({'movie_id':str})
moviename2img = {}
for movie in moviedirs[:]:
    if movie != '.ipynb_checkpo':
        imgs = os.listdir(img_dir+movie+'/')
        img = Image.open(img_dir+movie+'/'+str(imgs[0]))
        moviename = moviedirs2moviename[movie[:]]
        moviename2img[encoded_ratings[encoded_ratings['movie_id']== moviename].ITEM.unique()[0]] = img
ResNetInputSize = (224,224)

moviename2resizedImg = {}
for key,value in moviename2img.items():
    if value != None:
        moviename2resizedImg[key] = np.array(value.resize(ResNetInputSize),dtype=np.float)/255.0
    else:
        moviename2resizedImg[key] = None

def printOriginalImage(arr):
    img = Image.fromarray((arr*255).astype(np.uint8))
    img.show()
    return img

class DataLoader(tf.keras.utils.Sequence):
    def __init__(self,userid,movieid,imageDict, y_label, batch_size, shuffle = True,outputShape = None):
        self.userid = userid
        self.movieid = movieid
        self.imageDict = imageDict
        self.y_label = y_label
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.outputShape = outputShape
        self.on_epoch_end()
        
    def on_epoch_end(self):
        self.indexes = np.arange(self.movieid.shape[0]) #
        if self.shuffle:
            np.random.shuffle(self.indexes)
            
    def __len__(self):
        return int(np.floor(self.movieid.shape[0] / self.batch_size))
        
    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]
        userid = self.userid[indexes]
        movieid = self.movieid[indexes]
        y = self.y_label[indexes]
        images = np.zeros(shape=(len(movieid),224,224,3),dtype=np.float)
        for i in range(len(movieid)):
            images[i] = self.imageDict[(movieid[i])] 
        return [userid, movieid, images],y

n_factors = 10
batch_size =32
epochs = 20
model = single_layer_mf_image(n_users, n_items, n_factors)
output = model.fit_generator(generator=DataLoader(np_useridTr, np_itemidTr,moviename2resizedImg,np_rateidTr,batch_size=batch_size), epochs=epochs, verbose=1)
predict = model.predict([test.USER, test.ITEM])
predict = np.array(predict)
RMSE = mean_squared_error(test.RATING, predict)**0.5

