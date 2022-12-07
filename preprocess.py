#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def encoding(df,user,item,rating,time):
    """Function for labelencoder to user_col and item_col
    
    Params:     
        df (pd.DataFrame): Pandas data frame to be used.
        user (string): Name of the user column.
        item (string): Name of the item column.
        rating (string): Name of the rating column.
        time (string): Name of the timestamp column.
    
    Returns: 
        df_encoded (pd.DataFrame): Modifed dataframe with the users and items index
    """
    
    df_copy = df.copy()
    
    user_encoder = LabelEncoder()
    user_encoder.fit(df_copy[user].values)
    df_copy["USER"] = user_encoder.transform(df_copy[user])
    
    item_encoder = LabelEncoder()
    item_encoder.fit(df_copy[item].values)
    df_copy["ITEM"] = item_encoder.transform(df_copy[item])

    df_copy.rename({rating: "RATING", time: "TIMESTAMP"}, axis=1, inplace=True)

    return df_copy

