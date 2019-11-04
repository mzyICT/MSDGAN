#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[3]:


get_ipython().system("ls '../data/'")


# In[4]:


df = pd.read_csv('../data/train_manifest.csv', header=None)


# In[5]:


train_df = df[:5000]
val_df = df[5000:6000]
len(train_df), len(val_df)


# In[6]:


train_df.to_csv('train_manifest.csv', header=None, index=None)
val_df.to_csv('val_manifest.csv', header=None, index=None)


# In[ ]:




