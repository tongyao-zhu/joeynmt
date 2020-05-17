#!/usr/bin/env python
# coding: utf-8

# In[4]:


import torch
import os


# In[2]:


# Part 1: check whether individiually generated features are the same as combined

tensor_list = []

for i in range(0,95):
    print(f"Reading Current tensor {i}")
    current_tensor = torch.load("./../feature_tensors/feature_tensor_{}".format(i))
    tensor_list.append(current_tensor)

concat_tensor = torch.concat(tensor_list)
print("concatenated tensor has shape {}".format(concat_tensor.shape))

print("loading total features")

total_features = torch.load("./feature_tensor.pt")

print("total features has shape {}".format(total_features.shape))


# In[ ]:


print("total features equal to concatenated features {}".format((total_features == concatenated_features).all()))


# In[ ]:




