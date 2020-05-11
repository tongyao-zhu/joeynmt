#!/usr/bin/env python
# coding: utf-8

# In[10]:


import torch


# In[11]:


features_tensor = torch.load("../f_features_tensor.pt")


# In[12]:


print(f"input features_tensor shape {features_tensor.shape}")


# In[13]:


features_tensor.max()


# In[14]:


features_tensor.min()


# In[ ]:


# transposed_features_tensor = torch.transpose(features_tensor, 0,1)
# transposed_features_tensor = torch.cat([transposed_features_tensor, torch.zeros([3, transposed_features_tensor.shape[1]])])
# features_tensor =  torch.transpose(transposed_features_tensor, 0,1)
# print(f"features tensor now has shape {features_tensor.shape}")

# special_tokens_tensor = torch.zeros([4,features_tensor.shape[1]])

# special_tokens_tensor[0,-1]=1
# special_tokens_tensor[2,-2]=1
# special_tokens_tensor[3,-3]=1

# features_tensor = torch.cat([special_tokens_tensor, features_tensor])


# In[16]:


special_tokens_tensor = torch.zeros([4,features_tensor.shape[1]])

special_tokens_tensor[0,:]=-20
special_tokens_tensor[2,:]=-10
special_tokens_tensor[3,:]=10


# In[17]:


features_tensor = torch.cat([special_tokens_tensor, features_tensor])


# In[18]:


print(f"features tensor now has shape {features_tensor.shape}")


# In[ ]:


torch.save(features_tensor, "./complete_features_tensor_1024_dimension.pt")

