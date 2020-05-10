#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch


# In[37]:


features_tensor = torch.load("../features_tensor.pt")


# In[38]:


print(f"input features_tensor shape {features_tensor.shape}")


# In[39]:


transposed_features_tensor = torch.transpose(features_tensor, 0,1)
transposed_features_tensor = torch.cat([transposed_features_tensor, torch.zeros([3, transposed_features_tensor.shape[1]])])
features_tensor =  torch.transpose(transposed_features_tensor, 0,1)
print(f"features tensor now has shape {features_tensor.shape}")


# In[41]:


special_tokens_tensor = torch.zeros([4,features_tensor.shape[1]])


# In[42]:


special_tokens_tensor[0,-1]=1
special_tokens_tensor[2,-2]=1
special_tokens_tensor[3,-3]=1


# In[43]:


features_tensor = torch.cat([special_tokens_tensor, features_tensor])


# In[46]:


torch.save(features_tensor, "./complete_features_tensor.pt")

