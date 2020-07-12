#!/usr/bin/env python
# coding: utf-8

# In[10]:


import torch
import argparse
# In[11]:

def add_tokens(tensor_path, save_path):


    features_tensor = torch.load("../f_features_tensor.pt")
    print(f"input features_tensor shape {features_tensor.shape}")
    print(f"the max value in feature_tensor is {features_tensor.max()}")
    print(f"the min value in feature tensor is {features_tensor.min()}")
    special_tokens_tensor = torch.zeros([4, features_tensor.shape[1]])

    special_tokens_tensor[0, :] = -20
    special_tokens_tensor[2, :] = -10
    special_tokens_tensor[3, :] = 10
    features_tensor = torch.cat([special_tokens_tensor, features_tensor])

    print(f"features tensor now has shape {features_tensor.shape}")
    torch.save(features_tensor, save_path)
    print("finished saving to {}".format(save_path))

# transposed_features_tensor = torch.transpose(features_tensor, 0,1)
# transposed_features_tensor = torch.cat([transposed_features_tensor, torch.zeros([3, transposed_features_tensor.shape[1]])])
# features_tensor =  torch.transpose(transposed_features_tensor, 0,1)
# print(f"features tensor now has shape {features_tensor.shape}")

# special_tokens_tensor = torch.zeros([4,features_tensor.shape[1]])

# special_tokens_tensor[0,-1]=1
# special_tokens_tensor[2,-2]=1
# special_tokens_tensor[3,-3]=1

# features_tensor = torch.cat([special_tokens_tensor, features_tensor])




if __name__ == "__main__":
    ap = argparse.ArgumentParser()

    ap.add_argument("--tensor_path", type = str, required=True)
    ap.add_argument("--save_path", type = str, required=True)
    args = ap.parse_args()
    print(f"you've entered the following arguments: {args}")
    add_tokens(args.tensor_path, args.save_path)
