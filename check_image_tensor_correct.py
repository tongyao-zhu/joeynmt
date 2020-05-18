#!/usr/bin/env python
# coding: utf-8

# In[5]:


import json
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch import nn
import os
import random


# In[2]:


random_size = 50


# In[3]:


def get_image_tensor(images_list):
    print("input images list has length {}".format(len(images_list)))
    input_image_tensors = []
    for input_image in images_list:
        input_image = Image.open(input_image)
        # Preprocess image
        preprocess = transforms.Compose([
          transforms.Resize(256),
          transforms.CenterCrop(224),
          transforms.ToTensor(),
          transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        input_tensor = preprocess(input_image).unsqueeze(0)
        input_image_tensors.append(input_tensor)
        # create a minibatch
    input_batch = torch.cat(input_image_tensors)
    # print(input_batch.shape)
    return input_batch


# In[6]:


with open("./sign_text_data/images_name_list.txt", "r") as f:
    image_names = f.readlines()

image_names = list(map(lambda x:x.strip("\n").strip(), image_names))



random_names = [image_names[0],]
random_indices = [0]
while len(random_names) < random_size:
    random_index = random.randint(0, len(image_names)-1)
    if random_index not in random_indices:
        random_indices.append(random_index)
        random_names.append(image_names[random_index])
print("random names {}".format(random_names[:20]))

random_names = list(map(lambda x:"./." + x, random_names))


# In[7]:


random_indices[:10]


# In[9]:


image_tensors = get_image_tensor(random_names)


# In[ ]:


for index in range(len(random_indices)):
    print(f"currently  checking index {random_indices[index]}")
    real_tensor = image_tensors[index]
    acc_index = random_indices[index]
    larger_index = acc_index//10000
    print(f"current larger index{larger_index}")
    features_tensor = torch.load(f"./../image_tensors/image_tensor_{larger_index}.pt")
    offset = acc_index%10000
    assert (features_tensor[offset]==real_tensor).all()

