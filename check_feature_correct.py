#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import json


import torch
import torchvision.transforms as transforms
from PIL import Image
from googlenet_pytorch import GoogLeNet 
from torch import nn
import os


# In[ ]:


import torch
import os
import random

# In[2]:


# Part 1: check whether individiually generated features are the same as combined

tensor_list = []

for i in range(0,95):
    print(f"Reading Current tensor {i}")
    current_tensor = torch.load("./../feature_tensors/feature_tensor_{}".format(i))
    tensor_list.append(current_tensor)

concat_tensor = torch.cat(tensor_list)
print("concatenated tensor has shape {}".format(concat_tensor.shape))

print("loading total features")

total_features = torch.load("./features_tensor.pt")

print("total features has shape {}".format(total_features.shape))


# In[ ]:


print("total features equal to concatenated features {}".format((total_features == concat_tensor).all()))


# In[ ]:


# Check correctness of generated images


# In[ ]:


random_size = 100


# In[ ]:


os.getcwd()


# In[ ]:


with open("./sign_text_data/images_name_list.txt", "r") as f:
    image_names = f.readlines()


# In[ ]:


image_names = list(map(lambda x:x.strip("\n").strip(), image_names))


# In[ ]:





# In[ ]:


random_names = []
random_indices = []
while len(random_names) < random_size:
    random_index = random.randint(0, len(image_names)-1)
    if random_index not in random_indices:
        random_indices.append(random_index)
        random_names.append(image_names[random_index])
print("random names {}".format(random_names[:20]))


# In[ ]:


random_names = list(map(lambda x:"./." + x, random_names))


# In[ ]:


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


# In[ ]:


input_batch = get_image_tensor(random_names)


# In[ ]:


model = GoogLeNet.from_pretrained('googlenet')


# In[ ]:


extracted_featuers = model.extract_features(input_batch)


# In[ ]:


print("extracted features has shape {}".format(extracted_featuers.shape))


# In[ ]:


print("extracted features same as samved features {}".format(extracted_featuers == total_features[random_indices]))

