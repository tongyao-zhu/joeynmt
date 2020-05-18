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


# Check correctness of generated images
total_features = torch.load("./features_tensor.pt")


random_size = 100



with open("./sign_text_data/images_name_list.txt", "r") as f:
    image_names = f.readlines()


# In[ ]:


image_names = list(map(lambda x:x.strip("\n").strip(), image_names))


random_names = []
random_indices = []
while len(random_names) < random_size:
    random_index = random.randint(0, len(image_names)-1)
    if random_index not in random_indices:
        random_indices.append(random_index)
        random_names.append(image_names[random_index])
print("random names {}".format(random_names[:20]))


random_names = list(map(lambda x:"./." + x, random_names))


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



image_tensors = get_image_tensor(random_names)



model = GoogLeNet.from_pretrained('googlenet')
real_features = model.extract_features(image_tensors)
print(f"finished generating real features, has shape {real_features.shape}")

for index in range(len(random_indices)):
    print(f"currently  checking index {random_indices[index]}")
    real_tensor = image_tensors[index]
    acc_index = random_indices[index]
    larger_index = acc_index//10000
    # print(f"current larger index{larger_index}")
    saved_image_tensors = torch.load(f"./../image_tensors/image_tensor_{larger_index}.pt")
    offset = acc_index%10000
    assert (saved_image_tensors[offset]==real_tensor).all()
 	print("passed check of saved image")

 	real_feature = real_features[index]
 	saved_feature = total_features[acc_index]
 	assert (real_feature == saved_feature).all()
 	print("passed check of saved feature")
