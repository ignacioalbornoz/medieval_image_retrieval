# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 18:48:07 2023

@author: Mico
"""

import os
import torch
import numpy as np
from tqdm import tqdm
from skimage import io
from skimage.transform import resize
from skimage import morphology
from skimage import measure
from skimage.filters import threshold_otsu
from sklearn.decomposition import PCA

def get_grid_shape(h, w, patch_size):
    grid_shape = (h // patch_size), (w // patch_size)
    return grid_shape

def prepare_image(img, patch_size=14):
    #mean = (0.485, 0.456, 0.406) # DinoV2 mean std originales
    #std = (0.229, 0.224, 0.225)
    mean = (0.75811552, 0.7067092 , 0.61939638) # mean y std del dataset medieval
    std = (0.17571186, 0.17392427, 0.15809343)
    h, w, ch = img.shape
    grid_shape = get_grid_shape(h, w, patch_size)
    h_new = patch_size * grid_shape[0]
    w_new = patch_size * grid_shape[1]

    img_tensor = resize(img, (h_new, w_new, ch), order=3)
    img_tensor = (img_tensor - mean) / std
    img_tensor = img_tensor.transpose((2, 0, 1))
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor = torch.as_tensor(img_tensor, dtype=torch.float32).cuda()
    return img_tensor, grid_shape


def load_model(backbone_size='small'):
    backbone_archs = {"small": "vits14",
                      "base": "vitb14",
                      "large": "vitl14",
                      "giant": "vitg14"}

    backbone_arch = backbone_archs[backbone_size]
    backbone_name = f"dinov2_{backbone_arch}"
    model = torch.hub.load(repo_or_dir="facebookresearch/dinov2", model=backbone_name)
    model.eval()
    model.cuda()
    return model


def get_dense_descriptor(model, img):
    with torch.no_grad():
        img_tensor, grid_shape = prepare_image(img)
        features_tensor = model.patch_embed(img_tensor)
        attention_tensor = model.get_intermediate_layers(img_tensor)[0]

        features = features_tensor.cpu().detach().numpy()
        attention = attention_tensor.cpu().detach().numpy()

    del img_tensor
    del features_tensor
    del attention_tensor

    torch.cuda.empty_cache()

    features = np.squeeze(features)
    attention = np.squeeze(attention, axis=0)
    return features, attention, grid_shape


def min_max_scale(data):
    return (data - data.min()) / (data.max() - data.min())


def pca_colorize(features, output_shape, remove_bg=False):
    pca = PCA(n_components=3)
    pca.fit(features)
    rgb = pca.transform(features)
    rgb = min_max_scale(rgb)
    rgb = rgb.reshape(output_shape + (3,))

    if remove_bg:
        thresh = threshold_otsu(rgb[:, :, 0])
        rgb_mask = (rgb[:, :, 0] > thresh)*1
        rgb[:, :, 0] *= rgb_mask
        rgb[:, :, 1] *= rgb_mask
        rgb[:, :, 2] *= rgb_mask
        rgb = min_max_scale(rgb)

    return rgb

def foreground_mask(attention_rgb, use_bbox=True):
    attention_mask = attention_rgb.mean(axis=-1) > 0
    attention_mask = morphology.binary_dilation(attention_mask)
    if use_bbox:
        attention_labeled = measure.label(attention_mask)
        regions = measure.regionprops(attention_labeled)
        for props in regions:
            ymin, xmin, ymax, xmax = props.bbox
            attention_mask[ymin:ymax, xmin:xmax] = True
    return attention_mask

'''
if __name__ == "__main__":
    #dataset_path = r'D:\datasets\medieval_images\DocExplore_images'   
    dataset_path = r'/home/ignacio/2024-1/tesis/medieval_image_retrieval/DocExplore_images'   


    dinov2_sizes = {"small": 384,
                    "base": 768,
                    "large": 1024,
                    "giant": 1536} # tamaños del feature vector de cada version de dinov2

    backbone_size = 'small'
    model = load_model(backbone_size)

    image_filenames = os.listdir(dataset_path)[0:5] # las primeras 5 imagenes para probar


    
    for img_filename in tqdm(image_filenames):
        print(img_filename)
        image_path = os.path.join(dataset_path, img_filename)
        img = io.imread(image_path)


        features, attention, grid_shape = get_dense_descriptor(model, img)

        fmap_shape = grid_shape + (features.shape[-1],)
        
        # visualizar mapas usando PCA
      
        attention_rgb_bg = pca_colorize(attention, grid_shape, remove_bg=False)
       
        attention_rgb = pca_colorize(attention, grid_shape, remove_bg=True)
   
        attention_mask = foreground_mask(attention_rgb, use_bbox=False)
 
        attention_mask_box = foreground_mask(attention_rgb, use_bbox=True)

        features_rgb = pca_colorize(features, grid_shape, remove_bg=False)
        
        io.imshow(img)
        io.show()

        io.imshow(features_rgb)
        io.show()

        io.imshow(attention_rgb_bg)
        io.show()

        io.imshow(attention_rgb)
        io.show()

        io.imshow(attention_mask)
        io.show()

        io.imshow(attention_mask_box)
        io.show()
'''