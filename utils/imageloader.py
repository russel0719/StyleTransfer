import os
import numpy as np
import torch
import torchvision.transforms as transforms
import cv2

def style_transform(image):
    style_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    image = style_transform(image)
    return image

def load_images_from_folder(path):
    images = []
    for filename in os.listdir(path):
        img = cv2.imread(os.path.join(path,filename), cv2.IMREAD_COLOR)
        if img is not None:
            images.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return images

def crop_and_resize_images(raw_images):
    images = []
    for image in raw_images:
        H, W, _ = image.shape
        img_size = min(H, W)
        H_s = int((H-img_size)/2)
        H_e = int((H+img_size)/2)
        W_s = int((W-img_size)/2)
        W_e = int((W+img_size)/2)
        crop_img = image[H_s:H_e, W_s:W_e]
        resize_img = cv2.resize(crop_img, (256, 256), cv2.INTER_AREA)
        images.append(style_transform(resize_img))
    images = torch.stack(images, 0)
    return images


def load_images(path, season):
    raw_images = load_images_from_folder(path+season)
    crop_images = crop_and_resize_images(raw_images)
    return crop_images