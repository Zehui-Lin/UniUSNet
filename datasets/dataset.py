import os
import cv2
import random
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset

from datasets.omni_dataset import position_prompt_dict
from datasets.omni_dataset import nature_prompt_dict

from datasets.omni_dataset import position_prompt_one_hot_dict
from datasets.omni_dataset import nature_prompt_one_hot_dict
from datasets.omni_dataset import type_prompt_one_hot_dict


def random_horizontal_flip(image, label):
    axis = 1
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        if 'type_prompt' in sample:
            type_prompt = sample['type_prompt']

        if random.random() > 0.5:
            image, label = random_horizontal_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y, _ = image.shape

        if x > y:
            image = zoom(image, (self.output_size[0] / y, self.output_size[1] / y, 1), order=1)
            label = zoom(label, (self.output_size[0] / y, self.output_size[1] / y), order=0)
        else:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / x, 1), order=1)
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / x), order=0)

        scale = random.uniform(0.8, 1.2)
        image = zoom(image, (scale, scale, 1), order=1)
        label = zoom(label, (scale, scale), order=0)

        x, y, _ = image.shape
        if scale > 1:
            startx = x//2 - (self.output_size[0]//2)
            starty = y//2 - (self.output_size[1]//2)
            image = image[startx:startx+self.output_size[0], starty:starty+self.output_size[1], :]
            label = label[startx:startx+self.output_size[0], starty:starty+self.output_size[1]]
        else:
            if x > self.output_size[0]:
                startx = x//2 - (self.output_size[0]//2)
                image = image[startx:startx+self.output_size[0], :, :]
                label = label[startx:startx+self.output_size[0], :]
            if y > self.output_size[1]:
                starty = y//2 - (self.output_size[1]//2)
                image = image[:, starty:starty+self.output_size[1], :]
                label = label[:, starty:starty+self.output_size[1]]
            x, y, _ = image.shape
            new_image = np.zeros((self.output_size[0], self.output_size[1], 3))
            new_label = np.zeros((self.output_size[0], self.output_size[1]))
            if x < y:
                startx = self.output_size[0]//2 - (x//2)
                starty = 0
                new_image[startx:startx+x, starty:starty+y, :] = image
                new_label[startx:startx+x, starty:starty+y] = label
            else:
                startx = 0
                starty = self.output_size[1]//2 - (y//2)
                new_image[startx:startx+x, starty:starty+y, :] = image
                new_label[startx:startx+x, starty:starty+y] = label
            image = new_image
            label = new_label

        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32))
        if 'type_prompt' in sample:
            sample = {'image': image, 'label': label.long(), 'type_prompt': type_prompt}
        else:
            sample = {'image': image, 'label': label.long()}
        return sample


class CenterCropGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        if 'type_prompt' in sample:
            type_prompt = sample['type_prompt']
        x, y, _ = image.shape
        if x > y:
            image = zoom(image, (self.output_size[0] / y, self.output_size[1] / y, 1), order=1)
            label = zoom(label, (self.output_size[0] / y, self.output_size[1] / y), order=0)
        else:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / x, 1), order=1)
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / x), order=0)
        x, y, _ = image.shape
        startx = x//2 - (self.output_size[0]//2)
        starty = y//2 - (self.output_size[1]//2)
        image = image[startx:startx+self.output_size[0], starty:starty+self.output_size[1], :]
        label = label[startx:startx+self.output_size[0], starty:starty+self.output_size[1]]

        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32))
        if 'type_prompt' in sample:
            sample = {'image': image, 'label': label.long(), 'type_prompt': type_prompt}
        else:
            sample = {'image': image, 'label': label.long()}
        return sample


class USdatasetSeg(Dataset):
    def __init__(self, base_dir, list_dir, split, transform=None, prompt=False):
        self.transform = transform
        self.split = split
        self.sample_list = open(os.path.join(list_dir, self.split+'.txt')).readlines()

        # BUSI
        self.sample_list = [sample for sample in self.sample_list if not "normal" in sample]

        self.data_dir = base_dir
        self.label_info = open(os.path.join(list_dir, "config.yaml")).readlines()
        self.prompt = prompt

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):

        img_name = self.sample_list[idx].strip('\n')
        img_path = os.path.join(self.data_dir, "imgs", img_name)
        label_path = os.path.join(self.data_dir, "masks", img_name)

        image = cv2.imread(img_path)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

        label_info_list = [info.strip().split(":") for info in self.label_info]
        for single_label_info in label_info_list:
            label_index = int(single_label_info[0])
            label_value_in_image = int(single_label_info[2])
            label[label == label_value_in_image] = label_index

        label[label > 0] = 1

        sample = {'image': image/255.0, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        sample['case_name'] = self.sample_list[idx].strip('\n')
        if self.prompt:
            dataset_name = img_path.split("/")[-3]
            sample['type_prompt'] = type_prompt_one_hot_dict["whole"]
            sample['nature_prompt'] = nature_prompt_one_hot_dict[nature_prompt_dict[dataset_name]]
            sample['position_prompt'] = position_prompt_one_hot_dict[position_prompt_dict[dataset_name]]
        return sample


class USdatasetCls(Dataset):
    def __init__(self, base_dir, list_dir, split, transform=None, prompt=False):
        self.transform = transform  # using transform in torch!
        self.split = split
        self.sample_list = open(os.path.join(list_dir, self.split+'.txt')).readlines()

        # BUSI
        self.sample_list = [sample for sample in self.sample_list if not "normal" in sample]

        self.data_dir = base_dir
        self.label_info = open(os.path.join(list_dir, "config.yaml")).readlines()
        self.prompt = prompt

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):

        img_name = self.sample_list[idx].strip('\n')
        img_path = os.path.join(self.data_dir, img_name)

        image = cv2.imread(img_path)
        label = int(img_name.split("/")[0])

        sample = {'image': image/255.0, 'label': np.zeros(image.shape[:2])}
        if self.transform:
            sample = self.transform(sample)
        sample['label'] = torch.from_numpy(np.array(label))
        sample['case_name'] = self.sample_list[idx].strip('\n')
        if self.prompt:
            dataset_name = img_path.split("/")[-3]
            sample['type_prompt'] = type_prompt_one_hot_dict["whole"]
            sample['nature_prompt'] = nature_prompt_one_hot_dict[nature_prompt_dict[dataset_name]]
            sample['position_prompt'] = position_prompt_one_hot_dict[position_prompt_dict[dataset_name]]
        return sample
