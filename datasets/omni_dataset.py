import os
import cv2
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler
from torch import Tensor
from typing import Sequence

# prompt info dict
# task prompt
task_prompt_list = [
    "segmentation",
    "classification",
]
# position prompt
position_prompt_dict = {
    "BUS-BRA": "breast",
    "BUSIS": "breast",
    "CAMUS": "cardiac",
    "DDTI": "thyroid",
    "Fetal_HC": "head",
    "kidneyUS": "kidney",
    "UDIAT": "breast",
    "Appendix": "appendix",
    "Fatty-Liver": "liver",
    "BUSI": "breast",
}
# nature prompt
nature_prompt_dict = {
    "BUS-BRA": "tumor",
    "BUSIS": "tumor",
    "CAMUS": "organ",
    "DDTI": "tumor",
    "Fetal_HC": "organ",
    "kidneyUS": "organ",
    "UDIAT": "organ",
    "Appendix": "organ",
    "Fatty-Liver": "organ",
    "BUSI": "tumor",
}
# type prompt
available_type_prompt_list = [
    "BUS-BRA",
    "BUSIS",
    "CAMUS",
    "DDTI",
    "Fetal_HC",
    "kidneyUS",
    "UDIAT",
    "BUSI"
]

# prompt one-hot
# organ prompt
position_prompt_one_hot_dict = {
    "breast":  [1, 0, 0, 0, 0, 0, 0, 0],
    "cardiac": [0, 1, 0, 0, 0, 0, 0, 0],
    "thyroid": [0, 0, 1, 0, 0, 0, 0, 0],
    "head":    [0, 0, 0, 1, 0, 0, 0, 0],
    "kidney":  [0, 0, 0, 0, 1, 0, 0, 0],
    "appendix": [0, 0, 0, 0, 0, 1, 0, 0],
    "liver":   [0, 0, 0, 0, 0, 0, 1, 0],
    "indis":   [0, 0, 0, 0, 0, 0, 0, 1]
}
# task prompt
task_prompt_one_hot_dict = {
    "segmentation": [1, 0],
    "classification": [0, 1]
}
# nature prompt
nature_prompt_one_hot_dict = {
    "tumor": [1, 0],
    "organ": [0, 1],
}
# type prompt
type_prompt_one_hot_dict = {
    "whole": [1, 0, 0],
    "local": [0, 1, 0],
    "location": [0, 0, 1],
}


def list_add_prefix(txt_path, prefix_1, prefix_2):

    with open(txt_path, 'r') as f:
        lines = f.readlines()
    if prefix_2 is not None:
        return [os.path.join(prefix_1, prefix_2, line.strip('\n')) for line in lines]
    else:
        return [os.path.join(prefix_1, line.strip('\n')) for line in lines]


class WeightedRandomSamplerDDP(DistributedSampler):
    r"""Samples elements from ``[0,..,len(weights)-1]`` with given probabilities (weights).

    Args:
        data_set: Dataset used for sampling.
        weights (sequence)   : a sequence of weights, not necessary summing up to one
        num_replicas (int, optional): Number of processes participating in
            distributed training. By default, :attr:`world_size` is retrieved from the
            current distributed group.
        rank (int, optional): Rank of the current process within :attr:`num_replicas`.
            By default, :attr:`rank` is retrieved from the current distributed
            group.
        num_samples (int): number of samples to draw
        replacement (bool): if ``True``, samples are drawn with replacement.
            If not, they are drawn without replacement, which means that when a
            sample index is drawn for a row, it cannot be drawn again for that row.
        generator (Generator): Generator used in sampling.
    """
    weights: Tensor
    num_samples: int
    replacement: bool

    def __init__(self, data_set, weights: Sequence[float], num_replicas: int, rank: int, num_samples: int,
                 replacement: bool = True, generator=None) -> None:
        super(WeightedRandomSamplerDDP, self).__init__(data_set, num_replicas, rank)
        if not isinstance(num_samples, int) or isinstance(num_samples, bool) or \
                num_samples <= 0:
            raise ValueError("num_samples should be a positive integer "
                             "value, but got num_samples={}".format(num_samples))
        if not isinstance(replacement, bool):
            raise ValueError("replacement should be a boolean value, but got "
                             "replacement={}".format(replacement))
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.num_samples = num_samples
        self.replacement = replacement
        self.generator = generator
        self.num_replicas = num_replicas
        self.rank = rank
        self.weights = self.weights[self.rank::self.num_replicas]
        self.num_samples = self.num_samples // self.num_replicas

    def __iter__(self):
        rand_tensor = torch.multinomial(self.weights, self.num_samples, self.replacement, generator=self.generator)
        rand_tensor = self.rank + rand_tensor * self.num_replicas
        return iter(rand_tensor.tolist())

    def __len__(self):
        return self.num_samples


class USdatasetOmni_seg(Dataset):
    def __init__(self, base_dir, split, transform=None, prompt=False):
        self.transform = transform
        self.split = split
        self.data_dir = base_dir
        self.sample_list = []
        self.subset_len = []
        self.prompt = prompt

        self.sample_list.extend(list_add_prefix(os.path.join(
            base_dir, "segmentation", "BUS-BRA", split + ".txt"), "BUS-BRA", "imgs"))
        self.sample_list.extend(list_add_prefix(os.path.join(
            base_dir, "segmentation", "BUSIS", split + ".txt"), "BUSIS", "imgs"))
        self.sample_list.extend(list_add_prefix(os.path.join(
            base_dir, "segmentation", "CAMUS", split + ".txt"), "CAMUS", "imgs"))
        self.sample_list.extend(list_add_prefix(os.path.join(
            base_dir, "segmentation", "DDTI", split + ".txt"), "DDTI", "imgs"))
        self.sample_list.extend(list_add_prefix(os.path.join(base_dir, "segmentation",
                                "Fetal_HC", split + ".txt"), "Fetal_HC", "imgs"))
        self.sample_list.extend(list_add_prefix(os.path.join(base_dir, "segmentation",
                                "kidneyUS", split + ".txt"), "kidneyUS", "imgs"))
        self.sample_list.extend(list_add_prefix(os.path.join(
            base_dir, "segmentation", "UDIAT", split + ".txt"), "UDIAT", "imgs"))

        self.subset_len.append(len(list_add_prefix(os.path.join(
            base_dir, "segmentation", "BUS-BRA", split + ".txt"), "BUS-BRA", "imgs")))
        self.subset_len.append(len(list_add_prefix(os.path.join(
            base_dir, "segmentation", "BUSIS", split + ".txt"), "BUSIS", "imgs")))
        self.subset_len.append(len(list_add_prefix(os.path.join(
            base_dir, "segmentation", "CAMUS", split + ".txt"), "CAMUS", "imgs")))
        self.subset_len.append(len(list_add_prefix(os.path.join(
            base_dir, "segmentation", "DDTI", split + ".txt"), "DDTI", "imgs")))
        self.subset_len.append(len(list_add_prefix(os.path.join(
            base_dir, "segmentation", "Fetal_HC", split + ".txt"), "Fetal_HC", "imgs")))
        self.subset_len.append(len(list_add_prefix(os.path.join(
            base_dir, "segmentation", "kidneyUS", split + ".txt"), "kidneyUS", "imgs")))
        self.subset_len.append(len(list_add_prefix(os.path.join(
            base_dir, "segmentation", "UDIAT", split + ".txt"), "UDIAT", "imgs")))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):

        img_name = self.sample_list[idx].strip('\n')
        img_path = os.path.join(self.data_dir, "segmentation", img_name)
        label_path = os.path.join(self.data_dir, "segmentation", img_name).replace("imgs", "masks")

        image = cv2.imread(img_path)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

        dataset_name = img_name.split("/")[0]
        label_info = open(os.path.join(self.data_dir, "segmentation", dataset_name, "config.yaml")).readlines()

        label_info_list = [info.strip().split(":") for info in label_info]
        for single_label_info in label_info_list:
            label_index = int(single_label_info[0])
            label_value_in_image = int(single_label_info[2])
            label[label == label_value_in_image] = label_index

        label[label > 0] = 1

        if not self.prompt:
            sample = {'image': image/255.0, 'label': label}
        else:
            if random.random() > 0.5:
                x, y, w, h = cv2.boundingRect(label)
                length = max(w, h)

                if 0 in image[y:y+length, x:x+length, :].shape:
                    image = image
                    label = label
                    sample = {'image': image/255.0, 'label': label}
                    sample['type_prompt'] = type_prompt_one_hot_dict["whole"]
                else:
                    image = image[y:y+length, x:x+length, :]
                    label = label[y:y+length, x:x+length]
                    sample = {'image': image/255.0, 'label': label}
                    sample['type_prompt'] = type_prompt_one_hot_dict["local"]

            else:
                sample = {'image': image/255.0, 'label': label}
                sample['type_prompt'] = type_prompt_one_hot_dict["whole"]
                pass
        if self.transform:
            sample = self.transform(sample)
        sample['case_name'] = self.sample_list[idx].strip('\n')
        sample['nature_prompt'] = nature_prompt_one_hot_dict[nature_prompt_dict[dataset_name]]
        sample['position_prompt'] = position_prompt_one_hot_dict[position_prompt_dict[dataset_name]]
        sample['task_prompt'] = task_prompt_one_hot_dict["segmentation"]

        return sample


class USdatasetOmni_cls(Dataset):
    def __init__(self, base_dir, split, transform=None, prompt=False):
        self.transform = transform
        self.split = split
        self.data_dir = base_dir
        self.sample_list = []
        self.subset_len = []
        self.prompt = prompt

        self.sample_list.extend(list_add_prefix(os.path.join(
            base_dir, "classification", "Appendix", split + ".txt"), "Appendix", None))
        self.sample_list.extend(list_add_prefix(os.path.join(
            base_dir, "classification", "BUS-BRA", split + ".txt"), "BUS-BRA", None))
        self.sample_list.extend(list_add_prefix(os.path.join(base_dir, "classification",
                                "Fatty-Liver", split + ".txt"), "Fatty-Liver", None))
        self.sample_list.extend(list_add_prefix(os.path.join(
            base_dir, "classification", "UDIAT", split + ".txt"), "UDIAT", None))

        self.subset_len.append(len(list_add_prefix(os.path.join(
            base_dir, "classification", "Appendix", split + ".txt"), "Appendix", None)))
        self.subset_len.append(len(list_add_prefix(os.path.join(
            base_dir, "classification", "BUS-BRA", split + ".txt"), "BUS-BRA", None)))
        self.subset_len.append(len(list_add_prefix(os.path.join(base_dir, "classification",
                               "Fatty-Liver", split + ".txt"), "Fatty-Liver", None)))
        self.subset_len.append(len(list_add_prefix(os.path.join(
            base_dir, "classification", "UDIAT", split + ".txt"), "UDIAT", None)))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):

        img_name = self.sample_list[idx].strip('\n')
        img_path = os.path.join(self.data_dir, "classification", img_name)

        image = cv2.imread(img_path)
        dataset_name = img_name.split("/")[0]
        label = int(img_name.split("/")[-2])

        if not self.prompt:
            sample = {'image': image/255.0, 'label': np.zeros(image.shape[:2])}
        else:
            if dataset_name in available_type_prompt_list:
                random_number = random.random()
                mask_path = os.path.join(self.data_dir, "segmentation",
                                         "/".join([img_name.split("/")[0], "masks", img_name.split("/")[2]]))
                if random_number < 0.3:
                    sample = {'image': image/255.0, 'label': np.zeros(image.shape[:2])}
                    sample['type_prompt'] = type_prompt_one_hot_dict["whole"]
                elif random_number < 0.6:
                    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                    x, y, w, h = cv2.boundingRect(mask)
                    length = max(w, h)

                    if 0 in image[y:y+length, x:x+length, :].shape:
                        sample = {'image': image/255.0, 'label': np.zeros(image.shape[:2])}
                        sample['type_prompt'] = type_prompt_one_hot_dict["whole"]
                    else:
                        image = image[y:y+length, x:x+length, :]
                        sample = {'image': image/255.0, 'label': np.zeros(image.shape[:2])}
                        sample['type_prompt'] = type_prompt_one_hot_dict["local"]
                else:
                    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                    mask[mask > 0] = 255
                    image = image + (np.expand_dims(mask, axis=2)*0.1).astype('uint8')
                    sample = {'image': image/255.0, 'label': np.zeros(image.shape[:2])}
                    sample['type_prompt'] = type_prompt_one_hot_dict["location"]
            else:
                sample = {'image': image/255.0, 'label': np.zeros(image.shape[:2])}
                sample['type_prompt'] = type_prompt_one_hot_dict["whole"]
        if self.transform:
            sample = self.transform(sample)
        sample['label'] = torch.from_numpy(np.array(label))
        sample['case_name'] = self.sample_list[idx].strip('\n')
        sample['nature_prompt'] = nature_prompt_one_hot_dict[nature_prompt_dict[dataset_name]]
        sample['position_prompt'] = position_prompt_one_hot_dict[position_prompt_dict[dataset_name]]
        sample['task_prompt'] = task_prompt_one_hot_dict["classification"]

        return sample
