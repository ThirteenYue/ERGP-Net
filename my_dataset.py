import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
import cv2
import random
import torch


def _mask_to_img(mask_file):
    img_file = re.sub('masks', 'images', mask_file)
    img_file = re.sub('\.png$', '.png', img_file)
    return img_file



class DriveDataset(Dataset):
    def __init__(self, root: str, train: bool, transforms=None):
        super(DriveDataset, self).__init__()
        self.flag = "Train(COVID-19)" if train else "Val(COVID-19)"
        data_root = os.path.join(root, "COVID", self.flag)
        assert os.path.exists(data_root), f"path '{data_root}' does not exists."
        self.transform = transforms
        img_names = [i for i in os.listdir(os.path.join(data_root, "images")) if i.endswith(".png")]
        self.img_list = [os.path.join(data_root, "images", i) for i in img_names]
        # self.manual = [os.path.join(data_root, "1st_manual", i.split("_")[] + "_manual1.gif")
        #                for i in img_names]
        # 读取数据集COVID
        img_names = [i for i in os.listdir(os.path.join(data_root, "infection_masks")) if i.endswith(".png")]
        self.infection = [os.path.join(data_root, "infection_masks", i) for i in img_names]
        # img_names = [i for i in os.listdir(os.path.join(data_root, "lung_masks")) if i.endswith(".png")]
        # self.lung = [os.path.join(data_root, "lung_masks", i) for i in img_names]

    def __getitem__(self, idx):
        img = cv2.imread(self.img_list[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(self.infection[idx])

        # mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        # mask = mask[:, :, 0: 2]

        seed = random.randint(0, 2 ** 32)

        if self.transform is not None:
            # Apply transform to img
            random.seed(seed)
            torch.cuda.manual_seed(seed)
            torch.manual_seed(seed)
            img = Image.fromarray(img)
            img = self.transform(img)

            # Apply same transform to mask
            random.seed(seed)
            torch.cuda.manual_seed(seed)
            torch.manual_seed(seed)
            mask = Image.fromarray(mask)  # array 转化为image
            mask = self.transform(mask)  # tensor 的排列方式 NCWH
        # img_np= np.array(img)
        # print(img_np)
        mask = np.array(mask)
        # print(mask.shape)
        labels = mask[0, :, :]
        # labels = np.zeros_like(mask[0, :, :])
        # labels[np.where((mask[0, :, :] > 0) & (mask[0, :, :]  < 0.004))] = 1 # hair
        # labels[np.where(mask[0, :, :] > 0.004)] = 2  # face
        # labels = np.expand_dims(labels, axis=0)

        return img, np.int64(labels)

    def __len__(self):
        return len(self.img_list)
