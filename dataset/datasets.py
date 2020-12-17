import csv
import json
import numpy as np
import cv2
from torch.utils.data import Dataset
import os
import random
from dataset.transforms import *


def read_csv(csv_file):
    positive_sample, negative_sample = {}, {}
    fid = open(csv_file, 'r')
    data = csv.DictReader(fid)

    for row in data:
        try:
            img_path = row["img"]
            bbox_path = row["bbox"]
            anno_data = json.load(open(bbox_path, "r"))

            if len(anno_data["bbox"]) == 0:
                negative_sample[img_path] = []
            else:
                for idx, box in enumerate(anno_data["bbox"]):
                    bimg_path = img_path.replace(os.path.splitext(img_path)[-1], "_bimg{}".format(idx) + os.path.splitext(img_path)[-1])
                    positive_sample[bimg_path] = box["loc"]
        except Exception as e:
            raise ValueError(e + "read gt csv error...")

    return positive_sample, negative_sample


class ClsDataset(Dataset):
    def __init__(self, path, imgsz=(512, 512), crop_size=(64, 64), transforms=None, p_ratio=0.5):
        self.positive_sample, self.negative_sample = read_csv(path)
        self.transforms = transforms
        self.crop_size = crop_size
        self.imgsz = imgsz
        assert p_ratio <= 0.5, "positive case should little than negative case"
        self.p_ratio = p_ratio
        self.length = int(len(self.positive_sample) / self.p_ratio)

    def __getitem__(self, index):
        pos_keys = list(self.positive_sample.keys())
        neg_keys = list(self.negative_sample.keys())

        if index < len(self.positive_sample):
            bimg_path = pos_keys[index]
            ori_img_path = bimg_path.split('_bimg')[0] + os.path.splitext(bimg_path)[-1]
            box = self.positive_sample[bimg_path]
            box = np.array(box).astype(int)
            img = cv2.imread(ori_img_path)
            crop_img = box_crop(img, box)
            label = 1
        elif index < len(self.positive_sample) + len(self.negative_sample):
            ori_img_path = neg_keys[index - len(self.positive_sample)]
            img = cv2.imread(ori_img_path)
            crop_img = random_global_crop(img, self.imgsz, self.crop_size)
            label = 0
        else:
            random_idx = random.randint(0, len(self.positive_sample) - 1)
            bimg_path = pos_keys[random_idx]
            ori_img_path = bimg_path.split('_bimg')[0] + os.path.splitext(bimg_path)[-1]
            box = self.positive_sample[bimg_path]
            img = cv2.imread(ori_img_path)
            crop_img = random_box_crop(img, self.imgsz, box)
            label = 0

        if self.transforms:
            img = self.transforms(crop_img)
        else:
            img = crop_img
        # print(img.shape)
        return img, label

    def __len__(self):
        return self.length


if __name__ == "__main__":
    input_file = "/home/wenhai.zhang/WORK_SPACE/cdeg/DataSet/ESOPHAGUS_20201207/txt/train.csv"
    datasets = ClsDataset(input_file)
    a, label = datasets[0]
    cv2.imwrite("/home/wenhai.zhang/crop.png", a)
    print(a)


