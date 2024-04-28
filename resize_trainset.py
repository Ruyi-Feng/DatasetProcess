
import os
import random
import cv2
from data_provider.data_fomat_driver import DataConvert
import numpy as np

class ResizeTrainset:

    def __init__(self, ori_path, new_path, resize_limit, ratio=1.0):
        if os.path.exists(ori_path):
            self.img_path = os.path.join(ori_path, "images")
            self.label_path = os.path.join(ori_path, "labels")
            self.ori_path = ori_path
        else:
            raise ValueError("path not exists")
        if not os.path.exists(new_path):
            os.makedirs(new_path)
        self.new_path = new_path
        self.resize_limit = min(resize_limit, 100)  # resize_limit
        self.ratio = ratio
        self._run()

    def _load_one_img_with_label(self, img_nm):
        img_path = os.path.join(self.img_path, img_nm)
        label_path = os.path.join(self.label_path, img_nm[:-4] + ".txt")
        img = cv2.imread(img_path)
        labels = DataConvert.load_yolo2xyxy(label_path, img.shape[1], img.shape[0])
        return img, labels

    def _resize_img(self, img, labels):
        # rezize img
        background = np.zeros_like(img)
        resize = random.randint(30, self.resize_limit)  # limit of percentage
        img = cv2.resize(img, (int(img.shape[1] * resize / 100), int(img.shape[0] * resize / 100)))
        edge_x = int((background.shape[1] - img.shape[1]) / 2)
        edge_y = int((background.shape[0] - img.shape[0]) / 2)
        background[edge_y:edge_y + img.shape[0], edge_x:edge_x + img.shape[1]] = img
        # resize labels
        for label in labels:
            # 因为是xyxy形式的label所以都要加edge
            label[1] = int(label[1] * resize / 100) + edge_x
            label[2] = int(label[2] * resize / 100) + edge_y
            label[3] = int(label[3] * resize / 100) + edge_x
            label[4] = int(label[4] * resize / 100) + edge_y
        return background, labels

    def _save_new_img_and_labels(self, img, labels, img_nm):
        labels = DataConvert.xyxy2yolo(labels, img.shape[1], img.shape[0])
        new_img_nm = img_nm[:-4] + "_resize" + img_nm[-4:]
        DataConvert.save_yolo(self.new_path, new_img_nm, img, labels)

    def _run(self):
        img_list = os.listdir(self.img_path)
        sample_list = random.sample(img_list, int(len(img_list) * self.ratio))
        for img_nm in sample_list:
            img, label = self._load_one_img_with_label(img_nm)
            img, label = self._resize_img(img, label)
            self._save_new_img_and_labels(img, label, img_nm)


if __name__ == "__main__":
    ori_path = "F:/data/mot2yolo"
    new_path = "f:/data/resized"
    resize_limit = 80
    ratio = 1
    ResizeTrainset(ori_path, new_path, resize_limit, ratio)
