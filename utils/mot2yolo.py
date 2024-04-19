import os
import pandas as pd
import numpy as np
import random
import shutil
import string
import cv2
from data_provider.data_fomat_driver import DataConvert
from data_provider.utils import save_json, load_json

"""
有一定的筛选率，选择某些帧的图片和label作为training dataset
有查看数据集标签的能力 独立出来
yolo/video_mot

"""
class Mot2Yolo:

    def __init__(self, args, **kwargs):
        self._run_api = {
            "MOT": self._run_mot,
            "YOLO": self._run_yolo,
        }
        self.args = args
        self.run = self._run_api[self.args.mode]

    # _yolo dataset select_
    def _select_txt(self, obj_list):
        txt_list = []
        for i in obj_list:
            if i.endswith(".txt"):
                txt_list.append(i)
        return txt_list

    def copy2dir(self, select_list):
        save_dir = self.args.save_dir
        ori_img_dir = self.args.ori_img_dir
        ori_label_dir = self.args.ori_label_dir
        if not os.path.exists(os.path.join(save_dir, "images")):
            os.makedirs(os.path.join(save_dir, "images"))
        if not os.path.exists(os.path.join(save_dir, "labels")):
            os.makedirs(os.path.join(save_dir, "labels"))
        for i in select_list:
            _copy(os.path.join(ori_img_dir, i[:-4] + ".jpg"), os.path.join(save_dir, "images", self.video_mark + i[:-4] + ".jpg"))
            _copy(os.path.join(ori_img_dir, i[:-4] + ".png"), os.path.join(save_dir, "images", self.video_mark + i[:-4] + ".png"))
            _copy(os.path.join(ori_label_dir, i), os.path.join(save_dir, "labels", self.video_mark + i))

    def _run_yolo(self):
        obj_list = os.listdir(self.args.ori_label_dir)
        obj_list = self._select_txt(obj_list)
        select_list = random.sample(obj_list, int(len(obj_list) * self.args.ratio))
        self.copy2dir(select_list)

    # MOT dataset convert
    def _capture_img(self, cap, frame_num, select_frm):
        save_dir = self.args.save_dir
        if not os.path.exists(os.path.join(save_dir, "images")):
            os.makedirs(os.path.join(save_dir, "images"))
        for i in range(int(frame_num)):
            ret, frame = cap.read()
            if ret and (i in select_frm):
                cv2.imwrite(os.path.join(save_dir, "images", self.mark + str(i) + ".jpg"), frame)

    def _mot2yolo(self, labels, width, height):
        # 因为mot没有cls，所以默认是0
        yolo = np.zeros((len(labels), 5))
        yolo[:, 1] = (labels[:, 2] + labels[:, 4] / 2.0) / width
        yolo[:, 2] = (labels[:, 3] + labels[:, 5] / 2.0) / height
        yolo[:, 3] = labels[:, 4] / width
        yolo[:, 4] = labels[:, 5] / height
        np.around(yolo, decimals=6, out=yolo)
        return yolo

    def _save_yolo(self, labels, frm):
        save_dir = self.args.save_dir
        if not os.path.exists(os.path.join(save_dir, "labels")):
            os.makedirs(os.path.join(save_dir, "labels"))
        with open(os.path.join(save_dir, "labels", self.mark + str(frm) + ".txt"), "w") as f:
            for i in labels:
                f.write("{} {} {} {} {}\n".format(int(i[0]), i[1], i[2], i[3], i[4]))

    def _select_mot_labels(self, mot_labels, select_frm, width, height):
        for i in select_frm:
            labels = mot_labels[mot_labels[0] == i]
            labels = self._mot2yolo(labels.values, width, height)
            self._save_yolo(labels, i)

    def _capture_labels(self, select_frm, width, height):
        save_dir = self.args.save_dir
        if not os.path.exists(os.path.join(save_dir, "labels")):
            os.makedirs(os.path.join(save_dir, "labels"))
        mot_labels = pd.read_csv(self.args.csv_path, header=None)
        mot_labels = mot_labels.sort_values(by=0, ascending=True)
        mot_labels = mot_labels.reset_index(drop=True)

        self._select_mot_labels(mot_labels, select_frm, width, height)

    def _save_tmp_select_frm(self, select_frm):
        selected_json = {"selected": select_frm}
        save_json(selected_json, "./selected_frms.json")

    def _run_mot(self):
        cap = cv2.VideoCapture(self.args.video_path)
        frame_num = cap.get(7)
        width = cap.get(3)
        height = cap.get(4)
        if not os.path.exists("./selected_frms.json"):
            select_frm = random.sample(range(0, int(frame_num)), int(frame_num * self.args.ratio))
            self._save_tmp_select_frm(select_frm)
        else:
            select_frm = load_json("./selected_frms.json")["selected"]
        if self.args.video_mark is None:
            self.mark = random_string()
        else:
            self.mark = self.args.video_mark
        self._capture_img(cap, frame_num, select_frm)
        self._capture_labels(select_frm, width, height)

class Visual:
    def __init__(self, path):
        self.path = path
        img_list = os.listdir(os.path.join(path, "images"))
        self.visual(img_list)

    @staticmethod
    def _draw_frm(img, label):
        for i in label:
            cv2.rectangle(img, (i[1], i[2]), (i[3], i[4]), (255-i[0]*100, i[0]*100, 0), 4)
        return img

    def visual(self, img_list):
        for img_nm in img_list:
            img_path = os.path.join(self.path, "images", img_nm)
            img = cv2.imread(img_path)
            label_path = os.path.join(self.path, "labels", img_nm[:-4] + ".txt")
            labels = DataConvert.load_yolo2xyxy(label_path, img.shape[1], img.shape[0])
            img = self._draw_frm(img, labels)
            cv2.imwrite(os.path.join(self.path, "ref", img_nm), img)
            # cv2.imshow("img", img)
            # cv2.waitKey(0)

def visual_yolo(path):
    imgs_path = os.path.join(path, "images")
    imgs = os.listdir(imgs_path)
    for img in imgs:
        img_path = os.path.join(path, "images", img)
        img = cv2.imread(img_path)
        cv2.imshow("img", img)
        cv2.waitKey(0)

def random_string(length=8):
    strings = ''
    strings.join(random.SystemRandom().choice(string.ascii_letters + string.digits) for _ in range(length))
    return strings

def _copy(ori_file, cur_file):
    if os.path.exists(ori_file):
        shutil.copy(ori_file, cur_file)
