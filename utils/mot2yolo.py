import os
import pandas as pd
import random
import shutil
import string
import cv2

"""
有一定的筛选率，选择某些帧的图片和label作为training dataset
有查看数据集标签的能力 独立出来
img/video

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

    def _copy(ori_file, cur_file):
        shutil.copy(ori_file, cur_file)

    def copy2dir(self, select_list):
        save_dir = self.args.save_dir
        ori_img_dir = self.args.ori_img_dir
        ori_label_dir = self.args.ori_label_dir
        if not os.path.exists(os.path.join(save_dir, "images")):
            os.makedirs(os.path.join(save_dir, "images"))
        if not os.path.exists(os.path.join(save_dir, "labels")):
            os.makedirs(os.path.join(save_dir, "labels"))
        for i in select_list:
            self._copy(os.path.join(ori_img_dir, i[:-4] + ".jpg"), os.path.join(save_dir, "images", i[:-4] + ".jpg"))
            self._copy(os.path.join(ori_label_dir, i), os.path.join(save_dir, "labels", i))

    def _run_yolo(self):
        obj_list = os.listdir(self.args.ori_label_dir)
        obj_list = self._select_txt(obj_list)
        select_list = random.sample(obj_list, int(len(obj_list) * self.args.select_ratio))
        self.copy2dir(select_list)

    def _capture_img(self, cap, frame_num, select_frm):
        save_dir = self.args.save_dir
        if not os.path.exists(os.path.join(save_dir, "images")):
            os.makedirs(os.path.join(save_dir, "images"))
        if self.args.video_mark is None:
            mark = random_string()
        for i in range(len(frame_num)):
            ret, frame = cap.read()
            if ret and (i in select_frm):
                cv2.imwrite(os.path.join(save_dir, "images", mark + str(i) + ".jpg"), frame)

    # MOT dataset convert
    def _run_mot(self):
        cap = cv2.VideoCapture(self.args.video_path)
        frame_num = cap.get(7)
        width = cap.get(3)
        height = cap.get(4)
        select_frm = random.sample(range(0, int(frame_num)), int(frame_num * self.args.select_ratio))
        self._capture_img(cap, frame_num, select_frm)
        pass

def random_string(length=8):
    strings = ''
    strings.join(random.SystemRandom().choice(string.ascii_letters + string.digits) for _ in range(length))
    return strings