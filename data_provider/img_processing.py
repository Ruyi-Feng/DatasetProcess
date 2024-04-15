import cv2
import numpy as np
import os
from data_provider.utils import labels2img as labels2img
from data_provider.data_fomat_driver import DataConvert
import random
from data_provider.utils import load_top_pool, save_img, img2labels


class ImageCut:
    def __init__(
        self,
        ori_sample_dir,
        obj_class_list,
        enlarge_scale=1,
        top_pool_path=None,
        external_sample_dir=None,
        **kwargs
    ):
        """
        extend obj class list: []

        从这个list里选择可以裁剪剪贴的图像
        剪贴工作模式(按优先级排序)
        1. external_bbox: 按照class分了subdir, 直接找对应box obj的文件路径
        2. external_label: 如果有label,则在全label筛选。
        3. internal: 依赖top pool

        停止条件, enlarge_scale总数多少。
        """
        if enlarge_scale < 1:
            raise ValueError("enlarge_scale must be greater than 1")
        if top_pool_path is None and external_sample_dir is None:
            raise ValueError("top_pool_path or external_sample_dir must be not None")
        if top_pool_path is not None and not os.path.exists(top_pool_path):
            raise ValueError("top_pool_path not exist")
        if external_sample_dir is not None and not os.path.exists(external_sample_dir):
            raise ValueError("external_sample_dir not exist")
        if not os.path.exists(ori_sample_dir):
            raise ValueError("ori_sample_dir not exist")

        # 确定剪贴工作模式
        if external_sample_dir is not None:
            external_label_path = os.path.join(external_sample_dir, "labels")
            if not os.path.exists(external_label_path):
                self.work_mode = "external_bbox"
                self.external_img_dir = external_sample_dir
            else:
                self.work_mode = "external_label"
                self.external_img_dir = os.path.join(external_sample_dir, "images")
                self.external_label_dir = external_label_path
        else:
            self.work_mode = "internal"
            self.top_pool_path = top_pool_path
            self._load_top_pool()

        self.obj_class_list = obj_class_list
        self.enlarge_scale = enlarge_scale
        self.ori_sample_dir = ori_sample_dir
        self.data_convert = DataConvert(
            "yolo2xyxy"
        )  # x2xyxy 内部流通为xyxy不可更改，不然要改代码。
        self._generate_ori_img_list()
        self._init_remain_class()

    def _load_top_pool(self):
        self.top_pool = load_top_pool(self.top_pool_path)

    def _generate_ori_img_list(self):
        self.ori_label_list = os.listdir(os.path.join(self.ori_sample_dir, "labels"))

    def _init_remain_class(self):
        self.remain_class_dict = {}
        for obj_class in self.obj_class_list:
            self.remain_class_dict[obj_class] = self.enlarge_scale

    def _get_aim_class(self):
        for obj_class in self.remain_class_dict:
            if self.remain_class_dict[obj_class] > 0:
                self.remain_class_dict[obj_class] = (
                    self.remain_class_dict[obj_class] - 1
                )
                return obj_class

    def _still_remain_work(self):
        for obj_class in self.remain_class_dict:
            if self.remain_class_dict[obj_class] > 0:
                return True
        return False

    @staticmethod
    def _disted(px, py, xyxy):
        xvalid = px < min(xyxy[0], xyxy[2]) or px > max(xyxy[0], xyxy[2])
        yvalid = py < min(xyxy[1], xyxy[3]) or py > max(xyxy[1], xyxy[3])
        return xvalid and yvalid

    def _valid_pos(self, pos_x, pos_y, ori_labels):
        for label in ori_labels:
            if not self._disted(pos_x, pos_y, label[1:5]):
                return False
        return True

    def _choose_a_corner(self, ori_labels, fc, fr):
        """
        选corner的原则
        与原始obj的iou不超过某个阈值
        距离边界满足某个阈值
        """
        while True:
            pos_x = random.randint(self.bound_to_obj, fc - self.bound_to_obj)
            pos_y = random.randint(self.bound_to_obj, fr - self.bound_to_obj)
            if self._valid_pos(pos_x, pos_y, ori_labels):
                break
        return (pos_x, pos_y)

    def _choose_target_img(self):
        index = random.randint(0, len(self.ori_label_list) - 1)
        target_mark = self.ori_label_list[index]
        label_path = os.path.join(self.ori_sample_dir, "labels", target_mark)
        img_path = labels2img(label_path)
        target_img = cv2.imread(img_path)
        fc, fr = target_img.shape[1], target_img.shape[0]
        ori_labels = self.data_convert.load(label_path, fc, fr)
        pos = self._choose_a_corner(ori_labels, fc, fr)
        return ori_labels, target_img, pos, target_mark

    def _choose_aim_index(self, aim_cls, labels):
        index_list = []
        [
            index_list.append(int(i))
            for i, label in enumerate(labels)
            if label[0] == int(aim_cls)
        ]

        if len(index_list):
            idx = random.choice(index_list)
            return idx
        else:
            return None

    def _load_internal_img(self):
        aim_cls = self._get_aim_class()
        while True:
            label_name = random.choice(list(self.top_pool[aim_cls].keys()))
            labels = self.data_convert.load(
                os.path.join(self.ori_sample_dir, "labels", label_name), 1, 1
            )
            aim_cls_index = self._choose_aim_index(aim_cls, labels)
            if aim_cls_index is not None:
                break
        img_name = labels2img(os.path.join(self.ori_sample_dir, "labels", label_name))
        img = cv2.imread(img_name)
        labels = self.data_convert.load(
            os.path.join(self.ori_sample_dir, "labels", label_name),
            img.shape[1],
            img.shape[0],
        )
        obj = img[
            labels[aim_cls_index][1] : labels[aim_cls_index][3],
            labels[aim_cls_index][0] : labels[aim_cls_index][2],
        ]
        # 这里检验一下
        # cv2.imshow('obj', obj)
        return obj, aim_cls, label_name

    def _load_external_bbox(self):
        aim_cls = self._get_aim_class()
        img_name = random.choice(os.listdir(self.external_img_dir))
        img = cv2.imread(os.path.join(self.external_img_dir, img_name))
        return img, aim_cls, img_name

    def _load_external_label(self):
        aim_cls = self._get_aim_class()
        while True:
            label_name = random.choice(os.listdir(self.external_label_dir))
            labels = self.data_convert.load(
                os.path.join(self.external_label_dir, label_name), 1, 1
            )
            aim_cls_index = self._choose_aim_index(aim_cls, labels)
            if aim_cls_index is None:
                continue
            img = cv2.imread(
                labels2img(os.path.join(self.external_label_dir, label_name))
            )
            labels = self.data_convert.load(
                os.path.join(self.external_label_dir, label_name),
                img.shape[1],
                img.shape[0],
            )
            obj = img[
                labels[aim_cls_index][2] : labels[aim_cls_index][4],
                labels[aim_cls_index][1] : labels[aim_cls_index][3],
            ]
            return obj, aim_cls, label_name

    def _load_obj_img(self):
        if self.work_mode == "internal":
            return self._load_internal_img()
        elif self.work_mode == "external_bbox":
            return self._load_external_bbox()
        elif self.work_mode == "external_label":
            return self._load_external_label()

    def _paste_to_target(self, ori_labels, target_img, pos, obj_img, aim_cls):
        x1, y1 = pos
        x2 = x1 + obj_img.shape[1]
        y2 = y1 + obj_img.shape[0]
        gap_x, gap_y = (
            target_img[y1:y2, x1:x2].shape[1],
            target_img[y1:y2, x1:x2].shape[0],
        )
        target_img[y1:y2, x1:x2] = obj_img[:gap_y, :gap_x]
        ori_labels.append([aim_cls, x1, y1, x2, y2])
        return ori_labels, target_img

    def _combine_mark(self, target_mark, mark, cls, output_dir):
        target_mark.replace(".txt", "_")
        mark.replace(".", "_")
        save_nm = target_mark + "_" + mark + "_" + str(cls)
        i = 0
        while os.path.exists(os.path.join(output_dir, save_nm + ".txt")):
            i += 1
            save_nm = target_mark + "_" + mark + "_" + str(cls) + "_" + str(i)
        return save_nm

    def _save_new_img_and_labels(self, new_labels, new_img, save_name, output_dir):
        labels = self.data_convert.x2yolo(
            new_labels, new_img.shape[1], new_img.shape[0]
        )
        self.data_convert.save(output_dir, save_name, new_img, labels)

    def run(self, save_path="./cut_paste_imgs", bound_to_obj=20, **kwargs):
        """
        需要判断剪贴位置和别的目标iou相交不超过某个阈值
        随机选一个角点，距离原始label和边界有一定距离
        对目标缩放并计算缩放后的尺度
        增加label
        剪贴图像

        算法顺序：
        1. 随机选要贴的图, 随机在图上选角点
        2. 抓一个目标class 按照work mode 载入其obj原始尺寸的图片
        3. 角点和img 送入拼接函数
        """
        self.bound_to_obj = bound_to_obj
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        while self._still_remain_work():
            ori_labels, target_img, pos, target_mark = self._choose_target_img()
            obj_img, aim_cls, mark = self._load_obj_img()
            save_name = self._combine_mark(target_mark, mark, aim_cls, save_path)
            new_labels, new_img = self._paste_to_target(
                ori_labels, target_img, pos, obj_img, aim_cls
            )
            self._save_new_img_and_labels(new_labels, new_img, save_name, save_path)


class ImageLight:

    _ALPHA_ = 2.5  # 不可更改，contrast control (1.0 - 3.0)
    _BETA_ = 100  # 不可更改，brightness control (0 - 100)

    def __init__(
        self,
        dir=None,
        light_change_ratio: float = 0.2,
        contrast_change_ratio: float = 0.2,
        brightness_th: float = 0.8,
        contrast_th: float = 0.5,
        **kwargs
    ):
        """
        这里定义
        """
        self.dir = dir
        self.brightness_th = brightness_th
        self.contrast_th = contrast_th
        self.light_change_ratio = light_change_ratio
        self.contrast_change_ratio = contrast_change_ratio
        self._record_dir_info()
        self._get_total_change_num()

    def _record_dir_info(self):
        if len(os.listdir(self.dir)) == 0:
            raise ValueError("dir is empty")
        self.total_img_num = len(os.listdir(os.path.join(self.dir, "images")))
        self.img_list = os.listdir(os.path.join(self.dir, "images"))
        self.choosed_index = []

    def _get_total_change_num(self):
        if len(os.listdir(os.path.join(self.dir, "images"))) == 0:
            raise ValueError("dir is empty")
        self.total_change_num = int(
            len(os.listdir(os.path.join(self.dir, "images")))
            * (self.light_change_ratio + self.contrast_change_ratio)
        )
        self.max_alpha = (self._ALPHA_ - 1.0) * self.contrast_th + 1.0
        self.min_alpha = 1.0 - self.contrast_th
        self.max_beta = self._BETA_ * self.brightness_th
        self.min_beta = -self._BETA_ * self.brightness_th

    def _change_img(self, img, alpha, beta, if_show=False):
        new_img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
        if if_show:
            src = cv2.resize(new_img, dsize=None, fx=0.3, fy=0.3)
            cv2.imshow("new_img_alpha%.2f_beta%.2f" % (alpha, beta), src)
            # cv2.imshow("ori_img", img)
            cv2.waitKey(0)
        return new_img

    def _create_light_changed_img(self):
        while True:
            index = random.randint(0, self.total_img_num - 1)
            if index in self.choosed_index:
                continue
            else:
                self.choosed_index.append(index)
                break
        img_path = os.path.join(os.path.join(self.dir, "images"), self.img_list[index])
        label_path = img2labels(img_path)
        labels = DataConvert.load_yolo(label_path)
        img = cv2.imread(img_path)
        alpha, beta = 1.0, 0.0
        if_both_changed = random.random() < 0.3
        if if_both_changed:
            alpha = random.uniform(self.min_alpha, self.max_alpha)
            beta = random.uniform(self.min_beta, self.max_beta)
        elif random.random() < 0.5:
            alpha = random.uniform(self.min_alpha, self.max_alpha)
        else:
            beta = random.uniform(self.min_beta, self.max_beta)
        img = self._change_img(img, alpha, beta)
        mark = self.img_list[index] + "_alpha%.2f_beta%.2f" % (alpha, beta)
        return img, mark, labels

    def run(self, save_path=None, **kwargs):
        if save_path is None:
            save_path = self.dir
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        for i in range(self.total_change_num):
            img, mark, labels = self._create_light_changed_img()
            DataConvert.save_yolo(save_path, mark, img, labels)
