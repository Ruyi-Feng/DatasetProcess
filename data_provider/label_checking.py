import os
import cv2
import random
import numpy as np
from data_provider.data_fomat_driver import DataConvert
from data_provider.utils import select_error_samples
from data_provider.utils import del_error_samples as del_errs
from data_provider.utils import sampling_refer_balenced_log as sampling
from data_provider.utils import load_json, save_json, save_top_pool, load_top_pool, labels2img
import json
import warnings


class Checker:
    """
    对外接口及功能

    __init__(): 初始化
    get_sample_number(): 获取样本总数
    count_class_number(): 统计每个类别的数量
    generate_statistics_class_result(): 生成类别统计结果
    check_samples(): 可视化检查样本正误问题
    del_error_samples(): 删除错误样本
    """

    def __init__(
        self, dir=None, fmt="yolo2xyxy", if_check_inside_obj=False, json_path=None,
        **kwargs
    ):
        """
        Initializes an instance of the class.
        Args:
            dir (str): The directory path.
            fmt (str, optional): The data format. Defaults to "yolo2xyxy".
            if_check_inside_obj (bool, optional): Whether to check inside objects and generating statistics results. Defaults to False.
            json_path (str, optional): The path to the JSON file. Defaults to None.
        Returns:
            None
        """
        print("init checker")
        self.data_convert = DataConvert(fmt)
        if json_path is not None and os.path.exists(json_path):
            half_work_json = load_json(json_path)
            self.start_i = half_work_json["start_i"]
            self.check_dir = half_work_json["check_dir"]
            self.obj_class_count = half_work_json["obj_class_count"]
            self.class_percentage = half_work_json["class_percentage"]
            self.error_label = half_work_json["error_label"]
            self.error_images = half_work_json["error_images"]
            self.rand_index = half_work_json["rand_index"]
            self.sample_number = half_work_json["sample_number"]
            self.total_obj_num = half_work_json["total_obj_num"]
            self._count_obj_number()
        else:
            if dir is None:
                raise ValueError("json path not exist, and dir is None")
            self.check_dir = dir
            self.start_i = 0
            self.obj_class_count = {}
            self.class_percentage = {}
            self.error_label = []
            self.error_images = []
            self.rand_index = []
            self._count_obj_number()
            if if_check_inside_obj:
                print("start checking inside obj")
                _ = self.count_class_number()
                print("start generate statistic class result")
                self.generate_statistics_class_result()
                self._update_json("check_result.json", 0)

    def _update_json(self, json_path, i):
        """
        用于更新检查结果
        """
        half_work_json = load_json(json_path)
        half_work_json["start_i"] = i
        half_work_json["check_dir"] = self.check_dir
        half_work_json["obj_class_count"] = self.obj_class_count
        half_work_json["class_percentage"] = self.class_percentage
        half_work_json["error_label"] = self.error_label
        half_work_json["error_images"] = self.error_images
        half_work_json["rand_index"] = self.rand_index
        half_work_json["sample_number"] = self.sample_number
        half_work_json["total_obj_num"] = self.total_obj_num
        save_json(half_work_json, json_path)

    def _count_obj_number(self):
        """
        检查文件夹中需要check label的目标个数
        """
        img_list = os.listdir(os.path.join(self.check_dir, "images"))
        self.label_list = os.listdir(os.path.join(self.check_dir, "labels"))
        if len(img_list) != len(self.label_list):
            warnings.warn("图片和标签数量不一致")
        self.sample_number = len(self.label_list)
        print("finish count sample number")

    def get_sample_number(self):
        return self.sample_number

    def count_class_number(self):
        for label in self.label_list:
            obj_lists = self.data_convert.load(
                os.path.join(self.check_dir, "labels", label), 1, 1
            )
            for line in obj_lists:
                obj_class = line[0]
                self.class_percentage.setdefault(obj_class, 0)
                if obj_class not in self.obj_class_count:
                    self.obj_class_count[obj_class] = 1
                else:
                    self.obj_class_count[obj_class] += 1
        return self.obj_class_count

    def generate_statistics_class_result(self):
        self.total_class_num = len(self.obj_class_count)
        self.total_obj_num = 0
        for k, v in self.obj_class_count.items():
            self.total_obj_num = self.total_obj_num + v
        for k in self.class_percentage:
            self.class_percentage[k] = self.obj_class_count[k] / self.total_obj_num
        print("total class num: ", self.total_class_num)
        print("total obj num: ", self.total_obj_num)
        print("each class percentage: ", self.class_percentage)

    def _load_one_img_with_label(self, index):
        label_path = os.path.join(self.check_dir, "labels", self.label_list[index])
        img_path = labels2img(label_path)
        img = cv2.imread(img_path)
        fc, fr = img.shape[1], img.shape[0]
        labels = self.data_convert.load(label_path, fc, fr)
        for label in labels:
            cv2.rectangle(
                img, (label[1], label[2]), (label[3], label[4]), (0, 255, 0), 4
            )
            cv2.putText(
                img,
                "%s"%label[0],
                (label[1], label[2] - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )
        img = cv2.resize(
            img,
            (self.img_cols_pixel, self.img_rows_pixel),
            interpolation=cv2.INTER_CUBIC,
        )
        return img

    def _load_imges_with_labels(self, i, img_in_frame):
        frame_imgs = np.zeros(
            (self.img_rows_pixel * self.rows, self.img_cols_pixel * self.cols, 3),
            dtype=np.uint8,
        )
        for j in range(0, img_in_frame):
            if (i + j) < self.sample_number:
                index = self.rand_index[i + j]
            else:
                break
            corner = (j % self.cols) * self.img_cols_pixel, (
                j // self.cols
            ) * self.img_rows_pixel
            """ 排列方式举例
            0, 1, 2,
            3, 4, 5,
            6, 7, 8
            """
            frame_imgs[
                corner[1] : corner[1] + self.img_rows_pixel,
                corner[0] : corner[0] + self.img_cols_pixel,
            ] = self._load_one_img_with_label(index)
        return frame_imgs

    def _add_item_to_error_list(self, point_list, i):
        for point in point_list:
            x, y = point
            index = (
                (y // self.img_rows_pixel) * self.cols + (x // self.img_cols_pixel) + i
            )
            self.error_label.append(self.label_list[index])
            self.error_images.append(labels2img(self.label_list[index]))

    def check_samples(
        self, if_random=True, rows=3, cols=3, save_path="check_result.json", col_pix=1080, row_pix=720,
        **kwargs
    ):
        """
        用于检查数据集是否符合要求
        if_random: 是否随机显示目标
        rows: 显示的行数
        """
        self.rows = rows
        self.cols = cols
        self.img_cols_pixel = col_pix // cols
        self.img_rows_pixel = row_pix // rows
        if len(self.rand_index) == 0:
            self.rand_index = list(range(self.sample_number))
            if if_random:
                random.shuffle(self.rand_index)
        img_in_frame = rows * cols

        frms = 0
        for i in range(self.start_i, self.sample_number, img_in_frame):
            frms += 1
            frame_imgs = self._load_imges_with_labels(i, img_in_frame)
            point_list = select_error_samples(frame_imgs)
            self._add_item_to_error_list(point_list, i)
            if frms % 10 == 0:
                self._update_json(save_path, i)

    def del_error_samples(self, checker_log_path):
        del_errs(checker_log_path)


class Balencer(Checker):
    """
    用于平衡采样数据集

    对外接口及功能
    __init__(): 用于初始化
    balence(): 用于启动平衡数据集的处理
    load_top_pool(): 用于加载之前保存的top pool
    sampling_refer_balenced_log(): 根据log文件进行平衡采样，存储至save dir
    """

    def __init__(
        self,
        dir=None,
        fmt="yolo2xyxy",
        if_check_inside_obj=True,
        dataset_info_json_path=None,
        generate_top_pool=True,
        **kwargs
    ):
        """
        初始化
        generate_top_pool: 是否生成top pool，如果是，那么每个文件的class分布会保存在json中
        """
        super(Balencer, self).__init__(
            dir, fmt, if_check_inside_obj, dataset_info_json_path
        )
        self._max_sample_num = 10
        self.balenced_percentage_of_each_class = 1.0 / len(self.obj_class_count)
        self.class_offset_percentage = {}
        for k, v in self.class_percentage.items():
            self.class_offset_percentage[k] = v - self.balenced_percentage_of_each_class

        if generate_top_pool:
            print("start generating top pool")
            # 生成label list中每个文件的class分布，生成top pool list
            self._overall_class_distribution()
            save_top_pool(self.top_pool, "top_pool.json")
            print("finish generating top pool")

    def _get_top_class(self, classes):
        return int(np.argmax(np.bincount(classes)))

    def _overall_class_distribution(self):
        """
        遍历label list中每个文件的class
        按照top比例把文件名归类到top pool dict中
        top_pool: dict
        {
            "0": {},
            "1": {"label1": [1, 7, 2], "label2": [2, 5, 3]},
            "2": {"label3": [1, 2, 7]},
        }
        """
        self.top_pool = {}
        for label in self.label_list:
            lines = self.data_convert.load(
                os.path.join(self.check_dir, "labels", label), 1, 1
            )
            if len(lines) == 0:
                continue
            cls = np.array(lines)[:, 0]
            top_class = self._get_top_class(cls)
            self.top_pool.setdefault(top_class, {})
            cls_count = np.zeros(len(self.obj_class_count))
            tmp_cls_count = np.bincount(cls)
            cls_count[:len(tmp_cls_count)] = tmp_cls_count
            self.top_pool[top_class].setdefault(label, cls_count)

    def load_top_pool(self):
        self.top_pool = load_top_pool("top_pool.json")

    def _init_sample_balenced_dataset(self, n):
        """
        self.sampled_class_num 是np.array，用于记录每个类别的采样数量
        self.samped_dataset 是dict，用于记录每个类别的采样结果
        self.sampled_dataset = {
            "1": {
                "label1": 0,
                "label2": 0,
            }
        }
        """
        self.samped_dataset = {}
        self.sampled_class_num = np.zeros(len(self.obj_class_count))
        for k in self.top_pool:
            sample_list = random.sample(self.top_pool[k].keys(), n)
            _ = self._add_sample_list_in_k(sample_list, k)
        # sampled class num
        sampled_obj_num = np.sum(self.sampled_class_num)
        return sampled_obj_num

    def _add_sample_list_in_k(self, sample_list, k):
        fail_sample_num = 0
        for label in sample_list:
            self.samped_dataset.setdefault(k, {})
            self.samped_dataset[k].setdefault(label, 0)
            if self.samped_dataset[k][label] > self._max_sample_num:
                fail_sample_num += 1
                continue
            self.samped_dataset[k][label] += 1
            self.sampled_class_num += self.top_pool[k][label]
        return fail_sample_num

    def _get_current_aim_class(self, current_offset, tolerant_offset=0.1):
        """
        用于检查类别采样偏差是否超过容忍偏差
        如果在容忍范围内，就随机选一个类别作为当前采样类别
        如果不在容忍范围内，则选择class比例最小的作为当前采样的类别
        """
        if current_offset > tolerant_offset:
            return int(np.argmin(self.sampled_class_num))
        else:
            current_cls = random.randint(0, len(self.sampled_class_num) - 1)
            return current_cls

    def _get_current_offset(self):
        return (self.sampled_class_num.max() - self.sampled_class_num.min()) / self.sampled_class_num.sum()

    def _should_further_sampling(self, tolerant_offset, supposed_sampled_num):
        # if self._get_current_offset() < tolerant_offset:
        if self.sampled_class_num.sum() >= supposed_sampled_num:
            return False
        else:
            return True

    def _further_sample(
        self, supposed_sampled_num, tolerant_offset=0.1, sampled_batch=10
    ):
        """
        进一步增加sampled_dataset 直到满足supposed_sampled_num
        可以容忍的class 偏差为0.1

        算法逻辑：
        目标: 追求offset在合理范围内, 如果在合理范围内就随机选一个采样。
        while 可以继续采样，即不满足期望采样数，或超过了容忍偏差
            采样并加入样本, 控制同一sample的采样数不超过self._max_sample_num
            如果采样失败率大于80%的采样batch,则退出。
            （意味着小样本采样已经过多）
        """

        while self._should_further_sampling(tolerant_offset, supposed_sampled_num):
            current_aim_class = self._get_current_aim_class(
                self._get_current_offset(), tolerant_offset
            )
            sample_list = random.sample(
                self.top_pool[current_aim_class].keys(), sampled_batch
            )
            fail_sample_num = self._add_sample_list_in_k(sample_list, current_aim_class)
            if fail_sample_num > 0.8 * sampled_batch:
                break
        return (self.sampled_class_num.sum() >= supposed_sampled_num), (
            self._get_current_offset() < tolerant_offset
        )

    def balence(
        self,
        expect_dataset_scale=0.7,
        tolerant_offset=0.1,
        balenced_log_save_path="balence_log.json",
        **kwargs
    ):
        """
        用于平衡数据集
        # 组织成label，持续记录每个class的比例，不断加入文件名的dict，采样次数，采样次数大于N的不再重复采样，从top pool中删去。
        # 如平衡到达需要采样总数，暂停，并把文件名存入json
        # 如果没有到达，可接受最大偏移量，后续加入文件名。
        # 实在采不到就算了。提示增加样本数。
        """
        supposed_sampled_num = int(expect_dataset_scale * self.total_obj_num)
        # Step 1: 找到最大的可一次性采样数
        max_once_sample_num = min([len(self.top_pool[i]) for i in self.top_pool])
        sampled_num = self._init_sample_balenced_dataset(
            min(max_once_sample_num, supposed_sampled_num)
        )
        if sampled_num < supposed_sampled_num:
            count_meet, offset_meet = self._further_sample(
                supposed_sampled_num, tolerant_offset
            )
            print(f"count_meet: {count_meet}, offset_meet: {offset_meet}")
            if not count_meet:
                print("===============Warning=================")
                print(
                    "[Sample Num] can't meet the expect sample num, please check the dataset"
                )
                print("current sampled obj num: ", self.sampled_class_num.sum())
            if not offset_meet:
                print("===============Warning=================")
                print("[Offset] can't meet the expect offset, please check the dataset")
                print("current sampled min class: ", np.argmin(self.sampled_class_num))
                print("current sampled max class: ", np.argmax(self.sampled_class_num))
        save_json(self.samped_dataset, balenced_log_save_path)

    def sampling_refer_balenced_log(
        self, balencer_log_path, save_dir, img_exts=[".jpg", ".png"], label_ext=".txt",
        **kwargs
    ):
        sampling(balencer_log_path, self.check_dir, save_dir, img_exts, label_ext)
