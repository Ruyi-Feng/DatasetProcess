# DatasetProcess

一个数据集处理工具，用于对已有的数据集平衡样本，拼贴，更改光照与对比度等。

## 预处理功能列表

- **转换目标类别 --change_label_cls**

    在合成数据集时使用，用于将其他数据集类别编号转换成当前设置编号。如数据集A car为0，目标数据集car为1，本功能可将A内car转换为1

- **平衡数据集 --equal_img_label**

    将数据集中没有标签的图像或没有图像的标签删除。

- **划分训练集和测试集 --train_val_split**

    将总数据集按设定比例划分为训练集、验证集和测试集。

- **从MOT格式数据中制作训练集**

    从MOT或YOLO的数据格式中抽样制作训练集，MOT格式图片来源于视频，YOLO格式来源于图片。

- **滑窗切割数据集**

    用于无人机航拍视频训练集的滑窗切割，生成不同高度比例目标的数据集。

## 主体功能列表

- **检查统计数据集内容 --checker**

    检查现有数据集中样本数、目标数、目标类别数与对应类别所占比例。

- **平衡数据集类别 --balencer**

    对于类别比例偏差过大的数据集，对富余类别降采样，缺乏类别复采样，以达成数据集中各类别的基本平衡。

- **更改数据集光照 --light_change**

    按比例随机更改数据集的光照条件、对比度条件等。

- **剪切数据集样本 --img_cut**

    用于扩增某个特定类别的目标。将本数据集中样本、其他数据集中样本或已经裁剪好的图像拼贴到当前数据集中。同时增加标签，不与现有数据集中目标遮挡重叠。

# 使用文档

使用文档部分，分功能介绍了各模块的调用方式与对应参数含义。

## 预处理功能：转换目标类别

在合成数据集时使用，用于将其他数据集类别编号转换成当前设置编号。如数据集A car为0，目标数据集car为1，本功能可将A内car转换为1

### 脚本位置

``` python

from data_provider.data_fomat_driver import DataConvert

DataConvert.change_label_cls()

class DataConvert:
    @classmethod
    def change_label_cls(self, labels_path, ori_cls, new_cls, **kwargs)
        """change_label_cls

            labels_path: str
            -----------
            e.g. "/data1/fry/dataset/labels"
            需要具体到labels文件夹，文件夹内为每个样本的标签文件

            ori_cls: int
            -------
            e.g. 1
            需要改变的目标的原始cls

            new_cls: int
            -------
            e.g. 0
            需要改变的目标的新cls

        """

```

### 调用方式

配置script/change_label_cls.yml

``` yaml
labels_path: ""  # 这里写label文件夹的path
ori_cls: 1  # 需要改变的目标的原始cls
new_cls: 3  # 需要改变的目标的新cls
```

启动命令

```shell
# 从 main_chekcer 启动
python main_checker.py --change_label_cls --script_dir ./script
```

### 拆分测试

参考tests/test_checker

```python
def test_change_label_cls():
    DataConvert.change_label_cls(labels_path="J:\\yolov5projdataset\\Yolo Truck.v2i.yolov5pytorch\\train\\labels",
                                 ori_cls=1,
                                 new_cls=3)

```

<!--  ---------------------------------------    -->

## 预处理功能：平衡数据集

将数据集中没有标签的图像或没有图像的标签删除。

### 脚本位置

```python
from data_provider.utils import equal_img_labels

equal_img_labels(dir_path)

def equal_img_labels(dir_path):
    """equal_img_labels

    dir_path: str
    --------
    The path of dataset including sub dirs of images and labels
    """
```

### 调用方式

配置script/equal_img_label.yml

``` yaml
dir_path: "/data1/fry/dataset/"
```

启动命令

```shell
# 从 main_chekcer 启动
python main_checker.py --equal_img_label --script_dir ./script
```

### 拆分测试

参考tests/test_checker

```python
def test_euqal_img_labels(path):
    equal_img_labels(path)

```

<!--  ---------------------------------------    -->

## 预处理功能：划分训练集和测试集

将总数据集按设定比例划分为训练集、验证集和测试集。

### 脚本位置

```python
from data_provider.utils import train_val_split

train_val_split()

def train_val_split(ori_path, new_path, train_ratio=0.7, val_ratio=0.3, test_ratio=0.0):
    """train_val_split

    ori_path: str
    --------
    原始数据集的位置

    new_path: str
    --------
    新划分完的数据集的位置

    train_ratio: float
    -----------
    训练集比例

    val_ratio: float
    ---------
    验证集比例

    test_ratio: float
    ----------
    测试集比例

    """
    pass
```

### 调用方式

配置script/train_val_split.yml

``` yaml
ori_path: ""
new_path: ""
train_ratio: 0.7
val_ratio: 0.3
test_ratio: 0.0
```

启动命令

```shell
# 从 main_chekcer 启动
python main_checker.py --train_val_split --script_dir ./script
```

### 拆分测试

参考tests/test_checker

```python
def test_train_val_split():
    train_val_split(ori_path="E:\\yolov5projdataset\\afterbalence",
                    new_path="E:\\yolov5projdataset\\split_dataset",
                    train_ratio=0.7,
                    val_ratio=0.2,
                    test_ratio=0.1)
```

<!--  ---------------------------------------    -->

## 预处理功能：从MOT格式数据中制作训练集

从MOT或YOLO的数据格式中抽样制作训练集，MOT格式图片来源于视频，YOLO格式来源于图片。

### 脚本位置

```python
from utils.mot2yolo import Mot2Yolo

mot2yolo = Mot2Yolo(args)
```

### 调用方式

运行main_mot2yolo.py脚本，可选参数有：

```shell
# 通用参数
--mode   # select dataset from MOTcsv or YOLO
--ratio  # select the ratio of dataset
--save_dir  # the dir to save dataset

# MOT dataset params
--video_path  # MOT格式默认图像从video中截取，需要指定视频路径
--video_mark  # 随机字符，用于区分数据集名称
--csv_path  # MOT数据的csv文件地址

# YOLO dataset params
--ori_label_dir  # F:\data\samples\labels
--ori_img_dir  # F:\data\samples\images
```

## 预处理功能：滑窗切割数据集

对无人机航拍视频的滑窗切割脚本。

### 脚本位置

utils/hc_uav_fig.py

这个脚本是很多年以前的，注释在脚本里写的非常清楚，因为是预处理工作，也没有合并到这个pkg里。凑合用吧。

## 主体功能：检查统计数据集内容

检查现有数据集中样本数、目标数、目标类别数与对应类别所占比例。

### 脚本位置

```python
from data_provider.label_checking import Checker

checker = Checker(dir=path, fmt="yolo2xyxy", if_check_inside_obj=True,
                  json_path="check_result.json")
```

### 调用方式

配置script/checker.yml

算法会先检查json_path，如果有，会继承json里写的文件夹和进度。

``` yaml
# init params
dir: ""
fmt: "yolo2xyxy"
if_check_inside_obj: true
json_path: ""

# check_samples
if_random: true
rows: 3
cols: 3
save_path: "check_result.json"
col_pix: 1080
row_pix: 720

# del_error_samples
checker_log_path: "check_result.json"
```

启动命令

```shell
# 从 main_chekcer 启动
python main_checker.py --change_label_cls --script_dir ./script
```

查看结果：结果会储存在DataProcess下的check_result.json里。可以直接
cat check_result.json
查看

### 拆分测试

参考tests/test_checker

```python
def test_checker(path):
    checker = Checker(dir=path, fmt="yolo2xyxy", if_check_inside_obj=True,
                    #   json_path="check_result.json" # 如果已经check到一半可以直接load
                      )
```

<!--  ---------------------------------------    -->

## 主体功能：平衡数据集类别

对于类别比例偏差过大的数据集，对富余类别降采样，缺乏类别复采样，以达成数据集中各类别的基本平衡。

### 脚本位置

```python
from data_provider.label_checking import Balencer

balencer = Balencer(dir=path, fmt="yolo2xyxy", if_check_inside_obj=True,
                  dataset_info_json_path="check_result.json",
                  generate_top_pool=True,)
```

### 调用方式

配置script/balencer.yml

``` yaml
# checker init params
dir: ""
fmt: "yolo2xyxy"
if_check_inside_obj: true
json_path: ""

# check_samples
if_random: true
rows: 3
cols: 3
save_path: "check_result.json"
col_pix: 1080
row_pix: 720

# del_error_samples
checker_log_path: "check_result.json"

# balence
expect_dataset_scale: 0.7
tolerant_offset: 0.4
balenced_log_save_path: "balence_log.json"

# sampling_refer_balenced_log
balencer_log_path: "balence_log.json"
save_dir: ""
img_exts: [".jpg", ".png"]
label_ext: ".txt"

```

启动命令

```shell
# 从 main_chekcer 启动
python main_checker.py --balencer --script_dir ./script
```

### 拆分测试

参考tests/test_balencer

```python
def test_balencer():
    balencer = Balencer(dataset_info_json_path="check_result.json", generate_top_pool=True)
    # balencer.load_top_pool()
    balencer.balence(tolerant_offset=0.3,
                     expect_dataset_scale=0.30)
    balencer.sampling_refer_balenced_log("balence_log.json", "/data1/fry/balenced_combine_dataset/")
```

## 主体功能：更改数据集光照

按比例随机更改数据集的光照条件、对比度条件等。

### 脚本位置

```python
from data_provider.img_processing import ImageLight

lightchange = ImageLight(dir=path)
```

### 调用方式

配置script/light_change.yml

```yml
# init
dir: ""
light_change_ratio: 0.2
contrast_change_ratio: 0.2
brightness_th: 0.8
contrast_th: 0.5

# run
save_path: "light_change_img"
```

启动命令

```shell
# 从 main_chekcer 启动
python main_checker.py --light_change --script_dir ./script
```

### 拆分测试

参考tests/test_lightchange.py

```python
def test_lightchange(path):
    lightchange = ImageLight(dir=path)
    lightchange.run(save_path="E:\\yolov5projdataset\\changelight")
```

## 主体功能：剪切数据集样本

用于扩增某个特定类别的目标。将本数据集中样本、其他数据集中样本或已经裁剪好的图像拼贴到当前数据集中。同时增加标签，不与现有数据集中目标遮挡重叠。

有三种扩增模式：

（1）内部扩增。对当前数据集中的特定目标样本裁剪粘贴到内部随机某张图上，相应标签也增加。需要配置top_pool_path，依赖top_pool进行选取样本。top_pool由balencer生成。
（2）外部扩增。用额外的数据集中特定的目标列表obj_class_list裁剪粘贴到当前数据集中。需要配置external_sample_dir，其中存在labels文件夹，用于选取外部需要裁剪的目标。
（3）外部目标扩增。用已经裁剪好的图像直接粘贴到数据集中。需要配置external_sample_dir，但其中不应存在labels文件夹，直接包含裁剪的图像。

### 脚本位置

```python
from data_provider.img_processing import ImageCut
```

### 调用方式

配置script/img_cut.yml

```yml
# init
ori_sample_dir: ""   # 原始数据集存在文件夹，包含labels和images的父文件夹
obj_class_list: [""]  # 需要剪贴的cls列表
enlarge_scale: 1   # 需要剪贴多少张
top_pool_path: ""  # top_pool的地址，如果采用内部扩增需要配置
external_sample_dir: ""  # 外部数据集的地址，采用任何一种外部扩增都需要配置。

# run
save_path: ""  # 裁剪后的数据集存储位置
bound_to_obj: 40  # 裁剪过程距离原始目标和边缘的像素距离

```

### 拆分测试

参考tests/test_lightchange.py

```python
def test_cutchange_exteral_label():
    img_cut = ImageCut(
        ori_sample_dir="E:\\yolov5projdataset\\afterbalence",
        obj_class_list=["2", "3", "1", "4"],
        enlarge_scale=50,
        external_sample_dir="E:\\yolov5projdataset\\roadside",
    )
    img_cut.run(save_path="E:\\yolov5projdataset\\img_cut", bound_to_obj=40)
    # changed_num = len(os.listdir("G:\samples\img_cut\images"))
    # assert changed_num == 10


def test_cutchange_internal():
    img_cut = ImageCut(
        ori_sample_dir="G:\samples",
        obj_class_list=["2"],
        enlarge_scale=5,
        top_pool_path="./top_pool.json",
    )
    img_cut.run(save_path="G:\samples\img_cut", bound_to_obj=10)
    # changed_num = len(os.listdir("G:\samples\img_cut\images"))
    # assert changed_num == 10

def test_cutchange_exteral_bbox():
    img_cut = ImageCut(
        ori_sample_dir="G:\samples",
        enlarge_scale=5,
        external_sample_dir="G:\samples\imshot",
    )
    img_cut.run(save_path="G:\samples\img_cut", bound_to_obj=10)

```
