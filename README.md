# DatasetProcess

一个数据集处理工具，用于对已有的数据集平衡样本，拼贴，更改光照与对比度等。

## 功能列表

- **预处理功能：转换目标类别**

    在合成数据集时使用，用于将其他数据集类别编号转换成当前设置编号。如数据集A car为0，目标数据集car为1，本功能可将A内car转换为1

- **预处理功能：平衡数据集**

    将数据集中没有标签的图像或没有图像的标签删除。

- **预处理功能：划分训练集和测试集**

    将总数据集按设定比例划分为训练集、验证集和测试集。

- **主体功能：检查统计数据集内容**

    检查现有数据集中样本数、目标数、目标类别数与对应类别所占比例。

- **主体功能：平衡数据集类别**

    对于类别比例偏差过大的数据集，对富余类别降采样，缺乏类别复采样，以达成数据集中各类别的基本平衡。

- **主体功能：更改数据集光照**

    按比例随机更改数据集的光照条件、对比度条件等。

- **主体功能：剪切数据集样本**

    用于扩增某个特定类别的目标。将本数据集中样本、其他数据集中样本或已经裁剪好的图像拼贴到当前数据集中。同时增加标签，不与现有数据集中目标遮挡重叠。

- **主体功能：滑窗切割数据集**

    用于无人机航拍视频训练集的滑窗切割，生成不同高度比例目标的数据集。

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
<!--  ---写到这里了-----    -->
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

### 拆分测试



<!--  ---------------------------------------    -->

## 主体功能：检查统计数据集内容

将数据集中没有标签的图像或没有图像的标签删除。

### 脚本位置


### 调用方式

配置script/change_label_cls.yml

``` yaml

```

启动命令

```shell
# 从 main_chekcer 启动
python main_checker.py --change_label_cls --script_dir ./script
```

### 拆分测试


- **主体功能：平衡数据集类别**

    对于类别比例偏差过大的数据集，对富余类别降采样，缺乏类别复采样，以达成数据集中各类别的基本平衡。

- **主体功能：更改数据集光照**

    按比例随机更改数据集的光照条件、对比度条件等。

- **主体功能：剪切数据集样本**

    用于扩增某个特定类别的目标。将本数据集中样本、其他数据集中样本或已经裁剪好的图像拼贴到当前数据集中。同时增加标签，不与现有数据集中目标遮挡重叠。

- **主体功能：滑窗切割数据集**