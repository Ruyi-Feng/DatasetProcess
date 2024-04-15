
import argparse
import os
from data_provider.label_checking import Checker, Balencer
from data_provider.utils import equal_img_labels, train_val_split
from data_provider.img_processing import ImageLight, ImageCut
from data_provider.data_fomat_driver import DataConvert
import yaml


def params():
    parser = argparse.ArgumentParser()
    # general process portal
    parser.add_argument('--checker', action='store_true', help='start checker')
    parser.add_argument('--balencer', action='store_true', help='sampling training dataset')
    parser.add_argument('--use_img_visual_checking', action='store_true', help='check imgs and labels in dataset, use with balencer or checker')
    parser.add_argument('--light_change', action='store_true', help='change light and contrast in dataset')
    parser.add_argument('--img_cut', action='store_true', help='cut and paste obj in or cross datasets')

    # utils portal
    parser.add_argument('--equal_img_label', action='store_true', help='for del samples without labels or images')
    parser.add_argument('--train_val_split', action='store_true', help='split dataset into train val and test')
    parser.add_argument('--change_label_cls', action='store_true', help='change one class into another')

    # processing settings
    parser.add_argument('--script_dir', type=str, default='./script', help='script load from ...')

    return parser.parse_args()

def get_settings(dir, name):
    script = os.path.join(dir, name)
    with open(script, 'r') as f:
        settings = yaml.load(f, Loader=yaml.FullLoader)
    return settings

def pre_process_dataset(args):
    script_dir = args.script_dir

    # 先转换目标类别
    if args.change_label_cls:
        settings = get_settings(script_dir, "change_label_cls.yml")
        DataConvert.change_label_cls(**settings)

    # 平衡数据集，删除多余的
    if args.equal_img_label:
        settings = get_settings(script_dir, "equal_img_label.yml")
        equal_img_labels(**settings)

    # 划分训练集和测试集
    if args.train_val_split:
        settings = get_settings(script_dir, "train_val_split.yml")
        train_val_split(**settings)


def organize_dataset(args):
    script_dir = args.script_dir

    # 先处理image cut change
    if args.img_cut:
        settings = get_settings(script_dir, "img_cut.yml")
        im_cut = ImageCut(**settings)
        im_cut.run(**settings)

    # 再到balencer, 如果不需要再单独checker
    if args.balencer:
        settings = get_settings(script_dir, "balencer.yml")
        balencer = Balencer(**settings)
        if args.use_img_visual_checking:
            balencer.check_samples()
            balencer.del_error_samples()
        balencer.sampling_refer_balenced_log(**settings)

    # 再到checker
    if args.checker and not args.balencer:
        settings = get_settings(script_dir, "checker.yml")
        checker = Checker(**settings)
        if args.use_img_visual_checking:
            checker.check_samples()
            checker.del_error_samples()

    # 再到img light change
    if args.light_change:
        settings = get_settings(script_dir, "light_change.yml")
        im_light = ImageLight(**settings)
        im_light.run(**settings)



if __name__ == '__main__':
    args = params()
    pre_process_dataset(args)
    organize_dataset(args)
