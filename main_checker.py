

import argparse
from data_provider.label_checking import Checker


def params():
    parser = argparse.ArgumentParser()
    # general settings
    parser.add_argument('--check_dir', type=str, default=None, help='check dir, not necessary if json_path is set')
    parser.add_argument('--json_path', type=str, default="check_result.json", help='json path')
    parser.add_argument('--if_check_inside_obj', type=bool, default=True, help='if check inside obj when operating checker')
    parser.add_argument('--data_frm', type=str, default="yolo2xyxy", help='load data convert format, yolo2xyxy, yolo2bbox')

    # processing settings
    parser.add_argument('--use_balencer', type=bool, default=True, help='number of samples to check')
    parser.add_argument('--use_checker', type=bool, default=True, help='number of samples to check')
    parser.add_argument('--use_img_visual_checking', type=bool, default=False, help='It should be used with --use_checker')
    parser.add_argument('--use_img_light_change', type=bool, default=False, help='apply light change on dataset')
    parser.add_argument('--light_change_ratio', type=float, default=0.1, help='ratio of images be changed in light')
    parser.add_argument('--use_img_cut_change', type=bool, default=False, help='apply cut change on dataset')
    parser.add_argument('--cut_change_ratio', type=float, default=0.2, help='ratio of obj be cut and pasted in dataset')
    return parser.parse_args()

def organize_dataset(args):
    # 先处理image cut change
    if args.use_img_cut_change:
        im_cut = ImageCut()
        pass

    # 再到balencer, 如果不需要再单独checker
    if args.use_balencer:
        balencer = Balencer(args.check_dir, args.data_frm, args.if_check_inside_obj, args.json_path, True)
        if args.use_img_visual_checking:
            balencer.check_samples()

    # 再到checker
    if args.use_checker and not args.use_balencer:
        checker = Checker(args.check_dir, args.data_frm, args.if_check_inside_obj, args.json_path, True)
        if args.use_img_visual_checking:
            checker.check_samples()
            checker.del_error_samples()

    # 再到img light change
    if args.use_img_light_change:
        im_light = ImageLight()
        pass



if __name__ == '__main__':
    checker = Checker("E:\\YOLOTrainingSetSamples\\CCTVval", if_check_inside_obj=True)  # , json_path="check_result.json"
    class_count = checker.count_class_number()
    checker.check_samples()
    # print(class_count)
    # print(checker.get_obj_number())
