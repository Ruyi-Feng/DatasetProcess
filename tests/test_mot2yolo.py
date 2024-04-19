from utils.mot2yolo import Mot2Yolo
import argparse

def params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='MOT', help='select dataset from MOTcsv or YOLO')
    parser.add_argument('--ratio', type=float, default=0.05, help='the ratio of dataset')
    parser.add_argument('--save_dir', type=str, default='E://yolov5projdataset//repair_UAV_dataset//tju', help='the dir to save dataset')
    parser.add_argument('--video_mark', type=str, default="tju8_", help='随机字符，用于区分数据集名称')

    # MOT dataset params
    parser.add_argument('--video_path', type=str, default='E://yolov5projdataset//UAVfry//tju//s2TJU8.avi', help='MOT格式默认图像从video中截取，需要指定视频路径')
    parser.add_argument('--csv_path', type=str, default='E://yolov5projdataset//UAVfry//tju//UTE_MOTCdet.csv', help="MOT数据的csv文件地址")

    # YOLO dataset params
    parser.add_argument('--ori_label_dir', type=str, default='F:\data\samples\labels', )
    parser.add_argument('--ori_img_dir', type=str, default='F:\data\samples\images', )
    return parser.parse_args()


def test_mot2yolo():
    args = params()
    mot2yolo = Mot2Yolo(args)
    mot2yolo.run()




