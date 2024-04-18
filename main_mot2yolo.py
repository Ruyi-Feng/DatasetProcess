
import argparse



def params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='MOT', help='select dataset from MOTcsv or YOLO')
    parser.add_argument('--ratio', type=float, default=0.7, help='the ratio of dataset')
    parser.add_argument('--save_dir', type=str, default='', help='the dir to save dataset')

    # MOT dataset params
    parser.add_argument('--video_path', type=str, default='', )
    parser.add_argument('--video_mark', type=str, default=None, )
    parser.add_argument('--csv_path', type=str, default='', )

    # YOLO dataset params
    parser.add_argument('--ori_label_dir', type=str, default='', )
    parser.add_argument('--ori_img_dir', type=str, default='', )
    return parser.parse_args()



