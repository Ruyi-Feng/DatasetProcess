
from data_provider.label_checking import Checker, Balencer
from data_provider.utils import equal_img_labels, train_val_split
from data_provider.data_fomat_driver import DataConvert

def test_checker(path):
    checker = Checker(dir=path, fmt="yolo2xyxy", if_check_inside_obj=True,
                    #   json_path="check_result.json"
                      )
    sample_num = checker.get_sample_number()
    print("sample_num: ", sample_num)
    # checker.check_samples(if_random=True, rows=3, cols=3, save_path="check_result.json", col_pix=1920, row_pix=1080)
    # checker.del_error_samples("check_result.json")
    # assert sample_num == 19

def test_balencer():
    balencer = Balencer(dataset_info_json_path="check_result.json", generate_top_pool=True)
    # balencer.load_top_pool()
    balencer.balence(tolerant_offset=0.3,
                     expect_dataset_scale=0.30)
    balencer.sampling_refer_balenced_log("balence_log.json", "/data1/fry/balenced_combine_dataset/")

def test_euqal_img_labels(path):
    equal_img_labels(path)

def test_change_label_cls():
    DataConvert.change_label_cls(labels_path="J:\\yolov5projdataset\\Yolo Truck.v2i.yolov5pytorch\\train\\labels",
                                 ori_cls=1,
                                 new_cls=3)

def test_del_aim_cls():
    DataConvert.del_aim_cls(labels_path="J:\\yolov5projdataset\\Yolo Truck.v2i.yolov5pytorch\\train\\labels",
                            del_cls=[])

def test_train_val_split():
    train_val_split(ori_path="E:\\yolov5projdataset\\afterbalence",
                    new_path="E:\\yolov5projdataset\\split_dataset",
                    train_ratio=0.7,
                    val_ratio=0.2,
                    test_ratio=0.1)

