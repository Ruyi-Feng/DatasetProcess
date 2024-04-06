
from data_provider.label_checking import Checker, Balencer
from data_provider.utils import equal_img_labels

def test_checker():
    checker = Checker(dir="G:\samples", fmt="yolo2xyxy", if_check_inside_obj=True)
    sample_num = checker.get_sample_number()
    print("sample_num: ", sample_num)
    assert sample_num == 19

def test_balencer():
    balencer = Balencer(dataset_info_json_path="check_result.json")
    balencer.balence(tolerant_offset=0.7, expect_dataset_scale=0.999)

def test_euqal_img_labels():
    equal_img_labels("J:\\yolov5projdataset\\roadside")