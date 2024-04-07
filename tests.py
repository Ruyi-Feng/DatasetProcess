from tests.test_checker import (
    test_checker,
    test_balencer,
    test_euqal_img_labels,
    test_change_label_cls,
    test_train_val_split,
)
from tests.test_lightchange import (
    test_lightchange,
    test_cutchange_exteral_label,
    test_cutchange_internal,
    test_cutchange_exteral_bbox,
)
import time


if __name__ == "__main__":
    path = "E:\\yolov5projdataset\\afterbalence"
    start_time = time.time()
    test_euqal_img_labels(path)
    test_train_val_split()
    # test_change_label_cls()
    # test_checker(path)
    # test_balencer()
    # test_lightchange(path)
    # test_cutchange_exteral_label()
    # test_cutchange_exteral_bbox()
    # test_cutchange_internal()
    end_time = time.time()
    print("time:", end_time - start_time)
