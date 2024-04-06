from tests.test_checker import (
    test_checker,
    test_balencer,
    test_euqal_img_labels,
    test_change_label_cls,
)
from tests.test_lightchange import (
    test_lightchange,
    test_cutchange_exteral_label,
    test_cutchange_internal,
    test_cutchange_exteral_bbox,
)
import time


if __name__ == "__main__":
    start_time = time.time()
    test_euqal_img_labels()
    test_change_label_cls()
    test_checker()
    test_balencer()
    test_lightchange()
    test_cutchange_exteral_label()
    test_cutchange_exteral_bbox()
    test_cutchange_internal()
    end_time = time.time()
    print("time:", end_time - start_time)
