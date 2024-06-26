from data_provider.img_processing import ImageLight, ImageCut
import os


def test_lightchange(path):
    lightchange = ImageLight(dir=path)
    lightchange.run(save_path="E:\\yolov5projdataset\\changelight")
    # changed_num = len(os.listdir("G:\samples\change_light"))
    # assert changed_num == 7


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
        obj_class_list=["2"],  # 这里有个不符合常理的小问题。
        enlarge_scale=5,
        external_sample_dir="G:\samples\imshot",
    )
    """
    用imshot的时候跟obj_class_list无关, 设置了也没用。后面再改文件夹和挂钩。
    """
    img_cut.run(save_path="G:\samples\img_cut", bound_to_obj=10)
