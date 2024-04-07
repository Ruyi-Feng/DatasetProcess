import cv2
import json
import numpy as np
import os
import random
import shutil


def click_figure(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))


def select_error_samples(imori):
    global points
    points = []
    cv2.namedWindow("select_error_figures", cv2.WINDOW_AUTOSIZE)
    while 1:
        cv2.setMouseCallback("select_error_figures", click_figure)
        Key = cv2.waitKey(10)
        cv2.imshow("select_error_figures", imori)
        if Key == 13:
            break
    cv2.destroyAllWindows()
    return points


def save_list(list, path):
    with open(path, "w") as f:
        for item in list:
            f.write("%s\n" % item)


def load_json(path):
    if not os.path.exists(path):
        print("[Warning] No such json file and create one!")
        save_json({}, path)
    with open(path, "r") as f:
        return json.load(f)


def save_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f)

def save_img(img, path):
    cv2.imwrite(path, img)

def _del_samples(file_dir, file_name):
    if os.path.exists(os.path.join(file_dir, file_name)):
        os.remove(os.path.join(file_dir, file_name))
        print("delete " + file_name)


def del_error_samples(checker_log_path):
    checker_log = load_json(checker_log_path)
    dataset_dir = checker_log["check_dir"]
    img_dir = os.path.join(dataset_dir, "images")
    label_dir = os.path.join(dataset_dir, "labels")
    for name in checker_log["error_label"]:
        _del_samples(label_dir, name)
        img_nm = labels2img(os.path.join(img_dir, name))
        _del_samples("", img_nm)


def sampling_refer_balenced_log(
    balencer_log_path,
    dataset_dir,
    save_dir,
    img_exts=[".jpg", ".png"],
    label_ext=".txt",
):
    balencer_log = load_json(balencer_log_path)
    img_dir = os.path.join(dataset_dir, "images")
    label_dir = os.path.join(dataset_dir, "labels")
    if not os.path.exists(os.path.join(save_dir, "images")):
        os.makedirs(os.path.join(save_dir, "images"))
    save_dir = os.path.join(save_dir, "labels")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for k, lists in balencer_log.items():
        for name, times in lists.items():
            for i in range(times):
                # 这里仍然有bug，重复采样没有达成。
                if os.path.exists(os.path.join(label_dir, name)):
                    os.replace(
                        os.path.join(label_dir, name), os.path.join(save_dir, str(i) + name)
                    )
                for ext in img_exts:
                    img_nm = name.replace(label_ext, ext)
                    if os.path.exists(os.path.join(img_dir, img_nm)):
                        os.replace(
                            os.path.join(img_dir, img_nm),
                            labels2img(os.path.join(save_dir, str(i) + img_nm)),
                        )

def img2labels(img_path, label_ext=".txt", img_exts=[".jpg", ".png"]):
    label_path = img_path.replace("images", "labels")
    label_path = label_path[:-4] + ".txt"
    return label_path

def labels2img(label_path, label_ext=".txt", img_exts=[".jpg", ".png"]):
    img_path = label_path.replace("labels", "images")
    for ext in img_exts:
        tmp = label_path[:-4] + ext
        if os.path.exists(tmp):
            img_path = tmp
            break
    return img_path


def save_top_pool(top_pool, path):
    for k, v in top_pool.items():
        for label in v:
            v[label] = list(v[label])
    save_json(top_pool, path)


def load_top_pool(path):
    top_pool_num = {}
    top_pool = load_json(path)
    for k, v in top_pool.items():
        k_num = int(k)
        for label in v:
            for i in range(len(v[label])):
                v[label][i] = float(v[label][i])
            v[label] = np.array(v[label])
        top_pool_num[k_num] = v
    return top_pool_num


def del_files_ext(paths):
    names = []
    for i in paths:
        i = i.replace(".jpg", "")
        i = i.replace(".png", "")
        i = i.replace(".txt", "")
        names.append(i)
    return names


def del_extra_file(path, del_names, del_ext=[".jpg", ".png", ".txt"]):
    for name in del_names:
        for e in del_ext:
            if os.path.exists(os.path.join(path, name + e)):
                _del_samples(path, name + e)
                break


def equal_img_labels(dir_path):
    """for del samples without labels or images

    Args:
        dir_path (str): The path of dataset including sub dirs of images and labels
    """
    img_path = os.path.join(dir_path, "images")
    img_list = os.listdir(img_path)
    label_path = os.path.join(dir_path, "labels")
    label_list = os.listdir(label_path)
    if len(img_list) != len(label_list):
        im_name = del_files_ext(img_list)
        lb_name = del_files_ext(label_list)
        names = set(im_name) & set(lb_name)
        print("交集样本数量:", len(names))
        del_extra_file(img_path, set(im_name) - names)
        del_extra_file(label_path, set(lb_name) - names)

def _copy(ori_file, cur_file):
    if os.path.exists(ori_file):
        shutil.copy(ori_file, cur_file)

def copy_to_dir(ori, new, name_list, label_copy, img_copy):
    for name in name_list:
        if img_copy:
            orifile = os.path.join(ori, "images", name)
            newfile = os.path.join(new, "images", name)
            _copy(orifile, newfile)
        if label_copy:
            orifile = img2labels(os.path.join(ori, "images", name))
            newfile = img2labels(os.path.join(new, "images", name))
            _copy(orifile, newfile)

def _split(ori_path, new_path, train_list, val_list, test_list=[]):
    if not os.path.exists(os.path.join(new_path, "train")):
        os.makedirs(os.path.join(new_path, "train"))
        os.makedirs(os.path.join(new_path, "train", "images"))
        os.makedirs(os.path.join(new_path, "train", "labels"))
    if not os.path.exists(os.path.join(new_path, "val")):
        os.makedirs(os.path.join(new_path, "val"))
        os.makedirs(os.path.join(new_path, "val", "images"))
        os.makedirs(os.path.join(new_path, "val", "labels"))
    if not os.path.exists(os.path.join(new_path, "test")) and len(test_list):
        os.makedirs(os.path.join(new_path, "test"))
        os.makedirs(os.path.join(new_path, "test", "images"))
        os.makedirs(os.path.join(new_path, "test", "labels"))
    new_train = os.path.join(new_path, "train")
    new_val = os.path.join(new_path, "val")
    copy_to_dir(ori_path, new_train, train_list, True, True)
    copy_to_dir(ori_path, new_val, val_list, True, True)
    if len(test_list):
        copy_to_dir(ori_path, new_val, test_list, True, True)
    print("done split dataset")

def train_val_split(ori_path, new_path, train_ratio=0.7, val_ratio=0.3, test_ratio=0.0):
    ori_img_path = os.path.join(ori_path, "images")
    if not os.path.exists(ori_img_path):
        raise ValueError("not exist path")
    label_list = os.listdir(ori_img_path)
    train_sample = int(len(label_list) * train_ratio)
    train_list = random.sample(label_list, k=train_sample)
    extra_list = list(set(label_list) - set(train_list))
    if test_ratio == 0:
        val_list = extra_list
    else:
        val_sample = int(len(label_list) * val_ratio)
        val_list = random.sample(extra_list, val_sample)
        test_list = list(set(extra_list) - set(val_list))
    if not os.path.exists(new_path):
        os.makedirs(new_path)
    _split(ori_path, new_path, train_list, val_list, test_list)
