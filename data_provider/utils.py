import cv2
import json
import numpy as np
import os


def click_figure(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))


def select_error_samples(imori):
    global points
    points = []
    cv2.namedWindow("select_error_figures", cv2.WINDOW_NORMAL)
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
    for name in checker_log["error_img"]:
        _del_samples(img_dir, name)


def sampling_refer_balenced_log(
    balencer_log_path, save_dir, img_ext=".jpg", label_ext=".txt"
):
    balencer_log = load_json(balencer_log_path)
    dataset_dir = balencer_log["check_dir"]
    img_dir = os.path.join(dataset_dir, "images")
    label_dir = os.path.join(dataset_dir, "labels")
    if not os.path.exists(os.path.join(save_dir, "images")):
        os.makedirs(os.path.join(save_dir, "images"))
    save_dir = os.path.join(save_dir, "labels")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for k, lists in balencer_log:
        for name in lists:
            if os.path.exists(os.path.join(label_dir, name)):
                os.replace(os.path.join(label_dir, name), os.path.join(save_dir, name))
            img_nm = name.replace(label_ext, img_ext)
            if os.path.exists(os.path.join(img_dir, img_nm)):
                os.replace(
                    os.path.join(img_dir, img_nm), os.path.join(save_dir, img_nm)
                )


def img2labels(img_path, img_ext=".jpg", label_ext=".txt"):
    label_path = img_path.replace(img_ext, label_ext)
    label_path = label_path.replace("images", "labels")
    return label_path


def labels2img(label_path, img_ext=".jpg", label_ext=".txt"):
    img_path = label_path.replace(label_ext, img_ext)
    img_path = img_path.replace("labels", "images")
    return img_path


def save_top_pool(top_pool, path):
    for k, v in top_pool.items():
        for label in v:
            v[label] = list(v[label])
    save_json(top_pool, path)


def load_top_pool(path):
    top_pool = load_json(path)
    for k, v in top_pool.items():
        for label in v:
            for i in range(len(v[label])):
                v[label][i] = float(v[label][i])
            v[label] = np.array(v[label])
    return top_pool

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
        ext = ""
        for e in del_ext:
            if os.path.exists(os.path.join(path, name + e)):
                ext = e
                _del_samples(path, name + ext)
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

