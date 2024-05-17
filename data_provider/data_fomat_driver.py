import os
import cv2


class DataConvert:
    def __init__(self, fmt):
        self.load_api = {
            "yolo2bbox": self.load_yolo2bbox,
            "yolo2xyxy": self.load_yolo2xyxy,
        }
        self.convert2yolo_api = {
            "yolo2bbox": self.bbox2yolo,
            "yolo2xyxy": self.xyxy2yolo,
        }
        self.save_api = {
            "yolo2bbox": self.save_yolo,
            "yolo2xyxy": self.save_yolo,
        }
        self.load = self.load_api[fmt]
        self.x2yolo = self.convert2yolo_api[fmt]
        self.save = self.save_api[fmt]

    @staticmethod
    def load_yolo2bbox(path, fc, fr):
        labels = []
        with open(path) as f:
            lines = f.readlines()
            for line in lines:
                obj_class = int(float(line.split()[0]))
                obj_x = int(float(line.split()[1]) * fc)
                obj_y = int(float(line.split()[2]) * fr)
                obj_w = int(float(line.split()[3]) * fc)
                obj_h = int(float(line.split()[4]) * fr)
                labels.append([obj_class, obj_x, obj_y, obj_w, obj_h])
        return labels

    @staticmethod
    def load_yolo2xyxy(path, fc, fr):
        labels = []
        with open(path) as f:
            lines = f.readlines()
            for line in lines:
                obj_class = int(float(line.split()[0]))
                obj_x = float(line.split()[1]) * fc
                obj_y = float(line.split()[2]) * fr
                obj_w = float(line.split()[3]) * fc
                obj_h = float(line.split()[4]) * fr
                x1 = int(obj_x - obj_w / 2)
                y1 = int(obj_y - obj_h / 2)
                x2 = int(obj_x + obj_w / 2)
                y2 = int(obj_y + obj_h / 2)
                labels.append([obj_class, x1, y1, x2, y2])
        return labels

    @staticmethod
    def xyxy2yolo(labels, img_w, img_h):
        for label in labels:
            w = label[3] - label[1]
            h = label[4] - label[2]
            cx = (label[1] + label[3]) / 2
            cy = (label[2] + label[4]) / 2
            label[1] = cx / img_w
            label[2] = cy / img_h
            label[3] = w / img_w
            label[4] = h / img_h
        return labels

    @staticmethod
    def bbox2yolo(labels, img_w, img_h):
        for label in labels:
            label[1] = label[1] / img_w
            label[2] = label[2] / img_h
            label[3] = label[3] / img_w
            label[4] = label[4] / img_h
        return labels

    @staticmethod
    def bbox2yolo(labels, img_w, img_h):
        for label in labels:
            label[1] = (label[1] + label[3]) / 2 / img_w
            label[2] = (label[2] + label[4]) / 2 / img_h
            label[3] = (label[3] - label[1]) / img_w
            label[4] = (label[4] - label[2]) / img_h
        return labels

    @staticmethod
    def save_yolo(path, save_name, img, labels):
        if not os.path.exists(os.path.join(path, "labels")):
            os.makedirs(os.path.join(path, "labels"))
        if not os.path.exists(os.path.join(path, "images")):
            os.makedirs(os.path.join(path, "images"))
        new_label_path = os.path.join(path, "labels", "%s.txt" % save_name)
        new_img_path = os.path.join(path, "images", "%s.jpg" % save_name)
        cv2.imwrite(new_img_path, img)
        with open(new_label_path, "w") as f:
            for label in labels:
                f.write(
                    "{} {:.6f} {:.6f} {:.6f} {:.6f}\n".format(
                        label[0], label[1], label[2], label[3], label[4]
                    )
                )

    @staticmethod
    def load_yolo(path):
        labels = []
        if os.path.exists(path):
            with open(path) as f:
                lines = f.readlines()
                for line in lines:
                    obj_class = int(float(line.split()[0]))
                    obj_x = float(line.split()[1])
                    obj_y = float(line.split()[2])
                    obj_w = float(line.split()[3])
                    obj_h = float(line.split()[4])
                    labels.append([obj_class, obj_x, obj_y, obj_w, obj_h])
        return labels

    @staticmethod
    def replace_cls(labels, ori_cls, new_cls):
        for i in range(len(labels)):
            if labels[i][0] == ori_cls:
                labels[i][0] = new_cls
        return labels

    @staticmethod
    def freash_yolo_labels(path, labels):
        with open(path, "w") as f:
            for label in labels:
                f.write(
                    "{} {} {} {} {}\n".format(
                        label[0], label[1], label[2], label[3], label[4]
                    )
                )

    @classmethod
    def change_label_cls(self, labels_path, ori_cls, new_cls, **kwargs):
        label_list = os.listdir(labels_path)
        for label in label_list:
            label_path = os.path.join(labels_path, label)
            labels = self.load_yolo(label_path)
            labels = self.replace_cls(labels, ori_cls, new_cls)
            self.freash_yolo_labels(label_path, labels)
        print("done")

    @staticmethod
    def del_cls(labels, del_cls):
        new_labels = []
        for i in range(len(labels)):
            if labels[i][0] in del_cls:
                continue
            new_labels.append(labels[i])
        return new_labels

    @classmethod
    def del_aim_cls(self, labels_path, del_cls, **kwargs):
        label_list = os.listdir(labels_path)
        for label in label_list:
            label_path = os.path.join(labels_path, label)
            labels = self.load_yolo(label_path)
            labels = self.del_cls(labels, del_cls)
            self.freash_yolo_labels(label_path, labels)
        print("done")
