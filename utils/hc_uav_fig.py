# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 15:30:22 2021

@author: Polaris
"""
import cv2
import os

global xcut, ycut
xcut = 3
ycut = 3


def poscut(bbox, poslist, transpath, labelnm):
    posbox = []
    x = bbox[1]
    y = bbox[2]
    w = bbox[3]
    h = bbox[4]

    for pos in poslist:
        i = pos[0]
        j = pos[1]
        pos = pos[2:7]
        if int(x) in range(pos[0], pos[1]):
            if int(y) in range(pos[2], pos[3]):
                x = x - pos[0]
                y = y - pos[2]
                x = x / (pos[1] - pos[0])
                y = y / (pos[3] - pos[2])
                w = w / (pos[1] - pos[0])
                h = h / (pos[3] - pos[2])
                posbox = (
                    str(int(bbox[0]))
                    + " "
                    + str(x)
                    + " "
                    + str(y)
                    + " "
                    + str(w)
                    + " "
                    + str(h)
                    + "\n"
                )
                labelnew = os.path.join(
                    transpath, "0568_" + str(int(i)) + "_" + str(int(j)) + "_" + labelnm
                )  # ----------修改名称1
                with open(labelnew, "a+") as f:
                    f.writelines(posbox)
    return posbox


def CropImage4File(oripath, newpath):
    global xcut, ycut
    # ------------------------------------------------------------------------------------------------------------修改名称2
    filepath = os.path.join(oripath, "images")
    destpath = os.path.join(newpath, "images", "train")
    labelpath = os.path.join(oripath, "labels")
    transpath = os.path.join(newpath, "labels", "train")

    pathDir = os.listdir(filepath)  # 列出文件路径中的所有路径或文件
    """ """
    for allDir in pathDir:
        poslist = []
        imori = os.path.join(filepath, allDir)  # 打开了图像
        labelnm = allDir.replace("jpg", "txt")
        labelnm = labelnm.replace("png", "txt")
        labelori = os.path.join(labelpath, labelnm)  # 打开了图像同名的txt文件

        if os.path.isfile(imori):
            image = cv2.imread(imori)
            sp = image.shape  # 获取图像形状：返回【行数值，列数值】列表
            sz1 = sp[0]  # 图像的高度（行 范围）
            sz2 = sp[1]  # 图像的宽度（列 范围）
            # sz3 = sp[2]                #像素值由【RGB】三原色组成

            width = sz2 / xcut
            height = sz1 / ycut

            # 开始循环滑窗切割图像
            for i in range(xcut):
                for j in range(ycut):
                    x1 = i * width - 0.2 * width
                    y1 = j * height - 0.2 * height
                    x2 = i * width + width
                    y2 = j * height + height
                    if x1 >= x2:
                        continue
                    if y1 >= y2:
                        continue
                    if x1 < 1:
                        x1 = 1
                    if y1 < 1:
                        y1 = 1
                    if y2 >= sz1:
                        y2 = sz1 - 1
                    if x2 >= sz2:
                        x2 = sz2 - 1
                    cropImg = image[int(y1) : int(y2), int(x1) : int(x2)]
                    poslist.append([i, j, int(x1), int(x2), int(y1), int(y2)])
                    dest = os.path.join(
                        destpath, "0568_" + str(i) + "_" + str(j) + "_" + allDir
                    )  # ----------修改名称3
                    cv2.imwrite(dest, cropImg)

            if os.path.isfile(labelori):
                f = open(labelori)
                for line in f:
                    bbox = [float(i) for i in line.split()]
                    bbox[1] = bbox[1] * sz2
                    bbox[3] = bbox[3] * sz2
                    bbox[2] = bbox[2] * sz1
                    bbox[4] = bbox[4] * sz1
                    labelpos = poscut(bbox, poslist, transpath, labelnm)
                f.close()


if __name__ == "__main__":
    """
    图像的储存格式：example
    -UAV
    --ori
    ---image
    ---label

    --new
    ---image
    ---label
    """

    filepath = (
        "E://Traindata//UAV_0568"  # 源图像                     #-----------------修改名称4
    )
    destpath = "E://Traindata//UAV"  # resized images saved here
    # 这个函数是用于滑窗切割
    CropImage4File(filepath, destpath)
    """
    #以下部分是可视化框子和图片，检查是否对准使用
    testpath="D://PROGRAM//yolov5traex//UAV//new//labels"
    impa="D://PROGRAM//yolov5traex//UAV//new//images"

    ld =  os.listdir(testpath)    # 列出文件路径中的所有路径或文件
    for alltxt in ld:
        imnm=alltxt.replace("txt","png")
        imtest = os.path.join(impa,imnm)
        txttest=os.path.join(testpath,alltxt)
        img=cv2.imread(imtest)
        szx=img.shape[1]
        szy=img.shape[0]

        f = open(txttest)
        for line in f:
            bbox=[float(i) for i in line.split()]
            x1=bbox[1]-bbox[3]/2
            x2=bbox[1]+bbox[3]/2
            y1=bbox[2]-bbox[4]/2
            y2=bbox[2]+bbox[4]/2
            x1=x1*szx
            x2=x2*szx
            y1=y1*szy
            y2=y2*szy
            cv2.rectangle(img,(int(x1),int(y1)),(int(x2),int(y2)),(100,10,100),3)

        cv2.imshow("show",img)
        cv2.waitKey(0)
        """
