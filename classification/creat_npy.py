import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def get_train_files(filename):
    # 用于获取归一化组的文件
    # 此处返回的 image_list：一个每张图片的路径+名字的列表
    #           label_list: 图片对应的标签
    class_train = []
    label_train = []
    for root, dirs, files in os.walk(filename):
        for file in files:

            if "OK-4" not in str(root):
                class_train.append(str(root) + '/' + file)
                label_train.append(file[3])
                # 4倍和20倍分开

    # 转换成数组，再分割
    temp = np.array([class_train, label_train])
    temp = temp.transpose()
    np.random.shuffle(temp)
    image_list = list(temp[:, 0])
    label_list = list(temp[:, 1])
    return image_list, label_list


def read_image(img_path):
    try:
        img = Image.open(img_path)

        # np.mean(img)输出为一个值 如：148
        # 值越接近255，则 图片越接近全白，此处选取210作为阈值

        if np.mean(img) > 210:
            data = 'a'
        else:
            img = img.resize([224, 224], Image.ANTIALIAS)
            # print(np.std(img), np.mean(img))
            data = np.array(img)
            # print(img_path)
    except:
        print("error: " + img_path)
        data = 'a'
    return data


def get_batches(image_list, label_list):
    # 转换为np数组
    images = []
    label_lists = []
    for i in range(len(image_list)):
        if read_image(image_list[i]) != 'a':
            images.append(read_image(image_list[i]))
            label_lists.append(label_list[i])

    images = np.array(images)
    label = np.array(label_lists)
    return images, label


def get_data(filepath, name_x, name_y):
    # name_x,name_y 为保存路径
    # x 为数据， y为标签
    try:
        x = np.load('array111.npy')
        # x 对应为数据
        y = np.load('array2.npy')
        # y 对应为标签
    except:
        # filepath = r"E:\BaiduNetdiskDownload\第二批归一化\对照组"
        train_x, train_y = get_train_files(filepath)
        # 此时的train_x,train_y 是列表，里面分别为图片的地址和标签
        x, y = get_batches(train_x, train_y)
        # 保存成npy文件
        np.save(name_x, x)
        np.save(name_y, y)
        print("finished: " + name_x)

    return x, y


get_data("E:\BaiduNetdiskDownload\第二批归一化\对照组//" + tis, "train_data/"+tis + "_x", "train_data/"+tis + "_y")
