# import random
# from torch.utils.data import DataLoader
# import os
#
# from myTransform import train_transform, val_transform
# from MyDataset import MyDataset
#
#
# def getDataLoader(dirname, percentage=0.9, shuffle=True, batch_size=32):
#     classes = os.listdir(dirname)
#     images = []
#     for i, label in enumerate(classes):
#         classes_path = dirname + '\\' + label
#         for image_name in os.listdir(classes_path):
#             images.append((classes_path + '\\' + image_name, i))
#     random.shuffle(images)
#     boundary = int(len(images) * percentage)
#     trainList = images[:boundary]
#     train_dataset = MyDataset(classes=classes, images=trainList, transform=train_transform)
#     train_loader = DataLoader(train_dataset, shuffle=shuffle, batch_size=batch_size)
#     valList = images[boundary:]
#     val_dataset = MyDataset(classes=classes, images=valList, transform=val_transform)
#     val_loader = DataLoader(val_dataset, shuffle=shuffle, batch_size=batch_size)
#     return train_loader, val_loader, classes

import random
from torch.utils.data import DataLoader
import os

from myTransform import train_transform, val_transform
from MyDataset import MyDataset


def getDataLoader(dirname, percentage=0.8, shuffle=True, batch_size=32):
    classes = os.listdir(dirname)
    trainList = []
    valList = []
    for i, label in enumerate(classes):
        classes_path = dirname + '\\' + label
        temp = []
        for image_name in os.listdir(classes_path):
            temp.append((classes_path + '\\' + image_name, i))
        boundary = int(len(temp) * percentage)
        random.shuffle(temp)
        for value in temp[:boundary]:
            # print(value)
            trainList.append(value)
        for value in temp[boundary:]:
            valList.append(value)
    # random.shuffle(images)
    # boundary = int(len(images) * percentage)
    # trainList = images[:boundary]
    train_dataset = MyDataset(classes=classes, images=trainList, transform=train_transform)
    train_loader = DataLoader(train_dataset, shuffle=shuffle, batch_size=batch_size)
    # valList = images[boundary:]
    val_dataset = MyDataset(classes=classes, images=valList, transform=val_transform)
    val_loader = DataLoader(val_dataset, shuffle=shuffle, batch_size=batch_size)
    return train_loader, val_loader, classes
