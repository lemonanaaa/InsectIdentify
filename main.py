# -*- coding: utf-8 -*-
# @File    : test4.py
# @Blog    : https://blog.csdn.net/caomin1hao

from torchvision import datasets, transforms, models
import torch
from torch.utils.data import DataLoader, Dataset
import os
# import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import copy


# unzip('./data/hymenoptera_data.zip', './data/')

# 实现自己的Dataset方法，主要实现两个方法__len__和__getitem__
class MyDataset(Dataset):
    def __init__(self, dirname, transform=None):
        super(MyDataset, self).__init__()
        self.classes = os.listdir(dirname)
        self.images = []
        self.transform = transform

        for i, classes in enumerate(self.classes):
            classes_path = dirname + '/' + classes
            for image_name in os.listdir(classes_path):
                self.images.append((classes_path + '/' + image_name, i))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_name, classes = self.images[idx]
        image = Image.open(image_name)
        if self.transform:
            image = self.transform(image)
        return image, classes

    def get_claesses(self):
        return self.classes


def get_pre_model(only_train_fc=True):
    model = models.resnet18(pretrained=True)  # 使用预训练
    if only_train_fc:
        for param in model.parameters():
            param.requires_grad_(False)
    fc_in_features = model.fc.in_features
    model.fc = torch.nn.Linear(fc_in_features, 2, bias=True)
    # print(model.parameters())
    return model


def print_buffers(model):
    for buffer in model._buffers:
        print(buffer)
    for i in model.parameters():
        if i.requires_grad:
            print(i)


def train(model, epochs=50, loss_fn=torch.nn.CrossEntropyLoss(), sgd_lr=0.01,
          train_dirpath='./hymenoptera_data/train', val_dirpath='./hymenoptera_data/val'):
    # 分布实现训练和预测的transform
    train_transform = transforms.Compose([
        transforms.Grayscale(3),
        transforms.RandomResizedCrop(224),  # 随机裁剪一个area然后再resize
        transforms.RandomHorizontalFlip(),  # 随机水平翻转
        transforms.Resize(size=(256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Grayscale(3),
        transforms.Resize(size=(256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    # 分别实现loader
    train_dataset = MyDataset(train_dirpath, train_transform)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=32)
    val_dataset = MyDataset(val_dirpath, val_transform)
    val_loader = DataLoader(val_dataset, shuffle=True, batch_size=32)

    for type_id, loader in enumerate([train_loader, val_loader]):
        print(type_id,loader)

    opt = torch.optim.SGD(sgd_lr, params=model.parameters())
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    model.to(device)
    opt_step = torch.optim.lr_scheduler.StepLR(opt, step_size=20, gamma=0.1)
    max_acc = 0
    epoch_acc = []
    epoch_loss = []
    for epoch in range(epochs):
        for type_id, loader in enumerate([train_loader, val_loader]):
            mean_loss = []
            mean_acc = []
            for images, labels in loader:
                if type_id == 0:#训练集
                    # opt_step.step()
                    model.train()
                else:
                    model.eval()
                images = images.to(device)
                labels = labels.to(device).long()
                opt.zero_grad()
                with torch.set_grad_enabled(type_id == 0):
                    outputs = model(images)
                    _, pre_labels = torch.max(outputs, 1)
                    loss = loss_fn(outputs, labels)
                if type_id == 0:
                    loss.backward()
                    opt.step()
                acc = torch.sum(pre_labels == labels) / torch.tensor(labels.shape[0], dtype=torch.float32)
                mean_loss.append(loss.cpu().detach().numpy())
                mean_acc.append(acc.cpu().detach().numpy())
            if type_id == 1:#验证集
                epoch_acc.append(np.mean(mean_acc))
                epoch_loss.append(np.mean(mean_loss))
                if max_acc < np.mean(mean_acc):
                    max_acc = np.mean(mean_acc)
            print(type_id, np.mean(mean_loss), np.mean(mean_acc))

    print(max_acc)
    return model


# # 分布实现训练和预测的transform
# train_transform = transforms.Compose([
#     transforms.Grayscale(3),
#     transforms.RandomResizedCrop(224),  # 随机裁剪一个area然后再resize
#     transforms.RandomHorizontalFlip(),  # 随机水平翻转
#     transforms.Resize(size=(256, 256)),
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# ])
# val_transform = transforms.Compose([
#     transforms.Grayscale(3),
#     transforms.Resize(size=(256, 256)),
#     transforms.CenterCrop(224),
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# ])
#
# model = models.resnet34(pretrained=True)  # 使用预训练
# print(model)
# print(model._buffers)
#
# # 分别实现loader
# train_dirpath = 'G:\\比赛\\软件杯\\林业害虫识别\\phClass\\hymenoptera_data\\train'
# val_dirpath = 'G:\\比赛\\软件杯\\林业害虫识别\\phClass\\hymenoptera_data\\val'
#
# train_dataset = MyDataset(train_dirpath, train_transform)
# train_loader = DataLoader(train_dataset, shuffle=True, batch_size=32)
# val_dataset = MyDataset(val_dirpath, val_transform)
# val_loader = DataLoader(val_dataset, shuffle=True, batch_size=32)
train(get_pre_model())
