from random import random
from torchvision import transforms
from torch.utils.data import DataLoader
import os
from MyDataset import MyDataset

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


def getDataLoader(dirname, percentage=0.9):
    classes = os.listdir(dirname)
    images = []
    for i, classes in enumerate(classes):
        classes_path = dirname + '/' + classes
        for image_name in os.listdir(classes_path):
            images.append((classes_path + '/' + image_name, i))
    random.shuffle(images)
    boundary = int(len(images) * percentage)
    trainList = images[:boundary]
    train_dataset = MyDataset(classes=classes,images=trainList,transforms=train_transform)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=32)
    valList = images[boundary:]
    val_dataset = MyDataset(classes=classes,images=valList,transforms=val_transform)
    val_loader = DataLoader(val_dataset, shuffle=True, batch_size=32)
    return train_loader,val_loader,classes
