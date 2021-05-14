from torchvision import datasets, transforms, models
import torch
from torch.utils.data import DataLoader, Dataset
import os
import numpy as np
from PIL import Image


# 实现自己的Dataset方法，主要实现两个方法__len__和__getitem__
import myTransform


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


def get_pre_model(fc_out_features: int, only_train_fc=True):
    '''

    :param fc_out_features: 分类树木，即为全连接层的输出单元数
    :param only_train_fc: 是否只训练全连接层
    :return:
    '''
    model = models.resnet152(pretrained=True)  # 使用预训练

    # 先将所有的参数设置为不进行梯度下降
    if only_train_fc:
        for param in model.parameters():
            param.requires_grad_(False)
    # 将全连接层设置为进行梯度下降
    fc_in_features = model.fc.in_features
    model.fc = torch.nn.Linear(fc_in_features, fc_out_features, bias=True)
    return model


def print_buffers(model):
    for buffer in model._buffers:
        print(buffer)
    for i in model.parameters():
        if i.requires_grad:
            print(i)


def train(epochs=10, loss_fn=torch.nn.CrossEntropyLoss(), sgd_lr=0.01,
          train_dirpath='./hymenoptera_data/train', val_dirpath='./hymenoptera_data/val'):
    '''

    :param model: 模型
    :param epochs: 完整的数据集通过了神经网络一次并且返回了一次，这个过程称为一个 epoch。也就是训练几轮。
    :param loss_fn:使用哪种损失函数 默认使用交叉熵
        # output是网络的输出，size=[batch_size, class]
        #如网络的batch size为128，数据分为10类，则size=[128, 10]
        # target是数据的真实标签，是标量，size=[batch_size]
        #如网络的batch size为128，则size=[128]
        crossentropyloss=nn.CrossEntropyLoss()
        crossentropyloss_output=crossentropyloss(output,target)
    :param sgd_lr: 梯度下降的学习率
    :param train_dirpath: 训练数据路径
    :param val_dirpath: 验证数据路径
    :return:
    '''
    # 分布实现训练和预测的transform
    train_transform = myTransform.train_transform
    val_transform = myTransform.val_transform
    # 分别实现loader
    train_dataset = MyDataset(train_dirpath, train_transform)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=64)
    val_dataset = MyDataset(val_dirpath, val_transform)
    val_loader = DataLoader(val_dataset, shuffle=True, batch_size=64)

    model = get_pre_model(fc_out_features=2)
    print()

    # for type_id, loader in enumerate([train_loader, val_loader]):
    #     print(type_id, loader)

    opt = torch.optim.SGD(params = model.parameters(),lr=sgd_lr)  # 优化函数，model.parameters()为该实例中可优化的参数
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    model.to(device)

    # 对学习率进行等间隔地调整，调整倍数为gamma， 调整的epoch间隔为step_size 。
    opt_step = torch.optim.lr_scheduler.StepLR(opt, step_size=2, gamma=0.9)
    max_acc = 0
    epoch_acc = []
    epoch_loss = []
    for epoch in range(epochs):
        # 每个epoch有两个部分，train_loader 和 val_loader
        for type_id, loader in enumerate([train_loader, val_loader]):
            mean_loss = []
            mean_acc = []
            for images, labels in loader:
                print(type(images),images.shape) # <class 'torch.Tensor'> torch.Size([32, 3, 256, 256])
                if type_id == 0:  # 训练集
                    model.train()
                else:
                    model.eval()
                images = images.to(device)
                labels = labels.to(device).long()
                opt.zero_grad()
                with torch.set_grad_enabled(type_id == 0):
                    print(type(images), images.shape)  # <class 'torch.Tensor'> torch.Size([32, 3, 256, 256])
                    outputs = model(images)
                    _, pre_labels = torch.max(outputs, 1)
                    loss = loss_fn(outputs, labels)
                if type_id == 0:
                    loss.backward()
                    opt.step()
                acc = torch.sum(pre_labels == labels) / torch.tensor(labels.shape[0], dtype=torch.float32)
                mean_loss.append(loss.cpu().detach().numpy())
                mean_acc.append(acc.cpu().detach().numpy())
            if type_id == 1:  # 验证集
                epoch_acc.append(np.mean(mean_acc))
                epoch_loss.append(np.mean(mean_loss))
                if max_acc < np.mean(mean_acc):
                    max_acc = np.mean(mean_acc)
            print(type_id, np.mean(mean_loss), np.mean(mean_acc))
        opt_step.step()  # 调整学习率
    print("max_acc", max_acc)
    torch.save(model, './model/myModel.pkl')
    return model
train()

