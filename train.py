from torchvision import datasets, transforms, models
import torch
import numpy as np
import copy

from DataReader import getDataLoader
from getModel import get_pre_model


def train(dataPath, epochs=10, loss_fn=torch.nn.CrossEntropyLoss(), sgd_lr=0.01):
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
    :return:
    '''
    # 获取loader
    train_loader, val_loader, classes = getDataLoader(dataPath)
    # 获取model
    model = get_pre_model(len(classes))
    model = models.resnet18(pretrained=True)  # 使用预训练

    # 优化函数(用于梯度下降)     model.parameters()为该实例中可优化的参数
    opt = torch.optim.SGD(lr=sgd_lr, params=model.parameters())
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # 对学习率进行等间隔地调整，调整倍数为gamma， 调整的epoch间隔为step_size
    opt_step = torch.optim.lr_scheduler.StepLR(opt, step_size=5, gamma=0.5)
    max_acc = 0
    epoch_acc = []
    epoch_loss = []

    fcWithMaxAcc = copy.copy(model.fc)
    for epoch in range(epochs):
        # 每个epoch有两个部分，train_loader 和 val_loader
        for type_id, loader in enumerate([train_loader, val_loader]):
            mean_loss = []
            mean_acc = []
            for images, labels in loader:
                if type_id == 0:  # 训练集
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
                    opt.step()  # 调整学习率
                acc = torch.sum(pre_labels == labels) / torch.tensor(labels.shape[0], dtype=torch.float32)
                mean_loss.append(loss.cpu().detach().numpy())
                mean_acc.append(acc.cpu().detach().numpy())
            if type_id == 1:  # 验证集
                epoch_acc.append(np.mean(mean_acc))
                epoch_loss.append(np.mean(mean_loss))
                if max_acc < np.mean(mean_acc):
                    max_acc = np.mean(mean_acc)
                    fcWithMaxAcc = copy.copy(model.fc)
            print(type_id, np.mean(mean_loss), np.mean(mean_acc))

    print(max_acc)
    model.fc = copy.copy(fcWithMaxAcc)
    torch.save(model, './model/myModel'+max_acc+'.pkl')
    return model


train(dataPath="")
