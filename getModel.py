from torchvision import models
import torch


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


def get_myTrained_model(modelPath):
    model = torch.load(modelPath)
    return model


def print_buffers(model):
    for buffer in model._buffers:
        print(buffer)
    for i in model.parameters():
        if i.requires_grad:
            print(i)
