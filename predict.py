# from getModel import get_myTrained_model
import torch
from PIL import Image

from myTransform import train_transform
from myClasses import classesIdKey


def predict(image_path, device, model):
    image = Image.open(image_path)
    image = train_transform(image)  # <class 'torch.Tensor'> torch.Size([3, 256, 256])
    image = image.view(-1, 3, 256, 256)
    image = image.to(device)
    model.eval()  # 评估模式
    outputs = model(image)
    # torch.max(outputs, 1) 返回每行的最大值及其索引
    _, pre_labels = torch.max(outputs, 1)
    pre_labels_id = pre_labels.cpu().numpy()[0]
    pre_labels_name = classesIdKey[pre_labels_id]
    print(pre_labels_id)
    print(pre_labels_name)
    return pre_labels_id, pre_labels_name

# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# model = get_myTrained_model('./model/myModel-0.9.pkl')
# predict('./data/二尾蛱蝶/01372.jpg', model=model, device=device)
