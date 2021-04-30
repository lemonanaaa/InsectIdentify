from getModel import get_myTrained_model
import torch
from PIL import Image

from myTransform import train_transform


def predict(image_path, model=get_myTrained_model('./model/myModel.pkl')):
    image = Image.open(image_path)
    image = train_transform(image)  # <class 'torch.Tensor'> torch.Size([3, 256, 256])
    image = image.view(-1, 3, 256, 256)
    model.eval()  # 评估模式
    outputs = model(image)
    # torch.max(outputs, 1) 返回每行的最大值及其索引
    _, pre_labels = torch.max(outputs, 1)
    pre_labels_id = pre_labels.numpy()[0]
    print(pre_labels_id)
    return pre_labels_id


model = get_myTrained_model('./model/myModel.pkl')
predict('./hymenoptera_data/val/bees/10870992_eebeeb3a12.jpg', model=model)
predict('./hymenoptera_data/val/ants/57264437_a19006872f.jpg', model=model)
