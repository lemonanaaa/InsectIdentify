from torch.utils.data import Dataset
from PIL import Image


# 实现自己的Dataset方法，主要实现两个方法__len__和__getitem__
class MyDataset(Dataset):
    def __init__(self, classes=[], images=[], transform=None):
        super(MyDataset, self).__init__()
        self.classes = classes
        self.images = images
        self.transform = transform

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
