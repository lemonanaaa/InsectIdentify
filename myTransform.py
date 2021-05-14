from torchvision import transforms

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
