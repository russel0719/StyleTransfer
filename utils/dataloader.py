import torch
from torchvision import datasets
import torchvision.transforms as transforms

def load_data(path, batch_size):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])

    train_dataset = datasets.ImageFolder(path + "COCO", transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
    
    return train_dataset, train_loader