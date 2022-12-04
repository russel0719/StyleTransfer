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

    dataset = datasets.ImageFolder(path + "COCO", transform)
    train_len = int(len(dataset) * 0.8)
    val_len = len(dataset) - train_len
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_len, val_len])
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)
    
    return train_dataset, train_dataloader, val_dataset, val_dataloader