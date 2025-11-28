import torch 
from torchvision import datasets,transforms
from torch.utils.data import DataLoader

def get_dataloaders(batch_size=64):
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,),(0.3081,))
    ])

    train_dataset=datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )

    test_dataset=datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )

    train_loader=DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)
    test_loader=DataLoader(dataset=test_dataset,batch_size=1000,shuffle=False)

    return train_loader,test_loader
if __name__=="__main__":
    train_loader, _=get_dataloaders()
    images,labels=next(iter(train_loader))
    print(f"图片批次形状：{images.shape}")
    print(f"标签批次形状：{labels.shape}")