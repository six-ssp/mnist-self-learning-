import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import sys

from dataset import get_dataloaders
from model import SimpleCNN

DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS=5
LR=0.01

def train(model,device,train_loader,optimizer,epoch):
    model.train()
    pbar=tqdm(train_loader,desc=f"Epoch {epoch}",file=sys.stdout)
    for batch_idx,(data,target) in enumerate(pbar):
        data,target=data.to(device),target.to(device)
        optimizer.zero_grad()
        output=model(data)
        loss=F.cross_entropy(output,target)
        loss.backward()
        optimizer.step()
        pbar.set_postfix({"Loss":f"{loss.item():.4}"})
def test(model,device,test_loader):
    model.eval()
    test_loss=0
    correct=0
    with torch.no_grad():
        for data,target in test_loader:
            data,target=data.to(device),target.to(device)
            output=model(data)
            test_loss+=F.cross_entropy(output,target,reduction='sum').item()
            pred=output.argmax(dim=1,keepdim=True)
            correct+=pred.eq(target.view_as(pred)).sum().item()
    test_loss/=len(test_loader.dataset)
    accuracy=100.*correct/len(test_loader.dataset)
    print(f'\n测试结果：平衡损失：{test_loss:.4f},准确率：{correct}/{len(test_loader.dataset)}({accuracy:.2f}%)\n')
    return accuracy
def main():
    try:
        import tqdm
    except ImportError:
        print(">>> 警告：未找到 tqdm 库。请安装：pip install tqdm")
        return
    train_loader,test_loader=get_dataloaders()

    print(f"当前训练设备: {DEVICE}")
    model=SimpleCNN().to(DEVICE)
    print("模型初始化完成，准备训练。")

    optimizer=optim.Adam(model.parameters(),lr=LR)

    best_accuracy=0.0
    for epoch in range(1,EPOCHS+1):
        train(model,DEVICE,train_loader,optimizer,epoch)
        accuracy=test(model,DEVICE,test_loader)
        if accuracy>best_accuracy:
            best_accuracy=accuracy
            save_path = f'./saved_models/mnist_cnn_{epoch}_{accuracy:.2f}.pth'
            torch.save(model.state_dict(),save_path)
            print(f"模型准确率提高，已保存到: {save_path}")
if __name__=='__main__':
    main()