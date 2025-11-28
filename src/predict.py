import torch
import torch.nn.functional as F
from dataset import get_dataloaders
from model import SimpleCNN
import random
import os
import matplotlib.pyplot as plt 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_best_model(model_dir='./saved_models'):
    
    files = [f for f in os.listdir(model_dir) if f.endswith('.pth')]
    if not files:
        raise FileNotFoundError("没有找到模型文件，请先运行 train.py")
    
   
    best_file = sorted(files)[-1]
    print(f"Loading model: {best_file}")
    return os.path.join(model_dir, best_file)

def main():
    
    _, test_loader = get_dataloaders(batch_size=1)
    
    model = SimpleCNN().to(DEVICE)
    model_path = load_best_model()
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    dataset = test_loader.dataset
    
    print("正在抽取随机图片进行测试，请查看弹出的窗口...")
    
    for i in range(3):
        idx = random.randint(0, len(dataset)-1)
        image, label = dataset[idx] 
        
        input_tensor = image.unsqueeze(0).to(DEVICE) 
        
        with torch.no_grad():
            output = model(input_tensor)
            probs = F.softmax(output, dim=1)
            pred = output.argmax(dim=1).item()
            confidence = probs[0][pred].item() * 100

        print(f"[{i+1}/3] 真实标签: {label} | AI预测: {pred} (信心: {confidence:.2f}%)")
        
        img_numpy = image.squeeze().numpy()
        
        plt.figure(f"测试案例 {i+1}") 
        plt.imshow(img_numpy, cmap='gray') 
        plt.title(f"True: {label} | Pred: {pred} ({confidence:.1f}%)")
        plt.axis('off')
        
        plt.show() 

if __name__ == "__main__":
    main()