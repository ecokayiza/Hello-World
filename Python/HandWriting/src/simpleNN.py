import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt

BATCH_SIZE=16
EPOCH=4

device=torch.device( "cuda:0" if torch.cuda.is_available() else "cpu")

class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = torch.nn.Linear(28*28,64) # 输入28*28像素图像，中间层都为64个节点全连接层
        self.layer2 = torch.nn.Linear(64,64)
        self.layer3 = torch.nn.Linear(64,64)
        self.layer4 = torch.nn.Linear(64,10)    #输出10个数字类别
        
    def forward(self,x):  #向前传播
        x = x.to(device)
        x = torch.nn.functional.relu(self.layer1(x))  
        x = torch.nn.functional.relu(self.layer2(x))  #套上ReLU激活函数
        x = torch.nn.functional.relu(self.layer3(x))
        x = torch.nn.functional.log_softmax(self.layer4(x),dim=1) #用softmax归一化 log对其取对数提高稳定性
        return x
    
def get_data_loader(is_train):
    to_tensor = transforms.Compose([transforms.ToTensor()]) #转换为张量
    data_set = MNIST(root='./data/',train=is_train,transform=to_tensor,download=True)
    return DataLoader(data_set,batch_size=BATCH_SIZE,shuffle=True,pin_memory=True)

def evaluate(test_data,net):
    confusion_matrix=[[0 for _ in range(10)] for _ in range(10)]
    with torch.no_grad():
        for (x, y) in test_data:
            outputs = net.forward(x.view(-1,28*28)) #导出预测值
            for i, output in enumerate(outputs): #第i个数据
                confusion_matrix[y[i]][torch.argmax(output)]+=1 #混淆矩阵
    return confusion_matrix
    
def show(confusion_matrix):   
    explain = [i for i in range(10)]
    plt.imshow(confusion_matrix, cmap='Blues')
    plt.xticks(range(10), explain)
    plt.yticks(range(10), explain)
    plt.xlabel('Predict')
    plt.ylabel('True')
    plt.colorbar()
    # 在每个格子上标注数字
    for i in range(10):
        for j in range(10):
            plt.text(x=j, y=i, s=str(confusion_matrix[i][j]), ha='center', va='center')
    plt.show()
    
def main():
    
    print(f"use {device}")
    train_data = get_data_loader(is_train=True)   #导入训练集和测试集
    test_data = get_data_loader(is_train=False)
    net = Net() #初始化神经网络
    net =net.to(device)   
    optimizer = torch.optim.Adam(net.parameters(),lr=0.001)  #使用Adam优化器
    for epoch in range(EPOCH):
        for (x,y) in train_data:
            net.zero_grad()
            output = net.forward(x.view(-1,28*28))
            output = output.to(device)
            y = y.to(device)
            loss = torch.nn.functional.nll_loss(output,y)
            loss.backward()
            optimizer.step()
        result = evaluate(test_data,net)
        accuracy = sum([result[i][i] for i in range(10)])/sum([sum(result[i]) for i in range(10)])
        print("epoch",epoch+1,"accuracy",accuracy)

    show(result)
    
    torch.save(net.state_dict(),'simpleNN_Model.pth') #保存模型


if __name__=='__main__':
    main()          
              

            