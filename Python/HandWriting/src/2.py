from simpleNN import Net
import torch
model=Net() #实例化
model.load_state_dict(torch.load('simpleNN_Model.pth'))
model.eval()