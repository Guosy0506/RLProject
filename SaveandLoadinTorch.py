"""
1) torch.save： 将一个序列化的对象保存到磁盘。这个函数使用 Python 的 pickle 工具进行序列化。
                模型 (model)、张量 (tensor) 和各种对象的字典 (dict) 都可以用这个函数保存。
2) torch.load： 将 pickled 对象文件反序列化到内存，也便于将数据加载到设备中。
3) torch.nn.Module.load_state_dict()： 加载模型的参数。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy


# Define model
class TheModelClass(nn.Module):
    def __init__(self):
        super(TheModelClass, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize model
model = TheModelClass()

# Initialize optimizer
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Print model's state_dict
print("Model's state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())

# Print optimizer's state_dict
print("Optimizer's state_dict:")
for var_name in optimizer.state_dict():
    print(var_name, "\t", optimizer.state_dict()[var_name])

#%% 不再训练模型，如下：
# save state_dict
Para_PATH = "test/model_para.pt"
torch.save(model.state_dict(), Para_PATH)
print("Successfully save the parameters")

# load state_dict
model2 = TheModelClass()
model2.load_state_dict(torch.load(Para_PATH))
print("Successfully load the parameters")
print(model2)

# save model
Model_PATH = "test/model.pt"
torch.save(model, Model_PATH)
print("Successfully save the model")

# load model
model3 = torch.load(Model_PATH)
model3.eval()
print("Successfully load the model")

#%% 还需要训练模型，如下：
# 需要保存优化器状态、epoch和loss
# save
torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            ...
            }, PATH)

# load
model4 = TheModelClass()
optimizer = TheOptimizerClass()

checkpoint = torch.load(PATH)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']

model.eval()
# - or -
model.train()

#%% 将多个模型存入一个文件：注意命名
torch.save({
            'modelA_state_dict': modelA.state_dict(),
            'modelB_state_dict': modelB.state_dict(),
            'optimizerA_state_dict': optimizerA.state_dict(),
            'optimizerB_state_dict': optimizerB.state_dict(),
            ...
            }, PATH)

# 加载模型
modelA = TheModelAClass(*args, **kwargs)
modelB = TheModelBClass(*args, **kwargs)
optimizerA = TheOptimizerAClass(*args, **kwargs)
optimizerB = TheOptimizerBClass(*args, **kwargs)

checkpoint = torch.load(PATH)
modelA.load_state_dict(checkpoint['modelA_state_dict'])
modelB.load_state_dict(checkpoint['modelB_state_dict'])
optimizerA.load_state_dict(checkpoint['optimizerA_state_dict'])
optimizerB.load_state_dict(checkpoint['optimizerB_state_dict'])

modelA.eval()
modelB.eval()
# - or -
modelA.train()
modelB.train()

#%% 暖启动模型
torch.save(modelA.state_dict(), PATH)

modelB = TheModelBClass()
modelB.load_state_dict(torch.load(PATH), strict=False)
# 使用参数strict=False可以把 state_dict 能够匹配的 keys 加载进去，然后忽略无法匹配的keys


device = torch.device('cpu')
# - or -
device = torch.device("cuda:0")

model = TheModelClass()
model.load_state_dict(torch.load(PATH, map_location=device))
model.to(device)
# 使用参数map_location可以选择将模型(或state_dict)加载到GPU还是CPU
# 声明一个模型相当于将其在CPU上初始化，需要使用model.to加载进GPU
