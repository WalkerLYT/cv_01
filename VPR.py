import os
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split

# 设置设备（GPU 或 CPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 定义自定义数据集类
class LocationDataset(Dataset):
    def __init__(self, images_folder, annotations_file, transform=None):
        self.images_folder = images_folder  # 图像文件夹路径
        # 读取位置信息文件
        self.annotations = pd.read_csv(annotations_file, delim_whitespace=True, header=None,
                                       names=['image', 'latitude', 'longitude'])
        self.transform = transform  # 数据预处理方法

    def __len__(self):
        return len(self.annotations)  # 返回数据集的长度

    def __getitem__(self, idx):
        # 获取图像文件名
        img_name = os.path.join(self.images_folder, self.annotations.iloc[idx, 0])
        # 打开图像
        image = Image.open(img_name)
        # 获取对应的纬度和经度
        latitude = self.annotations.iloc[idx, 1]
        longitude = self.annotations.iloc[idx, 2]

        # 进行数据预处理
        if self.transform:
            image = self.transform(image)

        # 返回图像和位置信息
        return image, torch.tensor([latitude, longitude], dtype=torch.float32)


# 数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 调整图像大小
    transforms.ToTensor(),  # 转换为张量
    # 归一化：使用ImageNet的均值和标准差
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 加载自定义数据集
dataset = LocationDataset(images_folder='../College', annotations_file='../College.txt',
                          transform=transform)

# 计算分割大小
total_size = len(dataset)
val_size = int(0.1 * total_size)  # 10% 用作验证集
train_size = total_size - val_size  # 剩余部分用作训练集

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)  # 数据加载器
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)  # 验证数据加载器

# 模型文件路径
model_path = 'models/resnet34_VPR_College.pth'

# 加载预训练的 ResNet-34 模型，并修改输出层
resnet34 = models.resnet34(pretrained=True)
num_ftrs = resnet34.fc.in_features  # 获取全连接层输入特征数
resnet34.fc = nn.Linear(num_ftrs, 2)  # 修改为2个输出节点（纬度和经度）
resnet34.to(device)  # 将模型移到设备（GPU/CPU）

# 如果不存在模型则需要训练
if not os.path.exists(model_path):
    # 设置损失函数和优化器
    criterion = nn.MSELoss()  # 使用均方误差损失
    optimizer = optim.Adam(resnet34.parameters(), lr=0.001)  # 使用Adam优化器

    # 训练模型
    num_epochs = 50  # 训练轮数
    for epoch in range(num_epochs):
        resnet34.train()  # 切换到训练模式
        running_loss = 0.0  # 初始化损失

        for images, locations in train_dataloader:
            images, locations = images.to(device), locations.to(device)  # 将数据移到设备

            optimizer.zero_grad()  # 清零梯度
            outputs = resnet34(images)  # 前向传播
            loss = criterion(outputs, locations)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数

            running_loss += loss.item()  # 累加损失

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_dataloader):.4f}")  # 打印损失
        torch.save(resnet34.state_dict(), model_path)  # 保存模型权重

    print(f"Finished Training, Model saved to '{model_path}'")  # 训练完成


# 加载模型权重
resnet34.load_state_dict(torch.load(model_path))
resnet34.eval()  # 切换到评估模式

print(f"Loaded model from '{model_path}'")

# 评估模型
total_loss = 0.0  # 初始化总损失
num_successful = 0  # 成功预测的图像数量
total_images = 0  # 总图像数量

with torch.no_grad():  # 禁用梯度计算
    for images, locations in val_dataloader:
        images, locations = images.to(device), locations.to(device)  # 将数据移到设备
        outputs = resnet34(images)  # 前向传播，获取模型输出

        # 计算预测误差
        lat_error = torch.abs(outputs[:, 0] - locations[:, 0])
        lon_error = torch.abs(outputs[:, 1] - locations[:, 1])

        # 判断每张图像是否预测成功
        successful = (lat_error < 10) & (lon_error < 10)
        num_successful += successful.sum().item()  # 成功预测的图像数量
        total_images += images.size(0)  # 总图像数量

    success_rate = num_successful / total_images  # 计算成功比率
    print(f"Validation Success Rate: {success_rate:.4f}")  # 打印验证成功比率